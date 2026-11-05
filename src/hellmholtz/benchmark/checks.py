"""Deterministic response checks for benchmark prompts.

A prompt carries a list of check specs (plain dicts so they live in JSON/YAML
too). Each check yields a pass/fail; the auto score is the weighted share of
passed checks, so it needs no judge model and is reproducible.

Supported ``type`` values:

``contains`` / ``contains_any``  ``values`` (case-insensitive substrings)
``not_contains``                 ``values`` that must be absent
``regex`` / ``not_regex``        ``pattern`` (+ optional ``flags``: "i", "s", "m")
``final_number``                 ``value`` (+ ``tol``): last number in the answer
``json_valid``                   response (or a fenced block) parses as JSON;
                                 optional ``keys``, ``values`` ({path: expected}),
                                 ``lengths`` ({path: len}) and ``types`` ({path: type},
                                 one of string/integer/number/boolean/array/object/
                                 array_of_strings)
``max_words`` / ``min_words``    ``value``
``max_lines`` / ``min_lines``    ``value`` (non-empty lines)
``python_tests``                 run the extracted ``python`` block + ``tests`` code
                                 in a subprocess (``timeout`` seconds)
``sentence_count``               exactly ``value`` sentences
``max_sentence_words``           no sentence longer than ``value`` words
``ends_with``                    visible answer ends with ``value``
``language``                     response is predominantly ``de`` or ``en``

Text-matching checks (contains*, regex) run on normalised text: NFKC, straight quotes,
LaTeX markup stripped - so ``don’t`` / ``\\text{CO}_2`` match ``don't`` / ``CO2``.

Every spec accepts ``weight`` (default 1) and ``name`` (label for reports).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - used to run model-written code under a hard timeout
import sys
import tempfile
from typing import Any
import unicodedata

logger = logging.getLogger(__name__)

_MATCH_KINDS = {"contains", "contains_any", "not_contains", "regex", "not_regex"}
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(\w+)?[ \t]*\n(.*?)```", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FLAGS = {"i": re.IGNORECASE, "s": re.DOTALL, "m": re.MULTILINE}

_DE_WORDS = {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "für", "von", "zu"}
_EN_WORDS = {"the", "and", "is", "not", "a", "with", "for", "of", "to", "in", "that"}


@dataclass
class CheckResult:
    """Outcome of one check."""

    name: str
    passed: bool
    weight: float = 1.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "weight": self.weight,
            "detail": self.detail,
        }


def strip_reasoning(text: str) -> str:
    """Drop ``<think>`` blocks and a dangling unclosed ``<think>`` prefix."""
    text = _THINK_RE.sub("", text or "")
    if re.match(r"\s*<think(?:ing)?>", text, re.IGNORECASE):
        return ""  # unterminated reasoning: no visible answer was produced
    return text.strip()


def extract_code(text: str, lang: str | None = None) -> str:
    """Return the first fenced code block (preferring ``lang``), else the raw text."""
    blocks: list[tuple[str, str]] = _FENCE_RE.findall(text)
    aliases = {"python": {"python", "py", "python3"}}.get(lang or "", {lang} if lang else set())
    for block_lang, body in blocks:
        if block_lang.lower() in aliases:
            return body
    return blocks[0][1] if blocks else text


def normalize_for_match(text: str) -> str:
    """Make surface formatting irrelevant for substring/regex checks.

    NFKC folds sub/superscripts (CO₂ -> CO2) and typographic quotes; LaTeX markup
    that models emit around chemistry/maths (``\\text{CO}_2``) is reduced to plain text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(
        {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2212: "-"}
    )
    text = re.sub(r"\\(?:text|mathrm|mathbf|textbf|mathit|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\x?(?:long)?rightarrow(?:\{[^{}]*\})?|\u2192", "->", text)
    text = re.sub(r"_\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_(\w)", r"\1", text)
    text = re.sub(r"\\[\[\]()]|\$\$", "", text)
    return text


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


_FINAL_RE = re.compile(
    r"(?:final\s+answer|endergebnis|antwort)\s*(?:is|ist)?\s*[:=\-]?\s*[*_`$\\(\[]*\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def _last_number(text: str) -> float | None:
    """The number after a ``Final answer:`` marker, else the last number in the text."""
    marked = _FINAL_RE.findall(text)
    matches = [marked[-1]] if marked else _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _numbers(text: str) -> list[float]:
    out = []
    for raw in _NUMBER_RE.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def _load_json(text: str) -> Any:
    candidates = [text.strip()]
    candidates += [body.strip() for lang, body in _FENCE_RE.findall(text) if lang in ("", "json")]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError("no valid JSON found")


def _dig(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


_TYPE_TESTS: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "array_of_strings": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
}


_MEMORY_LIMIT = 1024 * 1024 * 1024  # address space for model-written code
_FILE_LIMIT = 10 * 1024 * 1024


def _limit_resources(cpu_seconds: int) -> Callable[[], None]:
    def apply() -> None:  # runs in the child between fork and exec
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT, _MEMORY_LIMIT))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FILE_LIMIT, _FILE_LIMIT))

    return apply


def _bwrap_command(workdir: str, script: str, python: str) -> list[str]:
    """bubblewrap invocation: no network, no home or repository, only a scratch dir."""
    cmd = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "PYTHONHASHSEED",
        "0",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",  # nosec B108 - private tmpfs inside the sandbox
        "--bind",
        workdir,
        "/work",
        "--chdir",
        "/work",
    ]
    mounted = {"/usr"}
    cmd += ["--ro-bind", "/usr", "/usr"]
    for link in ("/bin", "/lib", "/lib64", "/sbin"):
        if os.path.islink(link):
            cmd += ["--symlink", os.readlink(link), link]
        elif os.path.isdir(link):
            cmd += ["--ro-bind", link, link]
    for root in {sys.base_prefix, os.path.dirname(os.path.dirname(python))}:
        if root and not any(root == m or root.startswith(m + "/") for m in mounted):
            cmd += ["--ro-bind", root, root]
            mounted.add(root)
    return [*cmd, python, "-I", "/work/" + os.path.basename(script)]


_SANDBOX: dict[str, bool] = {}


def sandbox_available() -> bool:
    """True when bubblewrap can start an unprivileged, network-less sandbox (cached)."""
    if "bwrap" not in _SANDBOX:
        ok = False
        if shutil.which("bwrap"):
            python = os.path.realpath(sys.executable)
            with tempfile.TemporaryDirectory(prefix="hellm-probe-") as tmp:
                probe = os.path.join(tmp, "probe.py")
                with open(probe, "w", encoding="utf-8") as f:
                    f.write("pass\n")
                try:
                    ok = (
                        subprocess.run(  # noqa: S603  # nosec B603
                            _bwrap_command(tmp, probe, python),
                            capture_output=True,
                            timeout=20,
                        ).returncode
                        == 0
                    )
                except (OSError, subprocess.SubprocessError):
                    ok = False
        if not ok:
            logger.warning(
                "bubblewrap sandbox unavailable: python_tests run with resource limits only"
            )
        _SANDBOX["bwrap"] = ok
    return _SANDBOX["bwrap"]


def _run_python(code: str, tests: str, timeout: float) -> tuple[bool, str]:
    """Execute code + tests in an isolated interpreter; True when it exits 0.

    Model-written code runs inside bubblewrap when available: no network, no access to
    the home directory, repository or ``.env`` (only a scratch directory is visible).
    Without bubblewrap it still gets CPU, memory and file-size limits and a clean env.
    """
    with tempfile.TemporaryDirectory(prefix="hellm-check-") as tmp:
        script = os.path.join(tmp, "solution.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + tests + "\n")
        python = os.path.realpath(sys.executable)
        if sandbox_available():
            cmd, env, cwd = _bwrap_command(tmp, script, python), None, None
        else:
            cmd = [python, "-I", script]
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
            cwd = tmp
        try:
            proc = subprocess.run(  # noqa: S603  # nosec B603 - fixed interpreter, temp script
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
                preexec_fn=_limit_resources(int(timeout) + 2),  # noqa: PLW1509  # nosec B603
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
    if proc.returncode == 0:
        return True, "all assertions passed"
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:]
    return False, tail[0][:200] if tail else f"exit {proc.returncode}"


def _run_one(spec: dict[str, Any], text: str) -> tuple[bool, str]:  # noqa: C901
    kind = spec["type"]
    if kind in _MATCH_KINDS:
        text = normalize_for_match(text)
    lowered = text.lower()
    values = [str(v).lower() for v in spec.get("values", [])]

    if kind == "contains":
        missing = [v for v in values if v not in lowered]
        return not missing, f"missing: {missing}" if missing else "all present"
    if kind == "contains_any":
        hit = [v for v in values if v in lowered]
        return bool(hit), f"matched {hit}" if hit else f"none of {values}"
    if kind == "not_contains":
        found = [v for v in values if v in lowered]
        return not found, f"forbidden present: {found}" if found else "none present"
    if kind in ("regex", "not_regex"):
        flags = 0
        for ch in spec.get("flags", ""):
            flags |= _FLAGS.get(ch, 0)
        matched = re.search(spec["pattern"], text, flags) is not None
        return (
            matched if kind == "regex" else not matched
        ), f"/{spec['pattern']}/ found={matched}"
    if kind == "final_number":
        got = _last_number(text)
        want, tol = float(spec["value"]), float(spec.get("tol", 1e-6))
        if got is None:
            return False, "no number in answer"
        return abs(got - want) <= tol, f"got {got:g}, want {want:g}"
    if kind == "any_number":
        want, tol = float(spec["value"]), float(spec.get("tol", 1e-6))
        nums = _numbers(text)
        return any(abs(n - want) <= tol for n in nums), f"want {want:g} among {nums[-5:]}"
    if kind == "json_valid":
        try:
            data = _load_json(text)
        except ValueError as exc:
            return False, str(exc)
        for key in spec.get("keys", []):
            try:
                _dig(data, key)
            except (KeyError, IndexError, TypeError, ValueError):
                return False, f"missing key {key}"
        expected_values: dict[str, Any] = (
            spec["values"] if isinstance(spec.get("values"), dict) else {}
        )
        for path, expected in expected_values.items():
            try:
                got = _dig(data, path)
            except (KeyError, IndexError, TypeError, ValueError):
                return False, f"missing key {path}"
            if got != expected:
                return False, f"{path}={got!r}, want {expected!r}"
        lengths: dict[str, Any] = spec["lengths"] if isinstance(spec.get("lengths"), dict) else {}
        for path, want_len in lengths.items():
            try:
                got_len = len(_dig(data, path))
            except (KeyError, IndexError, TypeError, ValueError):
                return False, f"missing key {path}"
            if got_len != want_len:
                return False, f"len({path})={got_len}, want {want_len}"
        types: dict[str, Any] = spec["types"] if isinstance(spec.get("types"), dict) else {}
        for path, type_name in types.items():
            try:
                value = _dig(data, path)
            except (KeyError, IndexError, TypeError, ValueError):
                return False, f"missing key {path}"
            if not _TYPE_TESTS[type_name](value):
                return False, f"{path}={value!r} is not {type_name}"
        return True, "valid JSON"
    if kind in ("max_words", "min_words"):
        n = len(_words(text))
        limit = int(spec["value"])
        return (n <= limit if kind == "max_words" else n >= limit), f"{n} words (limit {limit})"
    if kind in ("max_lines", "min_lines"):
        n = len([ln for ln in text.splitlines() if ln.strip()])
        limit = int(spec["value"])
        return (n <= limit if kind == "max_lines" else n >= limit), f"{n} lines (limit {limit})"
    if kind in ("sentence_count", "max_sentence_words"):
        sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
        if kind == "sentence_count":
            return len(sentences) == int(spec["value"]), f"{len(sentences)} sentences"
        longest = max((len(_words(x)) for x in sentences), default=0)
        return longest <= int(spec["value"]), f"longest sentence {longest} words"
    if kind == "ends_with":
        return text.rstrip().endswith(str(spec["value"])), f"ends with {text.rstrip()[-1:]!r}"
    if kind == "python_tests":
        code = extract_code(text, "python")
        return _run_python(code, spec["tests"], float(spec.get("timeout", 10)))
    if kind == "language":
        words = [w.lower() for w in _words(text)]
        de = sum(w in _DE_WORDS for w in words)
        en = sum(w in _EN_WORDS for w in words)
        detected = "de" if de > en else "en"
        return detected == spec["value"], f"detected {detected} (de={de}, en={en})"
    raise ValueError(f"unknown check type: {kind}")


def run_checks(checks: list[dict[str, Any]], response: str | None) -> list[CheckResult]:
    """Run all checks against the visible part of ``response``."""
    text = strip_reasoning(response or "")
    results: list[CheckResult] = []
    for i, spec in enumerate(checks):
        name = spec.get("name") or f"{spec['type']}#{i + 1}"
        weight = float(spec.get("weight", 1))
        if not text:
            results.append(CheckResult(name, False, weight, "empty visible answer"))
            continue
        try:
            passed, detail = _run_one(spec, text)
        except Exception as exc:  # a broken spec must not crash a benchmark run
            passed, detail = False, f"check error: {exc}"
        results.append(CheckResult(name, bool(passed), weight, detail))
    return results


def auto_score(results: list[CheckResult]) -> float | None:
    """Weighted pass share in [0, 1]; None when the prompt has no checks."""
    total = sum(r.weight for r in results)
    if total <= 0:
        return None
    return sum(r.weight for r in results if r.passed) / total
