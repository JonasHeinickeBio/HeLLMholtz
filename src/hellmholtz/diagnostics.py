"""Connectivity and configuration diagnostics for ``hellm doctor``.

Pure check functions return :class:`CheckResult` objects so the CLI layer
only has to format them and unit tests only have to inject fakes. No check
ever raises; failures are reported as ``WARN``/``FAIL`` results instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import urllib.error
import urllib.request

__all__ = [
    "CheckResult",
    "DEFAULT_TIMEOUT",
    "api_paths",
    "check_chat",
    "check_credentials",
    "check_endpoint",
    "check_mcp_extra",
    "check_model_available",
    "check_python",
    "check_version",
    "probe_model_name",
]

DEFAULT_TIMEOUT = 10.0


def api_paths(base_url: str) -> tuple[str, str]:
    """Return ``(models_url, chat_url)`` for an OpenAI-compatible base URL.

    OpenAI-style base URLs end in ``/v1`` (LiteLLM, Blablador); when the
    version segment is absent it is inserted, mirroring SDK behaviour.
    """
    root = base_url.rstrip("/")
    prefix = root if root.endswith("/v1") else f"{root}/v1"
    return f"{prefix}/models", f"{prefix}/chat/completions"


def probe_model_name(model: str) -> str:
    """Bare model id sent in API requests (Blablador serves ``alias-x`` ids).

    The OpenAI-style ``provider:model`` prefix understood by aisuite is not
    part of the upstream model id, so it is stripped before probing.
    """
    return model.split(":", 1)[1] if ":" in model else model


def check_model_available(
    models_body: str,
    model: str,
) -> CheckResult:
    """Check that *model* appears in a ``/models`` listing body.

    Args:
        models_body: Raw JSON text of the ``/models`` response.
        model: Full model identifier (``provider:name``); prefix is stripped.
    """
    bare = probe_model_name(model)
    try:
        payload = json.loads(models_body)
        ids = [entry.get("id", "") for entry in payload.get("data", [])]
    except (ValueError, AttributeError, TypeError):
        return CheckResult("model", False, "could not parse model list", warn=True)
    if bare in ids:
        return CheckResult("model", True, f"'{bare}' is served by the endpoint")
    suggestions = [i for i in ids if bare in i or i in bare][:3]
    hint = f" - similar: {', '.join(suggestions)}" if suggestions else ""
    return CheckResult(
        "model",
        False,
        f"'{bare}' not in endpoint's model list ({len(ids)} served){hint}",
        warn=True,
    )


# A short chat body used only to prove authenticated reachability.
_PING_BODY = b'{"model": "%s", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}'


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single diagnostic check.

    Attributes:
        name: Human-readable check name.
        ok: Whether the check passed.
        detail: Short human-readable detail (never contains secrets).
        warn: If True a failure is downgraded to a warning (optional feature).
    """

    name: str
    ok: bool
    detail: str
    warn: bool = False

    @property
    def status(self) -> str:
        """One-char status used by the CLI renderer."""
        if self.ok:
            return "OK"
        return "WARN" if self.warn else "FAIL"


def check_python(min_version: tuple[int, int] = (3, 12)) -> CheckResult:
    """Check the running Python interpreter version."""
    import sys

    got = ".".join(map(str, sys.version_info[:2]))
    needed = ".".join(map(str, min_version))
    ok = sys.version_info[:2] >= min_version
    return CheckResult("python", ok, f"{got} (requires >= {needed})")


def check_version() -> CheckResult:
    """Check the installed hellmholtz version metadata."""
    try:
        ver = version("hellmholtz")
    except PackageNotFoundError:
        return CheckResult("hellmholtz package", False, "not installed (source checkout?)")
    return CheckResult("hellmholtz package", True, ver)


def check_mcp_extra() -> CheckResult:
    """Check whether the optional MCP extra is importable."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        return CheckResult(
            "mcp extra",
            False,
            "not installed (optional: pip install 'hellmholtz[mcp]')",
            warn=True,
        )
    return CheckResult("mcp extra", True, "installed")


def check_credentials(api_key: str | None) -> CheckResult:
    """Check that a Blablador API key is configured (value is never echoed)."""
    if api_key:
        return CheckResult("credentials", True, f"API key set ({_mask(api_key)})")
    return CheckResult(
        "credentials", False, "no API key (set BLABLADOR_API_KEY or run: hellm setup)"
    )


def check_endpoint(
    base_url: str | None,
    opener: Callable[[str, float], str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> CheckResult:
    """Check that the API base URL is set and reachable.

    Args:
        base_url: Configured API base URL.
        opener: ``callable(url, timeout) -> body`` hook for testing.
        timeout: Network timeout in seconds.
        api_key: Optional key sent as bearer auth on the ``/models`` probe.
    """
    if not base_url:
        return CheckResult("endpoint", False, "no base URL (set BLABLADOR_API_BASE)")
    url, _ = api_paths(base_url)
    get = opener or (
        lambda u, t: (
            _http_get_with_headers(u, t, {"Authorization": f"Bearer {api_key}"})
            if api_key
            else _http_get(u, t)
        )
    )
    try:
        get(url, timeout)
    except urllib.error.HTTPError as exc:
        return CheckResult("endpoint", False, f"{url} answered HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return CheckResult("endpoint", False, f"{url} unreachable: {_brief(exc)}")
    return CheckResult("endpoint", True, f"{url} reachable")


def check_chat(
    api_key: str | None,
    base_url: str | None,
    model: str,
    poster: Callable[[str, dict[str, str], str, float], str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> CheckResult:
    """Send a minimal chat request to prove end-to-end authenticated access.

    Args:
        api_key: API key for the ``Authorization`` header.
        base_url: Configured API base URL.
        model: Full model identifier (``provider:name``).
        poster: ``callable(url, headers, body, timeout) -> text`` hook for testing.
        timeout: Network timeout in seconds.
    """
    if not api_key or not base_url:
        return CheckResult("chat round-trip", True, "skipped (missing credentials)", warn=True)
    _, url = api_paths(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = _PING_BODY % probe_model_name(model).encode()
    try:
        (poster or _http_post_json)(url, headers, body.decode(), timeout)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return CheckResult("chat round-trip", False, f"HTTP {exc.code} (check API key)")
        if exc.code == 404:
            # Correct shape + auth, unknown model id: warn, do not fail.
            return CheckResult(
                "chat round-trip", False, "HTTP 404 (model id not found upstream)", warn=True
            )
        return CheckResult("chat round-trip", True, f"endpoint answered (HTTP {exc.code})")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return CheckResult("chat round-trip", False, _brief(exc))
    return CheckResult("chat round-trip", True, "model responded")


def _http_get(url: str, timeout: float) -> str:
    return _http_get_with_headers(url, timeout)


def _http_get_with_headers(url: str, timeout: float, headers: dict[str, str] | None = None) -> str:
    req = urllib.request.Request(  # nosec B310
        url, headers={"User-Agent": "hellm-doctor", **(headers or {})}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        data: bytes = resp.read()
        return data.decode("utf-8", errors="replace")


def _http_post_json(url: str, headers: dict[str, str], body: str, timeout: float) -> str:
    req = urllib.request.Request(  # nosec B310
        url, data=body.encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        data: bytes = resp.read()
        return data.decode("utf-8", errors="replace")


def _mask(secret: str) -> str:
    """Show enough of a secret to recognise it, never the whole value."""
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-3:]}"


def _brief(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
