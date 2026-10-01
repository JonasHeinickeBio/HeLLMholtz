#!/usr/bin/env python
"""Run the repo's LLM-as-a-Judge evaluation over an existing benchmark results file.

The benchmark CLI only judges freshly produced runs (``--evaluate-with``); this
driver judges an existing JSONL results file in-place and is resumable: re-running
it skips records that already carry a rating and retries the ones that failed
(e.g. after a transient 502 from the judge endpoint).

Usage:
    .venv/bin/python scripts/run_llm_judge.py \
        results/benchmark_2026-09-30T23-29-27.138617.partial.jsonl \
        --model "blablador:Muse Glimmer 30b" --workers 4
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hellmholtz.benchmark.evaluator import JUDGE_PROMPT_TEMPLATE  # noqa: E402
from hellmholtz.benchmark.prompts import get_all_prompts  # noqa: E402
from hellmholtz.benchmark.runner import _retry_with_backoff  # noqa: E402
from hellmholtz.client import chat_raw  # noqa: E402

logger = logging.getLogger("run_llm_judge")

RATING_RE = re.compile(r"RATING:\s*(\d+(?:\.\d+)?)")
CRITIQUE_RE = re.compile(r"CRITIQUE:\s*(.*)", re.DOTALL)

DEFAULT_JUDGE_MODEL = "blablador:Muse Glimmer 30b"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line %d in %s: %s", line_no, path, e)
    return records


def save_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to path via a temp file + os.replace (crash-safe)."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def record_key(rec: dict[str, Any]) -> str:
    """Content hash of everything the judge sees plus its identity fields.

    Records that hash equal (same model, prompt, params and response text) get
    the same judgement, so restored judgements can be shared safely across them.
    """
    payload = {k: v for k, v in rec.items() if k not in ("rating", "critique")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def merge_existing_judgements(records: list[dict[str, Any]], out_path: Path) -> int:
    """Resume support: copy ratings/critiques from a previous output file."""
    if not out_path.exists():
        return 0
    existing = load_jsonl(out_path)
    by_key: dict[str, dict[str, Any]] = {}
    for rec in existing:
        if rec.get("rating") is not None:
            by_key[record_key(rec)] = rec
    restored = 0
    for rec in records:
        prev = by_key.get(record_key(rec))
        if prev is not None and rec.get("rating") is None:
            rec["rating"] = prev["rating"]
            rec["critique"] = prev.get("critique")
            restored += 1
    return restored


def _extract_verdict(text: str) -> tuple[float | None, str | None]:
    """Parse rating/critique, taking the last RATING match to ignore echoed prompts."""
    matches = list(RATING_RE.finditer(text))
    if not matches:
        return None, None
    last = matches[-1]
    critique_match = CRITIQUE_RE.search(text, last.start())
    critique = critique_match.group(1).strip() if critique_match else None
    return float(last.group(1)), critique


def judge_record(
    rec: dict[str, Any], prompt_text: str, judge_model: str, max_tokens: int
) -> tuple[float | None, str | None]:
    """Ask the judge for a rating/critique; returns (rating, critique) or (None, None).

    Reasoning-style judges sometimes return `content=None` with the verdict in
    `reasoning_content` (or nothing at all when max_tokens is consumed by
    thinking), so we fall back to that field and retry a few times per record.
    """
    evaluation_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt_text, response=rec.get("response_text") or ""
    )
    for attempt in range(3):
        response = _retry_with_backoff(
            lambda: chat_raw(
                model=judge_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        )
        text = ""
        if response is not None and getattr(response, "choices", None):
            message = response.choices[0].message
            text = message.content or ""
            if not RATING_RE.search(text):
                text = getattr(message, "reasoning_content", None) or text
        rating, critique = _extract_verdict(text)
        if rating is not None:
            return rating, critique
        logger.warning(
            "No RATING parsed for %s/%s (attempt %d) — response head: %r",
            rec.get("model"),
            rec.get("prompt_id"),
            attempt + 1,
            text[:120],
        )
    return None, None


def _judge_all(
    records: list[dict[str, Any]],
    todo: list[int],
    prompts: dict[str, str],
    judge_model: str,
    judge_max_tokens: int,
    workers: int,
    out_path: Path,
) -> int:
    """Judge the given record indices in parallel; periodically checkpoint to disk.

    Records with an identical content key (same model, params, prompt and response
    text) yield an identical judgement, so each unique key is sent to the judge
    once and the result is fanned out to all its records.
    """
    lock = threading.Lock()
    completed = 0
    judged_ok = 0
    save_interval = 10

    groups: dict[str, list[int]] = {}
    for idx in todo:
        groups.setdefault(record_key(records[idx]), []).append(idx)
    logger.info("%d unique judge inputs across %d records to judge", len(groups), len(todo))

    def work(indices: list[int]) -> int:
        rec = records[indices[0]]
        try:
            rating, critique = judge_record(
                rec, prompts[rec["prompt_id"]], judge_model, judge_max_tokens
            )
        except Exception as e:  # retries exhausted or unexpected error
            logger.error("Judging failed for %s/%s: %s", rec.get("model"), rec.get("prompt_id"), e)
            return 0
        if rating is None:
            return 0
        with lock:
            for i in indices:
                records[i]["rating"] = rating
                records[i]["critique"] = critique
        return len(indices)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, idxs) for idxs in groups.values()]
        for fut in as_completed(futures):
            with lock:
                completed += 1
                judged_ok += fut.result()
                if completed % save_interval == 0:
                    save_jsonl_atomic(out_path, records)
            logger.info(
                "progress %d/%d requests (records judged: %d)", completed, len(groups), judged_ok
            )
    return judged_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="Benchmark results JSONL file")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model identifier")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <results>.judged.jsonl, resumable)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel judge requests")
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--limit", type=int, default=None, help="Judge at most N records (smoke test)"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "aisuite", "hellmholtz.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.results.exists():
        parser.error(f"results file not found: {args.results}")
    out_path = args.output or args.results.parent / (args.results.stem + ".judged.jsonl")

    records = load_jsonl(args.results)
    restored = merge_existing_judgements(records, out_path)
    if restored:
        logger.info("Restored %d existing judgements from %s (resume)", restored, out_path)

    prompts = {p.id: p.user_message for p in get_all_prompts()}

    def needs_judging(rec: dict[str, Any]) -> bool:
        return (
            rec.get("rating") is None
            and bool(rec.get("success"))
            and bool(rec.get("response_text"))
            and rec.get("prompt_id") in prompts
        )

    todo = [i for i, rec in enumerate(records) if needs_judging(rec)]
    if args.limit is not None:
        todo = todo[: args.limit]
    skipped_no_prompt = sum(
        1 for rec in records if rec.get("rating") is None and rec.get("prompt_id") not in prompts
    )
    skipped_empty = sum(
        1
        for rec in records
        if rec.get("rating") is None
        and rec.get("prompt_id") in prompts
        and not rec.get("response_text")
    )
    logger.info(
        "Records: %d total | %d to judge | %d already judged | "
        "%d skipped (unknown prompt) | %d skipped (empty response)",
        len(records),
        len(todo),
        sum(1 for r in records if r.get("rating") is not None),
        skipped_no_prompt,
        skipped_empty,
    )
    if not todo:
        save_jsonl_atomic(out_path, records)
        logger.info("Nothing to do; wrote %s", out_path)
        return 0

    judged_ok = _judge_all(
        records, todo, prompts, args.model, args.judge_max_tokens, args.workers, out_path
    )

    save_jsonl_atomic(out_path, records)
    n_rated = sum(1 for r in records if r.get("rating") is not None)
    logger.info(
        "Done: %d newly judged this run, %d/%d records rated overall -> %s",
        judged_ok,
        n_rated,
        len(records),
        out_path,
    )
    return 0 if judged_ok + restored > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
