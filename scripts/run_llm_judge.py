#!/usr/bin/env python
"""Score and judge an existing benchmark results file (JSON list or .partial.jsonl).

Re-runs the deterministic checks against the *current* prompt suite, then asks the
judge panel to grade every open-ended response that has no judgement yet. The output
is resumable: judged records are checkpointed, and re-running skips them and retries
failures. A judge never grades its own model.

Usage:
    .venv/bin/python -P scripts/run_llm_judge.py results/benchmark_X.partial.jsonl
    .venv/bin/python -P scripts/run_llm_judge.py results/benchmark_X.json \
        --judges blablador:alias-huge,blablador:alias-large --workers 8
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hellmholtz.benchmark.evaluator import DEFAULT_JUDGES, evaluate_responses  # noqa: E402
from hellmholtz.benchmark.prompts import get_all_prompts  # noqa: E402
from hellmholtz.benchmark.runner import BenchmarkResult, _score_with_checks  # noqa: E402
from hellmholtz.benchmark.scoring import dedupe_attempts  # noqa: E402

logger = logging.getLogger("run_llm_judge")
_FIELDS = {f.name for f in fields(BenchmarkResult)}


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return list(json.loads(text))
    out = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed line %d: %s", line_no, exc)
    return dedupe_attempts(out)


def save_atomic(path: Path, results: list[BenchmarkResult]) -> None:
    """Write results as a JSON list via temp file + rename (crash-safe)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--judges", default=",".join(DEFAULT_JUDGES), help="Comma-separated panel")
    parser.add_argument("--output", type=Path, default=None, help="default: <results>.judged.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Judge at most N results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "aisuite", "hellmholtz.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_path = args.output or args.results.with_name(args.results.stem + ".judged.json")
    source = out_path if out_path.exists() else args.results  # resume from prior output
    records = load_records(source)
    results = [BenchmarkResult(**{k: v for k, v in r.items() if k in _FIELDS}) for r in records]

    prompts = get_all_prompts()
    by_id = {p.id: p for p in prompts}
    for r in results:  # (re)run deterministic checks against the current suite
        prompt = by_id.get(r.prompt_id)
        if prompt is not None:
            r.category = prompt.category
            if r.judge_score is None:
                _score_with_checks(r, prompt)

    todo = results
    if args.limit is not None:
        pending = [r for r in results if r.judge_score is None and r.success]
        skip = {id(r) for r in pending[args.limit :]}
        todo = [r for r in results if id(r) not in skip]

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    evaluate_responses(
        todo,
        judges,
        prompts,
        workers=args.workers,
        checkpoint=lambda: save_atomic(out_path, results),
    )
    save_atomic(out_path, results)

    judged = sum(r.judge_score is not None for r in results)
    scored = sum(r.final_score is not None for r in results)
    logger.info("%d results | %d judged | %d with a final score -> %s", len(results), judged, scored, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
