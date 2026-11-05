"""Combine deterministic checks and LLM-judge verdicts into one 0-100 score.

Design notes
------------
* A failed or empty run scores 0 - a model that errors on half the prompts must
  not outrank one that answers all of them. Reliability is reported separately.
* Prompts with checks blend ``CHECK_WEIGHT`` auto score with the judge score;
  open-ended prompts rely on the judge alone, verifiable ones lean on checks.
* The leaderboard macro-averages categories so a category with many prompts
  does not dominate, and averages replications/temperatures per prompt first.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any

# Judge dimension weights (each dimension is scored 1-5).
DIMENSION_WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "completeness": 0.20,
    "instruction_following": 0.25,
    "clarity": 0.15,
}
# Share of the judge score taken from the per-prompt rubric when one exists.
RUBRIC_SHARE = 0.4
# Share of the final score from deterministic checks when both signals exist.
CHECK_WEIGHT = 0.6


def judge_score(verdict: dict[str, Any]) -> float | None:
    """Map a parsed judge verdict to [0, 1]; None if it has no usable scores."""
    total = weight = 0.0
    for dim, w in DIMENSION_WEIGHTS.items():
        value = verdict.get(dim)
        if isinstance(value, (int, float)) and 1 <= value <= 5:
            total += w * (float(value) - 1) / 4
            weight += w
    if weight == 0:
        return None
    dims = total / weight
    rubric = verdict.get("rubric_met")
    if isinstance(rubric, list) and rubric and all(isinstance(r, bool) for r in rubric):
        return (1 - RUBRIC_SHARE) * dims + RUBRIC_SHARE * (sum(rubric) / len(rubric))
    return dims


def rating_from_score(score: float) -> float:
    """Legacy 1-10 rating for a [0, 1] judge score."""
    return round(1 + 9 * score, 2)


def final_score(
    auto: float | None, judge: float | None, success: bool = True, empty: bool = False
) -> float | None:
    """Final 0-100 score for one run; None while a needed signal is still missing."""
    if not success or empty:
        return 0.0
    if auto is not None and judge is not None:
        return round(100 * (CHECK_WEIGHT * auto + (1 - CHECK_WEIGHT) * judge), 2)
    if auto is not None:
        return round(100 * auto, 2)
    if judge is not None:
        return round(100 * judge, 2)
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pct(values: list[float]) -> float | None:
    return round(100 * sum(values) / len(values), 1) if values else None


def _log_scale(value: float, worst: float, best: float) -> float:
    """Map ``value`` onto 0-100 on a log axis between ``worst`` and ``best``."""
    frac = (math.log(value) - math.log(worst)) / (math.log(best) - math.log(worst))
    return 100.0 * min(1.0, max(0.0, frac))


def speed_score(ttft: float | None, tokens_per_sec: float | None) -> float | None:
    """0-100 speed rating: half time-to-first-token, half decode throughput.

    Log scales: 10 tok/s -> 0 and 200 tok/s -> 100; 10 s TTFT -> 0 and 0.3 s -> 100.
    With only one component available that component is used alone.
    """
    parts = []
    if tokens_per_sec and tokens_per_sec > 0:
        parts.append(_log_scale(tokens_per_sec, 10.0, 200.0))
    if ttft and ttft > 0:
        parts.append(_log_scale(ttft, 10.0, 0.3))
    return round(statistics.fmean(parts), 1) if parts else None


def _ci95(values: list[float]) -> float:
    """Half-width of a normal-approximation 95% CI of the mean (0 for n < 2)."""
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def model_key(record: dict[str, Any]) -> str:
    name = str(record.get("model", ""))
    return name.split(":", 1)[1] if name.startswith("blablador:") else name


def leaderboard(
    records: list[dict[str, Any]], categories: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Aggregate raw result records into ranked per-model rows.

    ``categories`` maps prompt id -> category for records that predate the
    ``category`` field. Unscored runs (final_score is None) are ignored for the
    score but still count toward coverage.
    """
    categories = categories or {}
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_model[model_key(rec)].append(rec)

    rows: list[dict[str, Any]] = []
    for model, all_recs in by_model.items():
        # Speed-probe runs measure speed only; they never enter the quality score.
        probes = [r for r in all_recs if r.get("category") == "speed"]
        recs = [r for r in all_recs if r.get("category") != "speed"]
        # per (category, prompt): average over replications / temperatures
        per_prompt: dict[tuple[str, str], list[float]] = defaultdict(list)
        for rec in recs:
            score = rec.get("final_score")
            if score is None:
                score = 0.0 if not rec.get("success") else None
            if score is None:
                continue
            cat = rec.get("category") or categories.get(rec.get("prompt_id", ""), "other")
            per_prompt[(cat, rec.get("prompt_id", ""))].append(float(score))

        cat_scores: dict[str, list[float]] = defaultdict(list)
        for (cat, _pid), vals in per_prompt.items():
            cat_scores[cat].append(sum(vals) / len(vals))
        cat_means = {c: round(sum(v) / len(v), 2) for c, v in cat_scores.items()}
        if not cat_means:
            continue
        flat = [s for v in cat_scores.values() for s in v]

        ok = [r for r in recs if r.get("success")]
        latencies = [r["latency_seconds"] for r in ok if r.get("latency_seconds") is not None]
        tps = [r["tokens_per_sec"] for r in ok if r.get("tokens_per_sec")]
        out_tok = [r["output_tokens"] for r in ok if r.get("output_tokens") is not None]
        auto = [r["auto_score"] for r in recs if r.get("auto_score") is not None]
        judged = [r["judge_score"] for r in recs if r.get("judge_score") is not None]
        probe_ok = [r for r in probes if r.get("success")]
        ttfts = [r["ttft_seconds"] for r in probe_ok if r.get("ttft_seconds")]
        probe_tps = [r["tokens_per_sec"] for r in probe_ok if r.get("tokens_per_sec")]
        ttft = statistics.median(ttfts) if ttfts else None
        decode_tps = (
            statistics.median(probe_tps)
            if probe_tps
            else (statistics.median(tps) if tps else None)
        )
        # Share of output spent thinking, over runs where the API revealed it.
        think = [r for r in ok if r.get("reasoning_tokens") is not None and r.get("output_tokens")]
        think_total = sum(r["reasoning_tokens"] for r in think)
        out_total = sum(r["output_tokens"] for r in think)
        rows.append(
            {
                "model": model,
                "score": round(sum(cat_means.values()) / len(cat_means), 2),
                "ci95": round(_ci95(flat), 2),
                "categories": cat_means,
                "reliability": round(100 * len(ok) / len(recs), 1),
                "runs": len(recs),
                "prompts": len(per_prompt),
                "auto_score": _pct(auto),
                "judge_score": _pct(judged),
                "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
                "tokens_per_sec": round(statistics.median(tps), 1) if tps else None,
                "ttft_s": round(ttft, 2) if ttft is not None else None,
                "decode_tps": round(decode_tps, 1) if decode_tps is not None else None,
                "speed_score": speed_score(ttft, decode_tps),
                "reasoning_share": round(100 * think_total / out_total, 1) if out_total else None,
                "truncated_pct": round(
                    100 * sum(r.get("finish_reason") == "length" for r in recs) / len(recs), 1
                ),
                "avg_output_tokens": round(_mean([float(t) for t in out_tok]) or 0)
                if out_tok
                else None,
            }
        )
    rows.sort(key=lambda r: (-r["score"], r["model"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows
