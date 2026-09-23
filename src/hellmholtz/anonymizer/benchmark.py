"""Benchmarking and prompt comparison for the anonymizer.

Provides:

* :func:`run_benchmark` - latency (p50/p95/mean), throughput, entity counts
  and validation outcomes over repeated runs,
* :func:`evaluate_accuracy` - entity recall and leak rate against a ground
  truth annotation,
* :func:`compare_prompts` - side-by-side evaluation of prompt variants on the
  same texts,
* report writers producing Markdown + JSON under ``reports/anonymizer/``.

All functions accept an injectable ``chat_fn`` (via the Anonymizer keyword
arguments) so benchmarks can be rehearsed offline with a fake model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
from typing import Any

from hellmholtz.anonymizer.anonymizer import DEFAULT_MODEL, Anonymizer
from hellmholtz.anonymizer.models import AnonymizationResult
from hellmholtz.anonymizer.prompts import PROMPTS, get_prompt

DEFAULT_REPORT_DIR = "reports/anonymizer"


@dataclass
class BenchmarkRun:
    """Aggregate of repeated anonymization runs over the same text(s)."""

    label: str
    total_chars: int
    n_iterations: int
    latencies_ms: list[float] = field(default_factory=list)
    model_reported_ms: list[float] = field(default_factory=list)
    entity_counts: list[int] = field(default_factory=list)
    validation_ok: list[bool] = field(default_factory=list)
    consistent: list[bool] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)

    @property
    def throughput_chars_per_s(self) -> float:
        if not self.latencies_ms or self.mean_ms <= 0:
            return 0.0
        return self.total_chars / (self.mean_ms / 1000.0)

    @property
    def pass_rate(self) -> float:
        if not self.validation_ok:
            return 0.0
        return sum(self.validation_ok) / len(self.validation_ok)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def is_consistent(result: AnonymizationResult) -> bool:
    """Check that every original value maps to exactly one replacement."""
    seen: dict[str, str] = {}
    for e in result.entities:
        key = e.original.casefold()
        if key in seen and seen[key] != e.replacement:
            return False
        seen[key] = e.replacement
    return True


@dataclass
class AccuracyReport:
    """Accuracy of one anonymization result against a ground truth."""

    expected_entities: int
    detected_entities: int
    leaked_entities: list[str]
    consistent: bool

    @property
    def recall(self) -> float:
        return self.detected_entities / self.expected_entities if self.expected_entities else 1.0

    @property
    def leak_rate(self) -> float:
        if not self.expected_entities:
            return 0.0
        return len(self.leaked_entities) / self.expected_entities


def evaluate_accuracy(
    result: AnonymizationResult,
    ground_truth: list[dict[str, str]],
) -> AccuracyReport:
    """Compare an anonymization result against a ground truth annotation.

    Args:
        result: The anonymization result.
        ground_truth: List of ``{"type": "PERSON", "value": "John Doe"}``
            entries for distinct PII values in the source text.

    Returns:
        AccuracyReport with entity recall and the values that leaked.
    """
    detected_cf = {e.original.casefold() for e in result.entities}
    text_cf = result.text.casefold()

    detected = 0
    leaked: list[str] = []
    for entry in ground_truth:
        value = entry.get("value", "")
        if not value:
            continue
        vcf = value.casefold()
        if vcf in detected_cf:
            detected += 1
        elif vcf in text_cf:
            leaked.append(value)

    return AccuracyReport(
        expected_entities=len(ground_truth),
        detected_entities=detected,
        leaked_entities=leaked,
        consistent=is_consistent(result),
    )


def run_benchmark(
    texts: dict[str, str],
    *,
    iterations: int = 3,
    model: str = DEFAULT_MODEL,
    prompt: str = "default",
    collect_results: dict[str, AnonymizationResult] | None = None,
    **anonymizer_kwargs: Any,
) -> dict[str, BenchmarkRun]:
    """Benchmark an anonymizer configuration over several texts.

    Args:
        texts: name -> document text.
        iterations: Runs per text.
        model: Model string.
        prompt: Prompt variant name.
        collect_results: Optional dict; if provided, the last result per text
            is stored here (for downstream accuracy scoring).
        **anonymizer_kwargs: Forwarded to the :class:`Anonymizer`
            constructor (``validate``, ``chat_fn``, ...).

    Returns:
        name -> BenchmarkRun.
    """
    runs: dict[str, BenchmarkRun] = {}
    for name, text in texts.items():
        run = BenchmarkRun(label=name, total_chars=len(text), n_iterations=iterations)
        result: AnonymizationResult | None = None
        for _ in range(iterations):
            anon = Anonymizer(model=model, prompt=prompt, **anonymizer_kwargs)
            result = anon.anonymize(text)
            run.latencies_ms.append(result.duration_ms)
            if result.model_reported_ms is not None:
                run.model_reported_ms.append(result.model_reported_ms)
            run.entity_counts.append(len(result.entities))
            run.validation_ok.append(bool(result.validation and result.validation.is_anonymous))
            run.consistent.append(is_consistent(result))
        if collect_results is not None and result is not None:
            collect_results[name] = result
        runs[name] = run
    return runs


def compare_prompts(
    texts: dict[str, str],
    prompt_names: list[str] | None = None,
    *,
    iterations: int = 1,
    model: str = DEFAULT_MODEL,
    ground_truth: dict[str, list[dict[str, str]]] | None = None,
    **anonymizer_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate several prompt variants on the same texts.

    Args:
        texts: name -> document text.
        prompt_names: Variants to compare; defaults to all known variants.
        iterations: Runs per (variant, text).
        model: Model string.
        ground_truth: Optional name -> ground-truth annotation list.
        **anonymizer_kwargs: Forwarded to the :class:`Anonymizer` constructor.

    Returns:
        variant -> metrics dict (latency, accuracy, validation, per-text
        breakdown).
    """
    names = prompt_names if prompt_names is not None else list(PROMPTS)
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        get_prompt(name)  # fail fast on unknown names
        collected: dict[str, AnonymizationResult] = {}
        runs = run_benchmark(
            texts,
            iterations=iterations,
            model=model,
            prompt=name,
            collect_results=collected,
            **anonymizer_kwargs,
        )
        per_text: dict[str, Any] = {}
        latencies: list[float] = []
        recalls: list[float] = []
        leak_rates: list[float] = []
        pass_flags: list[bool] = []
        for text_name, run in runs.items():
            latencies.extend(run.latencies_ms)
            per_text[text_name] = {
                "p50_ms": round(run.p50_ms, 1),
                "p95_ms": round(run.p95_ms, 1),
                "throughput_chars_per_s": round(run.throughput_chars_per_s, 1),
                "entity_counts": run.entity_counts,
                "pass_rate": round(run.pass_rate, 3),
            }
            pass_flags.extend(run.validation_ok)
            if ground_truth and text_name in ground_truth:
                acc = evaluate_accuracy(collected[text_name], ground_truth[text_name])
                recalls.append(acc.recall)
                leak_rates.append(acc.leak_rate)
                per_text[text_name].update(
                    {
                        "recall": round(acc.recall, 3),
                        "leaked": acc.leaked_entities,
                        "consistent": acc.consistent,
                    }
                )
        out[name] = {
            "description": PROMPTS[name].description,
            "footer": PROMPTS[name].footer,
            "n_runs": len(latencies),
            "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50_ms": round(_percentile(latencies, 50), 1),
            "p95_ms": round(_percentile(latencies, 95), 1),
            "validation_pass_rate": round(statistics.fmean(pass_flags), 3) if pass_flags else 0.0,
            "mean_recall": round(statistics.fmean(recalls), 3) if recalls else None,
            "mean_leak_rate": round(statistics.fmean(leak_rates), 3) if leak_rates else None,
            "per_text": per_text,
        }
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def write_reports(
    comparison: dict[str, dict[str, Any]],
    *,
    out_dir: str | Path = DEFAULT_REPORT_DIR,
    title: str = "Anonymizer prompt comparison",
) -> tuple[Path, Path]:
    """Write Markdown + JSON reports for a :func:`compare_prompts` result.

    Args:
        comparison: Output of :func:`compare_prompts`.
        out_dir: Target directory (created if missing).
        title: Report heading.

    Returns:
        (markdown_path, json_path)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    md_path = out / f"prompt-comparison-{stamp}.md"
    json_path = out / f"prompt-comparison-{stamp}.json"

    json_path.write_text(
        json.dumps(_to_jsonable(comparison), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"_Generated: {datetime.now(UTC).isoformat(timespec='seconds')}_")
    lines.append("")
    header = (
        "| Variant | p50 (ms) | p95 (ms) | mean (ms) | Pass rate | Recall | Leak rate | Footer |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 8)
    for name, m in comparison.items():
        recall = m.get("mean_recall")
        leak = m.get("mean_leak_rate")
        lines.append(
            f"| {name} "
            f"| {m['p50_ms']} | {m['p95_ms']} | {m['mean_ms']} "
            f"| {m['validation_pass_rate']:.1%} "
            f"| {recall if recall is not None else '-'} "
            f"| {leak if leak is not None else '-'} "
            f"| {'yes' if m['footer'] else 'no'} |"
        )
    lines.append("")
    for name, m in comparison.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"_{m['description']}_")
        lines.append("")
        for text_name, tm in m["per_text"].items():
            detail = (
                f"- **{text_name}**: p50 {tm['p50_ms']} ms, "
                f"p95 {tm['p95_ms']} ms, "
                f"{tm['throughput_chars_per_s']} chars/s, "
                f"entities {tm['entity_counts']}, "
                f"pass {tm['pass_rate']:.0%}"
            )
            if "recall" in tm:
                extra = f", recall {tm['recall']}"
                if tm.get("leaked"):
                    extra += f", leaked: {', '.join(tm['leaked'])}"
                detail += extra
            lines.append(detail)
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path
