"""Reliability benchmarking for the System-One (Jev/Laya) decision endpoint.

System-One models have **no chat surface**, so the chat-based benchmark runner
does not apply. Instead this module exercises the typed
:func:`~hellmholtz.providers.systemone.route` call against a fixed set of
clinical decision scenarios and measures *reliability* rather than clinical
correctness:

* success rate and latency (mean / p50 / p95) per scenario,
* decision stability — whether repeated calls pick the same choice,
* mean answer confidence, and
* routing metadata (model / repo / detected language).

All functions accept an injectable ``route_fn`` so benchmarks can be rehearsed
offline with a fake client (mirroring :mod:`hellmholtz.anonymizer.benchmark`).
A single failing call is counted as an error, not treated as fatal, so a flaky
endpoint still yields a usable stability/latency picture. Reports are written
as Markdown + JSON under ``reports/systemone/``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any

from hellmholtz.providers.systemone import (
    DEFAULT_SYSTEMONE_MODEL,
    SystemOneError,
    SystemOneQuestion,
    SystemOneResponse,
    get_endpoint,
    route,
)

__all__ = [
    "DEFAULT_REPORT_DIR",
    "DEFAULT_SCENARIOS",
    "DecisionScenario",
    "QuestionStats",
    "RouteFn",
    "ScenarioBenchmark",
    "SystemOneBenchmarkReport",
    "format_summary",
    "load_scenarios",
    "run_systemone_benchmark",
    "write_reports",
]

#: Default output directory for benchmark reports.
DEFAULT_REPORT_DIR = "reports/systemone"

#: A single routed call. ``route_fn`` mirrors the signature of
#: :func:`~hellmholtz.providers.systemone.route`; tests substitute a fake that
#: accepts ``(state, questions, *, model, timeout)``.
RouteFn = Callable[..., SystemOneResponse]


@dataclass
class DecisionScenario:
    """A named decision scenario: a natural-language state plus typed questions."""

    name: str
    state: str
    questions: dict[str, SystemOneQuestion] = field(default_factory=dict)


@dataclass
class QuestionStats:
    """Reliability stats for one question, aggregated over all replications."""

    name: str
    instructions: str
    choice_counts: dict[str, int]
    attempts: int = 0
    no_answer: int = 0
    confidences: list[float] = field(default_factory=list)

    @property
    def majority_choice(self) -> str | None:
        """The most frequently chosen option (deterministic on ties)."""
        if not self.choice_counts:
            return None
        return max(sorted(self.choice_counts), key=lambda k: self.choice_counts[k])

    @property
    def stability(self) -> float:
        """Fraction of recorded answers that picked the majority choice (0..1)."""
        total = sum(self.choice_counts.values())
        if not total:
            return 0.0
        best = max(sorted(self.choice_counts), key=lambda k: self.choice_counts[k])
        return self.choice_counts[best] / total

    @property
    def answer_rate(self) -> float:
        """Fraction of attempts that produced a non-null choice (0..1)."""
        if not self.attempts:
            return 0.0
        return (self.attempts - self.no_answer) / self.attempts

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(self.confidences) if self.confidences else 0.0


@dataclass
class ScenarioBenchmark:
    """Reliability results for one scenario across all replications."""

    name: str
    state: str
    n_questions: int
    replications: int
    attempts: int
    successful: int
    errors: int
    latencies_ms: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    questions: list[QuestionStats] = field(default_factory=list)
    routing_model: str = ""
    routing_repo: str = ""
    routing_language: str = ""

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
    def success_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return self.successful / self.attempts

    @property
    def decisions_per_s(self) -> float:
        """Total questions answered divided by total wall time of the calls."""
        answered = sum(q.attempts - q.no_answer for q in self.questions)
        total_ms = sum(self.latencies_ms)
        if answered <= 0 or total_ms <= 0:
            return 0.0
        return answered / (total_ms / 1000.0)

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(self.confidences) if self.confidences else 0.0

    @property
    def mean_stability(self) -> float:
        stabilities = [q.stability for q in self.questions]
        return statistics.fmean(stabilities) if stabilities else 0.0

    @property
    def stable(self) -> bool:
        """True when every question picked the same choice every time."""
        return bool(self.questions) and all(q.stability >= 1.0 for q in self.questions)


@dataclass
class SystemOneBenchmarkReport:
    """Aggregate reliability report across all benchmarked scenarios."""

    model: str
    endpoint: str
    replications: int
    generated_at: str
    scenarios: list[ScenarioBenchmark] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return sum(s.attempts for s in self.scenarios)

    @property
    def total_successes(self) -> int:
        return sum(s.successful for s in self.scenarios)

    @property
    def overall_success_rate(self) -> float:
        if not self.total_attempts:
            return 0.0
        return self.total_successes / self.total_attempts

    @property
    def overall_stability(self) -> float:
        vals = [q.stability for s in self.scenarios for q in s.questions]
        return statistics.fmean(vals) if vals else 0.0

    @property
    def all_stable(self) -> bool:
        return bool(self.scenarios) and all(s.stable for s in self.scenarios)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the report."""
        return asdict(self)


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile of ``values`` (linear interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


#: Built-in clinical decision scenarios used when no scenario file is given.
DEFAULT_SCENARIOS: list[DecisionScenario] = [
    DecisionScenario(
        name="fatigue_triage",
        state=(
            "Adult patient with persistent, severe post-exertional fatigue for "
            "over 12 months, not explained by another condition, presenting for "
            "an initial diagnostic work-up."
        ),
        questions={
            "post_exertional_malaise": SystemOneQuestion(
                type="choice",
                instructions="Does the history strongly suggest post-exertional malaise?",
                criteria={
                    "yes": "Clear post-exertional malaise present",
                    "no": "No clear malaise",
                },
            )
        },
    ),
    DecisionScenario(
        name="antibiotic_stewardship",
        state=(
            "Patient with a suspected viral prodrome and low-grade fever who is "
            "requesting antibiotics at the visit."
        ),
        questions={
            "antibiotics": SystemOneQuestion(
                type="choice",
                instructions="Should antibiotics be prescribed at this time?",
                criteria={
                    "yes": "Start antibiotics now",
                    "hold": "Defer antibiotics and monitor",
                    "refer": "Refer for culture before deciding",
                },
            )
        },
    ),
    DecisionScenario(
        name="referral_urgency",
        state=(
            "Patient with worsening neurological symptoms, including brain fog "
            "and orthostatic intolerance."
        ),
        questions={
            "neurology": SystemOneQuestion(
                type="choice",
                instructions="Is a neurology referral warranted, and how urgent?",
                criteria={"urgent": "Refer within days", "routine": "Refer routinely"},
            ),
            "followup": SystemOneQuestion(
                type="choice",
                instructions="How soon should the next follow-up happen?",
                criteria={"week": "Within one week", "month": "Within one month"},
            ),
        },
    ),
    DecisionScenario(
        name="followup_frequency",
        state="Stable ME/CFS patient on a graded management plan, reviewing the care pathway.",
        questions={
            "frequency": SystemOneQuestion(
                type="choice",
                instructions="What follow-up cadence is appropriate?",
                criteria={
                    "monthly": "Monthly",
                    "quarterly": "Quarterly",
                    "as_needed": "As needed",
                },
            )
        },
    ),
]


def _parse_question(name: str, spec: Any) -> SystemOneQuestion:
    """Build a :class:`SystemOneQuestion` from its JSON spec.

    Args:
        name: Question name (used in error messages).
        spec: Raw JSON object for the question.

    Returns:
        The parsed question.

    Raises:
        ValueError: If ``spec`` is not a JSON object.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Question '{name}' must be a JSON object.")
    raw_criteria = spec.get("criteria") or {}
    return SystemOneQuestion(
        type=str(spec.get("type", "choice")),
        instructions=str(spec.get("instructions", "")),
        criteria={str(k): str(v) for k, v in raw_criteria.items()},
    )


def _parse_scenario(index: int, item: Any) -> DecisionScenario:
    """Build a :class:`DecisionScenario` from its JSON object.

    Args:
        index: Position in the list (used in error messages and name default).
        item: Raw JSON object for the scenario.

    Returns:
        The parsed scenario.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    if not isinstance(item, dict):
        raise ValueError(f"Scenario #{index} must be a JSON object.")
    name = str(item.get("name") or f"scenario_{index}")
    state = str(item.get("state", ""))
    if not state.strip():
        raise ValueError(f"Scenario '{name}' is missing a non-empty 'state'.")
    raw_questions = item.get("questions") or {}
    if not isinstance(raw_questions, dict):
        raise ValueError(f"Scenario '{name}' has invalid 'questions'.")
    questions = {
        str(qname): _parse_question(str(qname), spec) for qname, spec in raw_questions.items()
    }
    if not questions:
        raise ValueError(f"Scenario '{name}' defines no questions.")
    return DecisionScenario(name=name, state=state, questions=questions)


def load_scenarios(path: str | Path) -> list[DecisionScenario]:
    """Load decision scenarios from a JSON file.

    The file must contain a JSON list of objects, each with ``name``, ``state``
    and ``questions`` (a mapping of question name to an object with ``type``,
    ``instructions`` and ``criteria``).

    Args:
        path: Path to the JSON file.

    Returns:
        List of parsed :class:`DecisionScenario`.

    Raises:
        ValueError: If the file is missing or malformed.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Scenarios file not found: {p}")
    try:
        data: Any = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read scenarios file {p}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Scenarios file must contain a JSON list.")
    return [_parse_scenario(i, item) for i, item in enumerate(data)]


def _run_scenario(
    scenario: DecisionScenario,
    *,
    model: str,
    replications: int,
    route_fn: RouteFn,
    timeout: float | None,
) -> ScenarioBenchmark:
    """Run one scenario ``replications`` times and aggregate the results."""
    bench = ScenarioBenchmark(
        name=scenario.name,
        state=scenario.state,
        n_questions=len(scenario.questions),
        replications=replications,
        attempts=0,
        successful=0,
        errors=0,
        questions=[
            QuestionStats(name=qname, instructions=q.instructions, choice_counts={})
            for qname, q in scenario.questions.items()
        ],
    )
    stats_by_name = {q.name: q for q in bench.questions}

    for _ in range(replications):
        bench.attempts += 1
        start = time.perf_counter()
        try:
            response = route_fn(
                scenario.state,
                scenario.questions,
                model=model,
                timeout=timeout,
            )
        except SystemOneError:
            bench.errors += 1
            for q in bench.questions:
                q.attempts += 1
                q.no_answer += 1
            time.perf_counter()  # drain the clock; the call is not timed
            continue
        bench.latencies_ms.append((time.perf_counter() - start) * 1000.0)
        bench.successful += 1
        bench.routing_model = response.routing.model or bench.routing_model
        bench.routing_repo = response.routing.repo or bench.routing_repo
        bench.routing_language = response.routing.detection.language or bench.routing_language

        for qname, _question in scenario.questions.items():
            stats = stats_by_name[qname]
            stats.attempts += 1
            answer = response.answers.get(qname)
            if answer is None or answer.choice is None:
                stats.no_answer += 1
                continue
            stats.choice_counts[answer.choice] = stats.choice_counts.get(answer.choice, 0) + 1
            if answer.confidence is not None:
                stats.confidences.append(answer.confidence)
                bench.confidences.append(answer.confidence)
    return bench


def run_systemone_benchmark(
    scenarios: list[DecisionScenario],
    *,
    model: str = DEFAULT_SYSTEMONE_MODEL,
    replications: int = 3,
    route_fn: RouteFn = route,
    timeout: float | None = None,
    endpoint: str | None = None,
    progress: bool = True,
) -> SystemOneBenchmarkReport:
    """Run the System-One reliability benchmark over a set of scenarios.

    Each scenario is routed ``replications`` times; per-question answers are
    aggregated into stability / confidence / latency statistics.

    Args:
        scenarios: The decision scenarios to benchmark.
        model: System-One model to route to.
        replications: Number of identical runs per scenario.
        route_fn: Injectable routing function (default: the real
            :func:`~hellmholtz.providers.systemone.route`). Tests substitute a
            fake client here.
        timeout: Per-request timeout in seconds (default: configured timeout).
        endpoint: Endpoint label recorded in the report; defaults to the
            resolved endpoint when not given.
        progress: Whether to log per-scenario progress.

    Returns:
        A :class:`SystemOneBenchmarkReport` with per-scenario metrics.
    """
    import logging

    log = logging.getLogger(__name__)
    if endpoint is None:
        endpoint = get_endpoint()
    report = SystemOneBenchmarkReport(
        model=model,
        endpoint=endpoint,
        replications=replications,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    for scenario in scenarios:
        if progress:
            log.info("Benchmarking scenario %s (%d runs)...", scenario.name, replications)
        report.scenarios.append(
            _run_scenario(
                scenario,
                model=model,
                replications=replications,
                route_fn=route_fn,
                timeout=timeout,
            )
        )
        if progress:
            last = report.scenarios[-1]
            log.info(
                "  %s: success %.0f%%, mean %.0f ms, stability %.2f",
                scenario.name,
                100.0 * last.success_rate,
                last.mean_ms,
                last.mean_stability,
            )
    return report


def format_summary(report: SystemOneBenchmarkReport) -> str:
    """Format a benchmark report as a human-readable summary string.

    Args:
        report: The benchmark report to summarize.

    Returns:
        A multi-line plain-text summary (no network, no filesystem).
    """
    lines: list[str] = ["System-One reliability benchmark"]
    lines.append(f"  model:       {report.model}")
    lines.append(f"  endpoint:    {report.endpoint}")
    lines.append(f"  replications: {report.replications}")
    lines.append(f"  scenarios:   {len(report.scenarios)}")
    lines.append(
        f"  success:     {report.overall_success_rate:.1%} "
        f"({report.total_successes}/{report.total_attempts})"
    )
    lines.append(f"  stability:   {report.overall_stability:.2f}")
    all_lat = [ms for s in report.scenarios for ms in s.latencies_ms]
    if all_lat:
        lines.append(
            f"  latency:     mean {statistics.fmean(all_lat):.0f} ms, "
            f"p95 {_percentile(all_lat, 95):.0f} ms"
        )
    lines.append("")
    for s in report.scenarios:
        lines.append(f"  {s.name}")
        lines.append(
            f"    success {s.success_rate:.0%} ({s.successful}/{s.attempts}), "
            f"mean {s.mean_ms:.0f} ms, p95 {s.p95_ms:.0f} ms, "
            f"{s.decisions_per_s:.1f} decisions/s, "
            f"stability {s.mean_stability:.2f}, confidence {s.mean_confidence:.2f}"
        )
        for q in s.questions:
            majority = q.majority_choice or "?"
            counts = ", ".join(f"{k}={v}" for k, v in sorted(q.choice_counts.items()))
            counts = counts or "no answer"
            lines.append(f"      {q.name}: {majority} [{counts}] stability {q.stability:.2f}")
    return "\n".join(lines)


def _render_markdown(report: SystemOneBenchmarkReport, title: str) -> str:
    """Render the report as a Markdown document string."""
    all_lat = [ms for s in report.scenarios for ms in s.latencies_ms]
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"_Generated: {report.generated_at}_")
    lines.append("")
    lines.append(f"- **Model:** {report.model}")
    lines.append(f"- **Endpoint:** {report.endpoint}")
    lines.append(f"- **Replications per scenario:** {report.replications}")
    lines.append(
        f"- **Overall success:** {report.overall_success_rate:.1%} "
        f"({report.total_successes}/{report.total_attempts})"
    )
    lines.append(f"- **Overall decision stability:** {report.overall_stability:.2f}")
    if all_lat:
        lines.append(
            f"- **Latency (all calls):** mean {statistics.fmean(all_lat):.0f} ms, "
            f"p95 {_percentile(all_lat, 95):.0f} ms"
        )
    lines.append("")
    lines.append("## Scenarios")
    lines.append("")
    lines.append("| Scenario | Success | Mean (ms) | p95 | Stability | Confidence | Routing |")
    lines.append("|" + "---|" * 7)
    for s in report.scenarios:
        routing = s.routing_model or s.routing_repo or "-"
        lines.append(
            f"| {s.name} "
            f"| {s.success_rate:.1%} ({s.successful}/{s.attempts}) "
            f"| {s.mean_ms:.0f} | {s.p95_ms:.0f} "
            f"| {s.mean_stability:.2f} | {s.mean_confidence:.2f} "
            f"| {routing} |"
        )
    lines.append("")
    for s in report.scenarios:
        lines.append(f"### {s.name}")
        lines.append("")
        lines.append(f"_State: {s.state}_")
        lines.append("")
        for q in s.questions:
            majority = q.majority_choice or "-"
            counts = (
                ", ".join(f"{k}={v}" for k, v in sorted(q.choice_counts.items())) or "no answer"
            )
            lines.append(
                f"- **{q.name}** — {q.instructions}: majority **{majority}**, "
                f"stability {q.stability:.2f}, answer rate {q.answer_rate:.0%}, counts: {counts}"
            )
        lines.append("")
    return "\n".join(lines)


def write_reports(
    report: SystemOneBenchmarkReport,
    *,
    out_dir: str | Path = DEFAULT_REPORT_DIR,
    title: str = "System-One reliability benchmark",
) -> tuple[Path, Path]:
    """Write Markdown + JSON reports for a benchmark report.

    Args:
        report: The benchmark report to persist.
        out_dir: Target directory (created if missing).
        title: Report heading.

    Returns:
        (markdown_path, json_path)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = out / f"systemone-benchmark-{stamp}.json"
    md_path = out / f"systemone-benchmark-{stamp}.md"

    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(_render_markdown(report, title), encoding="utf-8")
    return md_path, json_path
