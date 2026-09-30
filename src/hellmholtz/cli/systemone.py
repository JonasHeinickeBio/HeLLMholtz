"""System-One command group: typed decision routing via the Blablador /v1/systemone endpoint."""

import logging
from pathlib import Path
from typing import Any

import typer

from hellmholtz.cli.common import handle_error
from hellmholtz.providers.systemone import (
    DEFAULT_SYSTEMONE_MODEL,
    QUESTION_TYPES,
    SystemOneAnswer,
    SystemOneError,
    SystemOneQuestion,
    SystemOneResponse,
    get_api_key,
    get_endpoint,
    parse_criteria,
    parse_question,
    route,
    validate_question_type,
)
from hellmholtz.providers.systemone_benchmark import (
    DEFAULT_REPORT_DIR,
    DEFAULT_SCENARIOS,
    format_summary,
    load_scenarios,
    run_systemone_benchmark,
    write_reports,
)

logger = logging.getLogger(__name__)

# Module-level OptionInfo for the repeatable (list-typed) route options. Typer
# maps ``list[str]`` to a repeatable option; keeping the ``typer.Option`` calls
# here leaves the function defaults as plain names (B008) and mirrors
# ``cli.anonymize``.
_QUESTION_OPTION = typer.Option(
    [],
    "-q",
    "--question",
    help='Question spec "name:instructions" (repeatable, at least one required)',
    show_default=False,
)
_CHOICE_OPTION = typer.Option(
    [],
    "-c",
    "--choice",
    help='Choice criteria spec "name=option:desc;option:desc" (repeatable)',
    show_default=False,
)
_QTYPE_OPTION = typer.Option(
    [],
    "-t",
    "--type",
    help=f'Question type spec "name={"|".join(QUESTION_TYPES)}" (repeatable)',
    show_default=False,
)


def register_systemone_commands(app: typer.Typer) -> None:
    """Register System-One typed-decision commands to the app."""

    systemone_app = typer.Typer(help="System-One typed decision support (Jev/Laya)")

    @systemone_app.command(name="route")
    def systemone_route(
        state: str = typer.Argument(
            ..., help="Decision context in natural language (e.g., a clinical case description)"
        ),
        question: list[str] = _QUESTION_OPTION,
        choice: list[str] = _CHOICE_OPTION,
        qtype: list[str] = _QTYPE_OPTION,
        model: str = typer.Option(
            DEFAULT_SYSTEMONE_MODEL,
            "-m",
            "--model",
            help="System-One model to route to",
            show_default=True,
        ),
        timeout: float = typer.Option(300.0, help="Request timeout in seconds", show_default=True),
        as_json: bool = typer.Option(False, "--json", help="Print the raw API response as JSON"),
    ) -> None:
        """Route a state and typed questions through a System-One model.

        Example:
            hellm systemone route "Patient presents with fatigue since 2019."
                -q "risk:Assess the clinical risk" -c "risk=low:Mild;high:Severe"
                -q "followup:Should the patient be referred urgently?" -t "followup=noul"
        """
        _route_impl(state, question, choice, qtype, model, timeout, as_json)

    @systemone_app.command(name="benchmark")
    def systemone_benchmark(
        model: str = typer.Option(
            DEFAULT_SYSTEMONE_MODEL,
            "-m",
            "--model",
            help="System-One model to benchmark",
            show_default=True,
        ),
        replications: int = typer.Option(
            3, "--replications", help="Identical runs per scenario", show_default=True
        ),
        scenarios_file: str | None = typer.Option(
            None,
            "-s",
            "--scenarios",
            help="Path to a JSON scenarios file (defaults to the built-in clinical set)",
            show_default=False,
        ),
        output_dir: str = typer.Option(
            DEFAULT_REPORT_DIR,
            "-o",
            "--output-dir",
            help="Directory for Markdown/JSON reports",
            show_default=True,
        ),
        timeout: float = typer.Option(
            300.0, help="Per-request timeout in seconds", show_default=True
        ),
        no_progress: bool = typer.Option(
            False, "--no-progress", help="Disable per-scenario progress output"
        ),
    ) -> None:
        """Benchmark System-One decision reliability over clinical scenarios.

        Measures success rate, latency (mean/p95), decision stability and mean
        confidence by routing each scenario ``--replications`` times, then
        writes Markdown + JSON reports under ``--output-dir``.

        Example:
            hellm systemone benchmark --replications 5 --output-dir reports/systemone
        """
        _benchmark_impl(model, replications, scenarios_file, output_dir, timeout, not no_progress)

    app.add_typer(systemone_app, name="systemone", help="System-One typed decision support")


# ============================================================================
# Implementation Functions
# ============================================================================


def _apply_type_spec(specs: dict[str, dict[str, Any]], raw: str) -> None:
    """Apply a single ``name=choice|score|noul`` spec; exits on error."""
    name, sep, value = raw.partition("=")
    name = name.strip()
    value = value.strip()
    if not sep or not name:
        handle_error(
            ValueError(f"Expected a type spec as 'name=choice|score|noul', got {raw!r}."),
            "Invalid type specification",
        )
    if name not in specs:
        handle_error(
            ValueError(f"Type spec references unknown question '{name}'."),
            "Invalid type specification",
        )
    try:
        value = validate_question_type(value)
    except ValueError as e:
        handle_error(e, "Invalid type specification")
    specs[name]["type"] = value


def _apply_criteria_spec(specs: dict[str, dict[str, Any]], raw: str) -> None:
    """Apply a single ``name=option:desc;option:desc`` spec; exits on error."""
    name, sep, criteria_raw = raw.partition("=")
    name = name.strip()
    if not sep or not name:
        handle_error(
            ValueError(
                f"Expected a criteria spec as 'name=option:desc;option:desc', got {raw!r}."
            ),
            "Invalid criteria specification",
        )
    if name not in specs:
        handle_error(
            ValueError(f"Criteria spec references unknown question '{name}'."),
            "Invalid criteria specification",
        )
    try:
        criteria = parse_criteria(criteria_raw)
    except ValueError as e:
        handle_error(e, "Invalid criteria specification")
    specs[name]["criteria"] = criteria


def _make_question(name: str, spec: dict[str, Any]) -> SystemOneQuestion:
    """Assemble one SystemOneQuestion from a spec dict; exits on conflict."""
    question_type = spec["type"] or ("choice" if spec["criteria"] else "score")
    if spec["criteria"] and question_type != "choice":
        handle_error(
            ValueError(
                f"Question '{name}' has criteria but type '{question_type}'; "
                "criteria are only valid for 'choice' questions."
            ),
            "Invalid question specification",
        )
    return SystemOneQuestion(
        type=question_type,
        instructions=spec["instructions"],
        criteria=spec["criteria"],
    )


def _build_questions(
    question_args: list[str],
    choice_args: list[str],
    type_args: list[str],
) -> dict[str, SystemOneQuestion]:
    """Build SystemOneQuestion objects from the CLI spec options.

    Args:
        question_args: Repeatable "name:instructions" question specs.
        choice_args: Repeatable "name=option:desc;option:desc" criteria specs.
        type_args: Repeatable "name=choice|score|noul" type specs.

    Returns:
        Mapping of question name to its SystemOneQuestion.

    Raises:
        typer.Exit: If any spec is malformed or inconsistent.
    """
    specs: dict[str, dict[str, Any]] = {}

    for raw in question_args:
        try:
            name, instructions = parse_question(raw)
        except ValueError as e:
            handle_error(e, "Invalid question specification")
        if name in specs:
            handle_error(
                ValueError(f"Duplicate question name '{name}'."),
                "Invalid question specification",
            )
        specs[name] = {"instructions": instructions, "type": None, "criteria": {}}

    for raw in type_args:
        _apply_type_spec(specs, raw)

    for raw in choice_args:
        _apply_criteria_spec(specs, raw)

    if not specs:
        handle_error(
            ValueError("At least one question is required (use -q/--question)."),
            "No questions specified",
        )

    return {name: _make_question(name, spec) for name, spec in specs.items()}


def _route_impl(
    state: str,
    question_args: list[str],
    choice_args: list[str],
    type_args: list[str],
    model: str,
    timeout: float,
    as_json: bool,
) -> None:
    """Implementation for the systemone route command."""
    from rich.console import Console

    if not get_api_key():
        handle_error(
            ValueError(
                "No API key configured for System-One. "
                "Set SYSTEMONE_API_KEY (or BLABLADOR_API_KEY) in the environment."
            ),
            "System-One routing",
        )

    questions = _build_questions(question_args, choice_args, type_args)

    console = Console()
    console.print(
        f"Routing [bold]{len(questions)}[/bold] question(s) via [bold]{model}[/bold] "
        f"at [dim]{get_endpoint()}[/dim] ..."
    )
    try:
        response = route(state, questions, model=model, timeout=timeout)
    except SystemOneError as e:
        handle_error(e, "System-One routing")

    if as_json:
        typer.echo(response.model_dump_json(indent=2))
        return

    _print_response(response, console)


def _benchmark_impl(
    model: str,
    replications: int,
    scenarios_file: str | None,
    output_dir: str,
    timeout: float,
    progress: bool,
) -> None:
    """Implementation for the systemone benchmark command."""
    from rich.console import Console

    if not get_api_key():
        handle_error(
            ValueError(
                "No API key configured for System-One. "
                "Set SYSTEMONE_API_KEY (or BLABLADOR_API_KEY) in the environment."
            ),
            "System-One benchmark",
        )

    if scenarios_file is not None:
        try:
            scenarios = load_scenarios(Path(scenarios_file))
        except ValueError as e:
            handle_error(e, "System-One benchmark")
    else:
        scenarios = DEFAULT_SCENARIOS

    console = Console()
    console.print(
        f"Benchmarking [bold]{model}[/bold] at [dim]{get_endpoint()}[/dim] "
        f"({len(scenarios)} scenario(s), {replications} run(s) each) ..."
    )
    report = run_systemone_benchmark(
        scenarios,
        model=model,
        replications=replications,
        timeout=timeout,
        progress=progress,
    )

    md_path, json_path = write_reports(report, out_dir=output_dir)
    console.print(format_summary(report))
    console.print(f"\n[dim]Reports written:[/dim] {md_path} {json_path}")


def _print_response(response: SystemOneResponse, console: Any) -> None:
    """Render a SystemOneResponse as rich tables."""
    from rich.table import Table

    for name, answer in response.answers.items():
        console.print(f"\n[bold]{name}[/bold] [dim]({answer.type})[/dim]")
        if answer.probabilities:
            table = Table(show_header=True)
            table.add_column("Option")
            table.add_column("Probability", justify="right")
            for option, probability in sorted(
                answer.probabilities.items(), key=lambda kv: kv[1], reverse=True
            ):
                is_chosen = option == answer.choice
                label = f"[bold green]◆ {option}[/bold green]" if is_chosen else option
                table.add_row(label, f"{probability:.4f}")
            console.print(table)
        elif answer.choice:
            console.print(f"  Answer: {answer.choice}")

        details: list[str] = []
        if answer.confidence is not None:
            details.append(f"confidence={answer.confidence:.4f}")
        if answer.action is not None and answer.action.act_probability is not None:
            details.append(f"act_probability={answer.action.act_probability:.4f}")
        if details:
            console.print(f"  [dim]{' | '.join(details)}[/dim]")

    console.print(
        f"\n[dim]Usage: {response.usage.input_tokens} input tokens, "
        f"{response.usage.output_tokens} output tokens | "
        f"routed to {response.routing.model} ({response.routing.repo})[/dim]"
    )


__all__ = [
    "register_systemone_commands",
    "SystemOneAnswer",
    "SystemOneError",
    "SystemOneQuestion",
    "SystemOneResponse",
]
