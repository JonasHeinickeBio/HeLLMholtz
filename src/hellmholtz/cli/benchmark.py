"""Benchmark command group: bench, report, chart, analyze."""

from datetime import datetime
import logging
from pathlib import Path

import typer

from hellmholtz.cli.common import (
    ALL_PROMPTS_OPTION,
    HTML_REPORT_OPTION,
    MAX_TOKENS_OPTION,
    PROMPTS_CATEGORY_OPTION,
    PROMPTS_FILE_OPTION,
    RESULTS_FILE_ARGUMENT,
    TEMPERATURES_OPTION,
    generate_output_path,
    handle_error,
    parse_models,
    parse_temperatures,
    save_report_to_file,
)
from hellmholtz.reporting import load_results

logger = logging.getLogger(__name__)

# Module-level constants for typer options (avoid B008 - function calls in defaults)
OUTPUT_CHART_OPTION = typer.Option(
    None, help="Output path for the chart image (default: auto-generated)"
)


def register_benchmark_commands(app: typer.Typer) -> None:
    """Register benchmark commands to the app."""

    @app.command()
    def bench(  # noqa: C901
        models: str | None = typer.Option(
            None,
            help=(
                "Comma-separated model names "
                "(e.g., 'openai:gpt-4o,anthropic:claude-3-haiku'). "
                "If not specified, uses all available models"
            ),
        ),
        prompts_file: Path | None = PROMPTS_FILE_OPTION,
        prompts_category: str | None = PROMPTS_CATEGORY_OPTION,
        all_prompts: bool = ALL_PROMPTS_OPTION,
        temperatures: str | None = TEMPERATURES_OPTION,
        max_tokens: int | None = MAX_TOKENS_OPTION,
        replications: int = typer.Option(
            3, help="Number of replications per configuration for statistical significance"
        ),
        evaluate_with: str | None = typer.Option(
            None, help="Model to use for evaluation (LLM-as-a-Judge)"
        ),
        system_prompt: str | None = typer.Option(
            None, help="System prompt to prepend to all messages"
        ),
    ) -> None:
        """Run benchmarks across models and prompts with temperature and replications."""
        _bench_impl(
            models,
            prompts_file,
            prompts_category,
            all_prompts,
            temperatures,
            max_tokens,
            replications,
            evaluate_with,
            system_prompt,
        )

    @app.command()
    def report(
        results_file: Path,
        format: str = typer.Option(
            "markdown",
            help="Output format: markdown, html, html-simple, html-detailed, html-full",
        ),
        output: Path | None = None,
    ) -> None:
        """Generate a report from benchmark results."""
        _report_impl(results_file, format, output)

    @app.command()
    def chart(
        results_file: Path = RESULTS_FILE_ARGUMENT,
        output: Path | None = OUTPUT_CHART_OPTION,
    ) -> None:
        """Generate a performance chart from benchmark results."""
        _chart_impl(results_file, output)

    @app.command()
    def analyze(
        results_file: Path = RESULTS_FILE_ARGUMENT,
        html_report: Path | None = HTML_REPORT_OPTION,
    ) -> None:
        """Analyze benchmark evaluation results with LLM-as-a-Judge scoring.

        Provides comprehensive analysis including model rankings, statistical
        summaries, and interactive visualizations. Supports evaluation results
        from benchmarks run with the --evaluate-with flag.
        """
        _analyze_impl(results_file, html_report)


# ============================================================================
# Implementation Functions (DRY - encapsulated logic)
# ============================================================================


def _bench_impl(
    models: str | None,
    prompts_file: Path | None,
    prompts_category: str | None,
    all_prompts: bool,
    temperatures: str | None,
    max_tokens: int | None,
    replications: int,
    evaluate_with: str | None,
    system_prompt: str | None,
) -> None:
    """Implementation for bench command."""
    from hellmholtz.benchmark import run_benchmarks
    from hellmholtz.benchmark.prompts import get_all_prompts, get_prompts_by_category
    from hellmholtz.benchmark.runner import save_results
    from hellmholtz.cli.common import load_prompts_from_file
    from hellmholtz.client import check_model_availability

    try:
        # Parse input
        model_list = parse_models(models)
        temp_list = parse_temperatures(temperatures)

        typer.echo(f"Testing {len(model_list)} models: {', '.join(model_list)}")

        # Load prompts
        if prompts_file:
            prompts = load_prompts_from_file(prompts_file)
        elif prompts_category:
            prompts = get_prompts_by_category(prompts_category)
            if not prompts:
                handle_error(
                    ValueError(f"Invalid category '{prompts_category}'"),
                    "Category validation failed",
                )
            typer.echo(f"Using {len(prompts)} prompts from category '{prompts_category}'")
        elif all_prompts:
            prompts = get_all_prompts()
            typer.echo(f"Using all {len(prompts)} available prompts")
        else:
            prompts = get_prompts_by_category("reasoning")
            typer.echo(f"Using {len(prompts)} reasoning prompts (default)")

        # Calculate and display metrics
        total_tests = len(model_list) * len(prompts) * len(temp_list) * replications
        typer.echo(
            f"Total benchmark tests: {total_tests} "
            f"({len(temp_list)} temperatures × {replications} replications)"
        )
        typer.echo(f"Estimated time: ~{total_tests * 3 // 60} minutes (assuming 3s per test)")

        # Run benchmarks
        results = run_benchmarks(
            model_list,
            prompts,
            temperatures=temp_list,
            max_tokens=max_tokens,
            replications=replications,
            system_prompt=system_prompt,
        )

        # Evaluate if requested
        if evaluate_with:
            typer.echo(f"\nEvaluating results with {evaluate_with}...")
            if not check_model_availability(evaluate_with):
                typer.echo(
                    f"⚠️  Judge model '{evaluate_with}' is not available. "
                    f"Skipping evaluation and saving results without ratings.",
                    err=True,
                )
                timestamp = datetime.now().isoformat()
                results_path = Path("results")
                save_results(results, results_path, timestamp)
            else:
                from hellmholtz.benchmark.evaluator import evaluate_responses

                results = evaluate_responses(results, evaluate_with, prompts)
                typer.echo("✅ Evaluation completed successfully")
                timestamp = datetime.now().isoformat()
                results_path = Path("results")
                save_results(results, results_path, f"{timestamp}_evaluated")
        else:
            timestamp = datetime.now().isoformat()
            results_path = Path("results")
            save_results(results, results_path, timestamp)

        typer.echo("\nBenchmarks completed! Results saved to results/ directory")
        typer.echo("Use 'hellm report <results_file>' to generate detailed reports")

    except Exception as e:
        handle_error(e, "Benchmark error")


def _report_impl(results_file: Path, format: str, output: Path | None) -> None:
    """Implementation for report command."""
    from hellmholtz.reporting import (
        generate_html_report,
        generate_html_report_detailed,
        generate_html_report_full,
        generate_html_report_simple,
        generate_markdown_report,
    )

    try:
        results = load_results(str(results_file))

        if format.lower() == "html":
            content = generate_html_report(results)
        elif format.lower() == "html-simple":
            content = generate_html_report_simple(results)
        elif format.lower() == "html-detailed":
            content = generate_html_report_detailed(results)
        elif format.lower() == "html-full":
            content = generate_html_report_full(results)
        else:
            content = generate_markdown_report(results)

        if not output:
            output = generate_output_path(results, format)

        save_report_to_file(content, output)

    except Exception as e:
        handle_error(e, "Report generation error")


def _chart_impl(results_file: Path, output: Path | None) -> None:
    """Implementation for chart command."""
    try:
        from hellmholtz.reporting.chart import generate_performance_chart

        if not output:
            results = load_results(str(results_file))
            output = generate_output_path(results, "chart")

        output.parent.mkdir(parents=True, exist_ok=True)
        generate_performance_chart(str(results_file), str(output))
        typer.echo(f"Chart saved to {output}")

    except ImportError:
        handle_error(
            ImportError("matplotlib not installed"),
            "Chart generation requires matplotlib. Install with: pip install matplotlib",
        )
    except Exception as e:
        handle_error(e, "Chart generation error")


def _analyze_impl(results_file: Path, html_report: Path | None) -> None:
    """Implementation for analyze command."""
    from hellmholtz.evaluation_analysis import analyze_evaluations_cli

    try:
        analyze_evaluations_cli(str(results_file), str(html_report) if html_report else None)

        if html_report:
            typer.echo(f"\n📊 Analysis complete! HTML report saved to: {html_report}")
        else:
            typer.echo("\n📊 Analysis complete! Use --html-report to generate visualizations.")

    except Exception as e:
        handle_error(e, "Analysis error")
