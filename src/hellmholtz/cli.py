import logging
from pathlib import Path

import typer

from hellmholtz.client import chat
from hellmholtz.core.config import get_settings
from hellmholtz.core.prompts import Message, Prompt
from hellmholtz.reporting import (
    generate_html_report,
    generate_html_report_detailed,
    generate_html_report_full,
    generate_html_report_simple,
    generate_markdown_report,
    load_results,
)

app = typer.Typer()
logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure logging for the CLI application.

    Sets up INFO level logging with a formatted output including
    timestamp, logger name, level, and message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command(name="chat")
def chat_cmd(
    model: str = typer.Option(..., help="Model name (e.g., openai:gpt-4o)"),
    message: str = typer.Argument(..., help="Message to send"),
    temperature: float | None = typer.Option(None, help="Temperature"),
    max_tokens: int | None = typer.Option(None, help="Max tokens"),
) -> None:
    """Chat with an LLM."""
    # The original logic for default model is now handled by typer.Option(..., help=...)
    # If temperature is not provided, use a default value
    if temperature is None:
        temperature = 0.7

    try:
        response = chat(
            model=model, messages=[{"role": "user", "content": message}], temperature=temperature
        )
        typer.echo(response)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e


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
    prompts_file: Path | None = None,
    prompts_category: str | None = typer.Option(
        None, help="Category of prompts to use (reasoning, coding, creative, knowledge)"
    ),
    all_prompts: bool = typer.Option(False, help="Use all available prompts"),
    temperatures: str | None = typer.Option(
        "0.1,0.7,1.0", help="Comma-separated temperature values to test"
    ),
    max_tokens: int | None = typer.Option(None, help="Maximum tokens for responses"),
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
    settings = get_settings()

    # Determine models to test
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        # Use all available models
        model_list = settings.default_models.copy()

        # Try to add Blablador models if available
        try:
            from hellmholtz.providers.blablador import list_models

            blablador_models = list_models()
            if blablador_models:
                # Add a few representative Blablador models
                blablador_to_add = []
                for model in blablador_models[:3]:  # Limit to first 3 to avoid too many
                    identifier = model.name if model.name else model.id
                    blablador_to_add.append(f"blablador:{identifier}")
                model_list.extend(blablador_to_add)
                typer.echo(f"Added {len(blablador_to_add)} Blablador models to test")
        except Exception as e:
            typer.echo(f"Could not load Blablador models: {e}")

    if not model_list:
        typer.echo("No models available for testing.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Testing {len(model_list)} models: {', '.join(model_list)}")

    # Determine prompts to use
    if prompts_file:
        typer.echo(f"Loading prompts from file: {prompts_file}")
        with open(prompts_file) as f:
            file_prompts = [line.strip() for line in f if line.strip()]
        prompts = [
            Prompt(
                id=f"custom_{i}",
                category="custom",
                messages=[Message(role="user", content=line)],
            )
            for i, line in enumerate(file_prompts)
        ]
        typer.echo(f"Loaded {len(prompts)} custom prompts")
    elif prompts_category:
        from hellmholtz.benchmark.prompts import get_prompts_by_category

        prompts = get_prompts_by_category(prompts_category)
        if not prompts:
            from hellmholtz.benchmark.prompts import PROMPTS

            available_categories = set(prompt.category for prompt in PROMPTS)
            available_str = ", ".join(sorted(available_categories))
            typer.echo(
                f"Invalid category '{prompts_category}'. Available: {available_str}",
                err=True,
            )
            raise typer.Exit(1)
        typer.echo(f"Using {len(prompts)} prompts from category '{prompts_category}'")
    elif all_prompts:
        from hellmholtz.benchmark.prompts import get_all_prompts

        prompts = get_all_prompts()
        typer.echo(f"Using all {len(prompts)} available prompts")
    else:
        # Default: use reasoning prompts as a representative sample
        from hellmholtz.benchmark.prompts import get_prompts_by_category

        prompts = get_prompts_by_category("reasoning")
        typer.echo(f"Using {len(prompts)} reasoning prompts (default)")

    # Parse temperatures
    if temperatures:
        try:
            temp_list = [float(t.strip()) for t in temperatures.split(",")]
        except ValueError as e:
            typer.echo("Invalid temperature values. Use comma-separated floats.", err=True)
            raise typer.Exit(1) from e
    else:
        temp_list = [0.1, 0.7, 1.0]

    total_tests = len(model_list) * len(prompts) * len(temp_list) * replications
    typer.echo(
        f"Total benchmark tests: {total_tests} "
        f"({len(temp_list)} temperatures × {replications} replications)"
    )
    typer.echo(f"Estimated time: ~{total_tests * 3 // 60} minutes (assuming 3s per test)")

    try:
        from hellmholtz.benchmark import run_benchmarks

        results = run_benchmarks(
            model_list,
            prompts,
            temperatures=temp_list,
            max_tokens=max_tokens,
            replications=replications,
            system_prompt=system_prompt,
        )

        if evaluate_with:
            typer.echo(f"\nEvaluating results with {evaluate_with}...")
            from hellmholtz.benchmark.evaluator import evaluate_responses

            results = evaluate_responses(results, evaluate_with, prompts)

            # Note: run_benchmarks already saved results,
            # evaluation results are not automatically saved
            # To save evaluation results, call save_results manually here

        typer.echo("\nBenchmarks completed! Results saved to results/ directory")
        typer.echo("Use 'hellm report <results_file>' to generate detailed reports")

    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        typer.echo(f"Error running benchmarks: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def report(
    results_file: Path,
    format: str = typer.Option(
        "markdown", help="Output format: markdown, html, html-simple, html-detailed, html-full"
    ),
    output: Path | None = None,
) -> None:
    """Generate a report from benchmark results."""
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

        if output:
            output_path = output
        else:
            # Auto-generate filename based on format and timestamp
            timestamp = (
                results[0].timestamp.replace(":", "-").replace(".", "-") if results else "unknown"
            )
            if format.lower().startswith("html"):
                output_path = Path(
                    f"reports/html/benchmark_report_{format.lower()}_{timestamp}.html"
                )
            else:
                output_path = Path(f"reports/benchmark_report_{timestamp}.md")

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(content)
        typer.echo(f"Report saved to {output_path}")

    except Exception as e:
        logger.error(f"Report error: {e}")
        typer.echo(f"Error reading results: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def lm_eval(
    model: str, tasks: str, num_fewshot: int | None = None, limit: float | None = None
) -> None:
    """Run LM Evaluation Harness."""
    from hellmholtz.integrations.lm_eval import run_lm_eval

    try:
        task_list = [t.strip() for t in tasks.split(",")]
        run_lm_eval(model, task_list, num_fewshot=num_fewshot, limit=limit)
    except Exception as e:
        logger.error(f"LM Eval error: {e}")
        typer.echo(f"Error running evaluation: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def proxy(model: str, port: int = 4000, debug: bool = False) -> None:
    """Start LiteLLM Proxy."""
    from hellmholtz.integrations.litellm import start_proxy

    try:
        start_proxy(model, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        typer.echo(f"Error starting proxy: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def bench_throughput(
    model: str, prompt: str = "Write a long story about a space adventure.", max_tokens: int = 100
) -> None:
    """Run throughput benchmark."""
    from hellmholtz.benchmark import run_throughput_benchmark

    try:
        result = run_throughput_benchmark(model, prompt, max_tokens)

        if result["success"]:
            typer.echo(f"Model: {result['model']}")
            typer.echo(f"Tokens/sec: {result['tokens_per_sec']:.2f}")
            typer.echo(f"Latency: {result['latency']:.2f}s")
            typer.echo(f"Output Tokens: {result['output_tokens']}")
        else:
            logger.error(f"Throughput benchmark failed: {result.get('error')}")
            typer.echo(f"Error: {result.get('error')}", err=True)
    except Exception as e:
        logger.error(f"Throughput benchmark error: {e}")
        typer.echo(f"Error running throughput benchmark: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def models() -> None:
    """List available models from Blablador."""
    from hellmholtz.providers.blablador import list_models

    try:
        models = list_models()
        typer.echo(f"{'ID':<5} | {'Name':<40} | {'Alias':<10} | {'Source':<10} | {'Description'}")
        typer.echo("-" * 100)
        for model in models:
            alias = model.alias if model.alias else ""
            # If ID is same as Name (fallback), just show Name
            if model.id == model.name:
                typer.echo(
                    f"{'':<5} | {model.name:<40} | {alias:<10} | "
                    f"{model.source:<10} | {model.description}"
                )
            else:
                typer.echo(
                    f"{model.id:<5} | {model.name:<40} | {alias:<10} | "
                    f"{model.source:<10} | {model.description}"
                )
    except Exception as e:
        logger.error(f"Model list error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e


def main() -> None:
    """Main entry point for the HeLLMholtz CLI application.

    Configures logging and launches the Typer app.
    """
    configure_logging()
    app()


if __name__ == "__main__":
    main()
