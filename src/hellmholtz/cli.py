import logging
from pathlib import Path

import typer

from hellmholtz.benchmark import run_benchmarks
from hellmholtz.client import chat
from hellmholtz.core.config import get_settings
from hellmholtz.reporting import load_results, summarize_results

app = typer.Typer()
logger = logging.getLogger(__name__)


def configure_logging() -> None:
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
def bench(models: str | None = None, prompts_file: Path | None = None, repeat: int = 1) -> None:
    """Run benchmarks."""
    settings = get_settings()

    model_list = [m.strip() for m in models.split(",")] if models else settings.default_models

    if not model_list:
        typer.echo("No models specified and no defaults found.", err=True)
        raise typer.Exit(1)

    if prompts_file:
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = ["Hello, how are you?", "What is 2+2?"]

    try:
        typer.echo(f"Running benchmarks for models: {', '.join(model_list)}")
        results = run_benchmarks(model_list, prompts, repeat=repeat)

        typer.echo("Benchmarks completed. Results saved to results/")
        typer.echo(summarize_results(results))
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        typer.echo(f"Error running benchmarks: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def report(results_file: Path) -> None:
    """Generate a report from benchmark results."""
    try:
        results = load_results(str(results_file))
        summary = summarize_results(results)
        typer.echo(summary)
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
    configure_logging()
    app()


if __name__ == "__main__":
    main()
