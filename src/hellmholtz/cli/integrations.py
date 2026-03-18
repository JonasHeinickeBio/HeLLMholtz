"""Integration command group: lm_eval, proxy, bench_throughput."""

import logging

import typer

from hellmholtz.cli.common import handle_error

logger = logging.getLogger(__name__)


def register_integration_commands(app: typer.Typer) -> None:
    """Register integration commands to the app."""

    @app.command()
    def lm_eval(
        model: str, tasks: str, num_fewshot: int | None = None, limit: float | None = None
    ) -> None:
        """Run LM Evaluation Harness."""
        _lm_eval_impl(model, tasks, num_fewshot, limit)

    @app.command()
    def proxy(model: str, port: int = 4000, debug: bool = False) -> None:
        """Start LiteLLM Proxy."""
        _proxy_impl(model, port, debug)

    @app.command()
    def bench_throughput(
        model: str,
        prompt: str = "Write a long story about a space adventure.",
        max_tokens: int = 100,
    ) -> None:
        """Run throughput benchmark."""
        _bench_throughput_impl(model, prompt, max_tokens)


# ============================================================================
# Implementation Functions
# ============================================================================


def _lm_eval_impl(model: str, tasks: str, num_fewshot: int | None, limit: float | None) -> None:
    """Implementation for lm_eval command."""
    from hellmholtz.integrations.lm_eval import run_lm_eval

    try:
        task_list = [t.strip() for t in tasks.split(",")]
        run_lm_eval(model, task_list, num_fewshot=num_fewshot, limit=limit)
    except Exception as e:
        handle_error(e, "LM Eval error")


def _proxy_impl(model: str, port: int, debug: bool) -> None:
    """Implementation for proxy command."""
    from hellmholtz.integrations.litellm import start_proxy

    try:
        start_proxy(model, port=port, debug=debug)
    except Exception as e:
        handle_error(e, "Proxy error")


def _bench_throughput_impl(model: str, prompt: str, max_tokens: int) -> None:
    """Implementation for bench_throughput command."""
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
        handle_error(e, "Throughput benchmark error")
