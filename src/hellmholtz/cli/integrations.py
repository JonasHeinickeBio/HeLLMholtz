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
    def proxy(
        model: str,
        port: int = typer.Option(4000, help="Port to listen on"),
        host: str = typer.Option("127.0.0.1", help="Host to bind the proxy to"),
        name: str | None = typer.Option(None, "--name", help="Alias to expose the model under"),
        master_key: str | None = typer.Option(
            None, "--master-key", help="Proxy master key (generated in --claude-code mode)"
        ),
        config: str | None = typer.Option(
            None, "--config", help="Path to an existing LiteLLM config file"
        ),
        claude_code: bool = typer.Option(
            False, "--claude-code", help="Print snippet that points Claude Code at the proxy"
        ),
        debug: bool = typer.Option(False, help="Run the proxy in debug mode"),
    ) -> None:
        """Start LiteLLM Proxy (OpenAI- and Anthropic-compatible endpoints)."""
        _proxy_impl(model, port, host, name, master_key, config, claude_code, debug)

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


def _proxy_impl(
    model: str,
    port: int,
    host: str,
    name: str | None,
    master_key: str | None,
    config: str | None,
    claude_code: bool,
    debug: bool,
) -> None:
    """Implementation for proxy command."""
    from hellmholtz.integrations.litellm import start_proxy

    try:
        start_proxy(
            model,
            port=port,
            config_path=config,
            debug=debug,
            host=host,
            model_name=name,
            master_key=master_key,
            claude_code=claude_code,
        )
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
