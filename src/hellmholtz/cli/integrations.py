"""Integration command group: lm_eval, proxy, bench_throughput, anythingllm."""

import logging
from pathlib import Path

import typer

from hellmholtz.cli.common import handle_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AnythingLLM Typer sub-app
# ---------------------------------------------------------------------------

anythingllm_app = typer.Typer(
    name="anythingllm",
    help=(
        "Chat with your documents via AnythingLLM. "
        "Requires a running AnythingLLM instance (see docker/docker-compose.yml). "
        "Set ANYTHINGLLM_BASE_URL and ANYTHINGLLM_API_KEY in your environment."
    ),
    no_args_is_help=True,
)

_BASE_URL_OPT = typer.Option(
    None,
    "--base-url",
    envvar="ANYTHINGLLM_BASE_URL",
    help="AnythingLLM base URL (default: http://localhost:3001).",
)
_API_KEY_OPT = typer.Option(
    None,
    "--api-key",
    envvar="ANYTHINGLLM_API_KEY",
    help="AnythingLLM API key (prefer env var to avoid leaking in shell history).",
    hide_input=True,
)
_WORKSPACE_OPT = typer.Option(..., "--workspace", "-w", help="Workspace slug.")


@anythingllm_app.command(name="ping")
def anythingllm_ping(
    base_url: str | None = _BASE_URL_OPT,
    api_key: str | None = _API_KEY_OPT,
) -> None:
    """Check whether the AnythingLLM instance is reachable."""
    from hellmholtz.integrations.anythingllm import AnythingLLMClient

    client = AnythingLLMClient(base_url=base_url, api_key=api_key)
    if client.ping():
        typer.echo(f"✓ AnythingLLM is reachable at {client._base_url}")
    else:
        typer.echo(f"✗ Could not reach AnythingLLM at {client._base_url}", err=True)
        raise typer.Exit(1)


@anythingllm_app.command(name="workspaces")
def anythingllm_workspaces(
    base_url: str | None = _BASE_URL_OPT,
    api_key: str | None = _API_KEY_OPT,
) -> None:
    """List all workspaces."""
    from hellmholtz.integrations.anythingllm import AnythingLLMClient

    try:
        client = AnythingLLMClient(base_url=base_url, api_key=api_key)
        workspaces = client.list_workspaces()
        if not workspaces:
            typer.echo("No workspaces found.")
            return
        for ws in workspaces:
            typer.echo(f"  slug={ws.get('slug')!r}  name={ws.get('name')!r}")
    except Exception as exc:
        handle_error(exc, "AnythingLLM workspaces error")


@anythingllm_app.command(name="chat")
def anythingllm_chat(
    message: str = typer.Argument(..., help="Message / question to send."),
    workspace: str = _WORKSPACE_OPT,
    mode: str = typer.Option(
        "chat",
        "--mode",
        help="'chat' (history kept) or 'query' (one-shot RAG).",
    ),
    session_id: str | None = typer.Option(None, "--session-id", help="Conversation session ID."),
    base_url: str | None = _BASE_URL_OPT,
    api_key: str | None = _API_KEY_OPT,
) -> None:
    """Send a message to an AnythingLLM workspace and print the reply.

    The workspace must already exist and have documents ingested.

    Examples::

        hellm anythingllm chat --workspace my-docs "What is the main finding?"
        hellm anythingllm chat --workspace my-docs --mode query "Summarise chapter 3"
    """
    from hellmholtz.integrations.anythingllm import AnythingLLMClient

    try:
        client = AnythingLLMClient(base_url=base_url, api_key=api_key)
        reply = client.chat(workspace, message, mode=mode, session_id=session_id)
        typer.echo(reply)
    except Exception as exc:
        handle_error(exc, "AnythingLLM chat error")


_FILE_ARG = typer.Argument(..., help="Local file to upload (PDF, DOCX, TXT, MD …).")


@anythingllm_app.command(name="upload")
def anythingllm_upload(
    file: Path = _FILE_ARG,
    workspace: str = _WORKSPACE_OPT,
    base_url: str | None = _BASE_URL_OPT,
    api_key: str | None = _API_KEY_OPT,
) -> None:
    """Upload a local document to an AnythingLLM workspace.

    The document is automatically embedded into the workspace's vector store.

    Examples::

        hellm anythingllm upload report.pdf --workspace my-docs
        hellm anythingllm upload ./papers/ --workspace research
    """
    from hellmholtz.integrations.anythingllm import AnythingLLMClient

    try:
        client = AnythingLLMClient(base_url=base_url, api_key=api_key)
        result = client.upload_document(workspace, file)
        typer.echo(f"Uploaded: {file.name}")
        if "document" in result:
            typer.echo(f"  location: {result['document'].get('location', 'unknown')}")
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        handle_error(exc, "AnythingLLM upload error")


def register_integration_commands(app: typer.Typer) -> None:
    """Register integration commands to the app."""

    # Existing top-level commands
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

    # AnythingLLM sub-app
    app.add_typer(anythingllm_app, name="anythingllm")


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
