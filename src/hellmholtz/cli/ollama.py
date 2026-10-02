"""Ollama command group: local server status, models, and chat."""

import logging

import typer

from hellmholtz.client import ollama_chat
from hellmholtz.providers import ollama

logger = logging.getLogger(__name__)


def register_ollama_commands(app: typer.Typer) -> None:
    """Register the ``hellm ollama`` command group to the app."""
    ollama_app = typer.Typer(help="Use local Ollama models (also the default chat fallback).")
    app.add_typer(ollama_app, name="ollama")

    @ollama_app.command(name="status")
    def status_cmd() -> None:
        """Check whether a local Ollama server is reachable."""
        try:
            models = ollama.list_models()
        except ollama.OllamaUnavailableError as e:
            typer.echo(f"Ollama: unavailable - {e}", err=True)
            raise typer.Exit(code=1) from e
        typer.echo(f"Ollama: running at {ollama.get_base_url()} ({len(models)} model(s))")

    @ollama_app.command(name="models")
    def models_cmd() -> None:
        """List models installed on the local Ollama server."""
        from hellmholtz.cli.common import handle_error

        try:
            for name in ollama.list_models():
                typer.echo(ollama.qualify(name))
        except Exception as e:
            handle_error(e, "Ollama error")

    @ollama_app.command(name="chat")
    def chat_cmd(
        message: str = typer.Argument(..., help="Message to send"),
        model: str | None = typer.Option(
            None, help="Ollama model (default: HELLM_OLLAMA_MODEL, then first installed)"
        ),
        temperature: float = typer.Option(0.7, help="Temperature"),
    ) -> None:
        """Chat with a local Ollama model."""
        from hellmholtz.cli.common import handle_error

        try:
            typer.echo(ollama_chat(message, model=model, temperature=temperature))
        except Exception as e:
            handle_error(e, "Ollama error")
