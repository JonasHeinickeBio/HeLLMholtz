"""Chat command group."""

import logging

import typer

from hellmholtz.client import chat

logger = logging.getLogger(__name__)


def register_chat_commands(app: typer.Typer) -> None:
    """Register chat commands to the app."""

    @app.command(name="chat")
    def chat_cmd(
        model: str = typer.Option(..., help="Model name (e.g., openai:gpt-4o)"),
        message: str = typer.Argument(..., help="Message to send"),
        temperature: float | None = typer.Option(None, help="Temperature"),
        max_tokens: int | None = typer.Option(None, help="Max tokens"),
    ) -> None:
        """Chat with an LLM."""
        from hellmholtz.cli.common import handle_error

        # If temperature is not provided, use a default value
        if temperature is None:
            temperature = 0.7

        try:
            response = chat(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
            )
            typer.echo(response)
        except Exception as e:
            handle_error(e, "Chat error")
