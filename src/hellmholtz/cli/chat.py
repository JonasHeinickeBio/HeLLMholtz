"""Chat command group."""

import logging

import typer

from hellmholtz.client import chat, chat_with_fallback_detailed

logger = logging.getLogger(__name__)

# Module-level constant for the typer option (avoid B008 - function calls in defaults)
_FALLBACK_OPTION = typer.Option(
    None,
    "--fallback",
    "-f",
    help=(
        "Model to try if --model fails; repeatable. Use 'ollama' for the "
        "local default or a bare Ollama name such as llama3.2:3b"
    ),
)


def register_chat_commands(app: typer.Typer) -> None:
    """Register chat commands to the app."""

    @app.command(name="chat")
    def chat_cmd(
        model: str = typer.Option(..., help="Model name (e.g., openai:gpt-4o)"),
        message: str = typer.Argument(..., help="Message to send"),
        temperature: float | None = typer.Option(None, help="Temperature"),
        max_tokens: int | None = typer.Option(None, help="Max tokens"),
        fallback: list[str] | None = _FALLBACK_OPTION,
    ) -> None:
        """Chat with an LLM."""
        from hellmholtz.cli.common import handle_error

        # If temperature is not provided, use a default value
        if temperature is None:
            temperature = 0.7

        messages = [{"role": "user", "content": message}]
        try:
            if fallback:
                result = chat_with_fallback_detailed(
                    model,
                    messages,
                    fallbacks=fallback,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if result.used_fallback:
                    typer.echo(f"[{model} failed, answered by {result.model}]", err=True)
                typer.echo(result.text)
                return
            response = chat(model=model, messages=messages, temperature=temperature)
            typer.echo(response)
        except Exception as e:
            handle_error(e, "Chat error")
