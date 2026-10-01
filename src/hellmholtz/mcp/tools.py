"""Plain-Python logic behind the HeLLMholtz MCP tools.

This module deliberately avoids importing ``mcp`` so it stays testable and
importable without the optional dependency. :mod:`hellmholtz.mcp.server`
binds these methods to FastMCP tool declarations.
"""

import json
import logging
from typing import Any

from hellmholtz.core.config import get_settings

logger = logging.getLogger(__name__)

#: Environment variable naming the default model used when a tool call does
#: not pass one explicitly.
DEFAULT_MODEL_ENV_VAR = "HELLM_MCP_MODEL"

#: Fallback default when neither a CLI option, the env var, nor a configured
#: default model is available.
FALLBACK_MODEL = "blablador:alias-large"

SERVER_NAME = "hellmholtz"


def resolve_default_model(model: str | None = None) -> str:
    """Resolve the model an MCP tool call should use.

    Precedence: explicit argument > ``HELLM_MCP_MODEL`` env var > first
    configured default model > built-in fallback.

    Args:
        model: Explicit model, if the caller provided one.

    Returns:
        The effective model identifier (``provider:model`` form).
    """
    if model:
        return model
    import os

    env_model = os.getenv(DEFAULT_MODEL_ENV_VAR)
    if env_model:
        return env_model
    settings = get_settings()
    if settings.default_models:
        return settings.default_models[0]
    return FALLBACK_MODEL


def claude_desktop_config(
    model: str | None = None,
    command: str = "hellm",
    args: list[str] | None = None,
) -> str:
    """Build the Claude Desktop ``claude_desktop_config.json`` snippet.

    Args:
        model: Default model to launch the server with (omitted if None).
        command: Executable MCP clients should run (``hellm`` or, from a
            source checkout, e.g. ``python -m hellmholtz.mcp`` is expressed
            via ``command``/``args``).
        args: Extra arguments appended after ``["mcp"]``.

    Returns:
        Pretty-printed JSON for the ``mcpServers`` section.
    """
    server_args: list[str] = ["mcp"]
    if model:
        server_args.extend(["--model", model])
    if args:
        server_args.extend(args)
    config = {
        "mcpServers": {
            SERVER_NAME: {
                "command": command,
                "args": server_args,
            }
        }
    }
    return json.dumps(config, indent=2)


class HellmTools:
    """Tool implementations exposed over MCP.

    Thin, dependency-light wrappers around :func:`hellmholtz.client.chat`
    and the Blablador model catalog. All methods return plain strings so
    FastMCP can serialize them directly.
    """

    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def list_models(self) -> str:
        """List the model identifiers that ask_external can use.

        Returns:
            One ``provider:model`` identifier per line, or an error string.
        """
        try:
            from hellmholtz.providers.blablador import list_models as blablador_list_models

            models = blablador_list_models()
        except Exception as e:  # noqa: BLE001 - surface errors as tool text
            logger.error(f"MCP list_models failed: {e}")
            return f"Error listing models: {e}"
        if not models:
            return "No Blablador models available."
        return "\n".join(f"blablador:{model.name or model.id}" for model in models)

    def ask_external(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Ask a HeLLMholtz (Blablador) model a single-question task.

        Use this to offload long or heavy generation tasks (summarizing,
        drafting, translating, long-context analysis) to external models.

        Args:
            prompt: The user prompt to send.
            system: Optional system instructions for the model.
            model: Model identifier; defaults to the server's default model.
                Call list_models first to see valid identifiers.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum number of tokens to generate.

        Returns:
            The model answer, or a readable error string.
        """
        if not prompt or not prompt.strip():
            return "Error: prompt must not be empty."
        messages: list[dict[str, Any]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat(messages, model, temperature=temperature, max_tokens=max_tokens)

    def chat_external(
        self,
        messages: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Continue a conversation on a HeLLMholtz model with full context.

        Args:
            messages: JSON array of chat messages, e.g.
                '[{"role":"system","content":"..."},{"role":"user","content":"..."}]'.
                The last message should be from the user.
            model: Model identifier; defaults to the server's default model.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum number of tokens to generate.

        Returns:
            The model answer, or a readable error string.
        """
        try:
            parsed = json.loads(messages)
        except json.JSONDecodeError as e:
            return f"Error: messages must be a JSON array of chat messages ({e})."
        if not isinstance(parsed, list) or not parsed:
            return "Error: messages must be a non-empty JSON array."
        for item in parsed:
            if (
                not isinstance(item, dict)
                or item.get("role") not in ("system", "user", "assistant")
                or not isinstance(item.get("content"), str)
            ):
                return (
                    "Error: each message needs a 'role' "
                    "(system/user/assistant) and string 'content'."
                )
        return self._chat(parsed, model, temperature=temperature, max_tokens=max_tokens)

    def get_info(self) -> str:
        """Describe the server configuration (default model and endpoint state).

        Returns:
            Short human-readable status text.
        """
        settings = get_settings()
        model = resolve_default_model(self.default_model)
        has_key = bool(settings.blablador_api_key)
        has_url = bool(settings.blablador_base_url)
        if has_key and has_url:
            endpoint = "configured"
        else:
            endpoint = "MISSING (set BLABLADOR_API_KEY / BLABLADOR_API_BASE)"
        return f"HeLLMholtz MCP server. Default model: {model}. Blablador endpoint: {endpoint}."

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run one chat completion, converting failures into readable text."""
        from hellmholtz.client import chat

        effective = resolve_default_model(model or self.default_model)
        try:
            return chat(
                effective,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001 - MCP tools return errors as text
            logger.error(f"MCP chat failed for {effective}: {e}")
            return f"Error calling model '{effective}': {e}"
