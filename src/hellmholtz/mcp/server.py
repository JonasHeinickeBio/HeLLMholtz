"""FastMCP server binding for the HeLLMholtz tools, resources and prompts.

Requires the optional ``mcp`` dependency (``pip install "hellmholtz[mcp]"``).
"""

from collections.abc import Callable
from functools import wraps
import logging
from typing import Any

from hellmholtz.mcp.tools import (
    SERVER_NAME,
    HellmTools,
    prompt_explain,
    prompt_summarize,
    prompt_translate,
)

logger = logging.getLogger(__name__)


def _require_fastmcp() -> Any:
    """Import FastMCP or exit with an actionable message."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            'MCP support requires the mcp package. Install with: pip install "hellmholtz[mcp]"'
        ) from e
    return FastMCP


def create_server(default_model: str | None = None) -> Any:
    """Create a FastMCP server exposing the HeLLMholtz tools, resources and prompts.

    Args:
        default_model: Model used by tool calls that do not name one.

    Returns:
        A configured ``FastMCP`` instance.
    """
    FastMCP = _require_fastmcp()
    mcp = FastMCP(SERVER_NAME)
    tools = HellmTools(default_model=default_model)

    def expose(func: Callable[..., str]) -> Callable[..., str]:
        """Register a HellmTools method as an MCP tool, keeping its schema."""
        exposed = wraps(func)(lambda *args, **kwargs: func(*args, **kwargs))
        mcp.tool()(exposed)
        return exposed

    expose(tools.ask_external)
    expose(tools.chat_external)
    expose(tools.check_model)
    expose(tools.run_doctor)
    expose(tools.list_models)
    expose(tools.get_info)

    def expose_resource(uri: str, mime_type: str, func: Callable[..., str]) -> None:
        """Register a function as an MCP resource without an untyped decorator."""
        mcp.resource(uri, mime_type=mime_type)(func)

    def expose_prompt(
        name: str,
        description: str,
        func: Callable[..., list[dict[str, str]]],
    ) -> None:
        """Register a prompt-message builder as an MCP prompt."""
        mcp.prompt(name=name, description=description)(func)

    def hellm_info() -> str:
        """HeLLMholtz MCP server status: default model and endpoint state."""
        return tools.get_info()

    def hellm_models() -> str:
        """Catalog of model identifiers served by the configured endpoint (JSON)."""
        return tools.models_json()

    def summarize_prompt(text: str, style: str | None = None) -> list[dict[str, str]]:
        return prompt_summarize(text, style)

    def translate_prompt(text: str, target_language: str) -> list[dict[str, str]]:
        return prompt_translate(text, target_language)

    def explain_prompt(text: str, audience: str | None = None) -> list[dict[str, str]]:
        return prompt_explain(text, audience)

    expose_resource("hellm://info", "text/plain", hellm_info)
    expose_resource("hellm://models", "application/json", hellm_models)
    expose_prompt("summarize", "Summarize a text with a HeLLMholtz model.", summarize_prompt)
    expose_prompt(
        "translate",
        "Translate a text into a target language with a HeLLMholtz model.",
        translate_prompt,
    )
    expose_prompt(
        "explain",
        "Explain a text in plain language with a HeLLMholtz model.",
        explain_prompt,
    )

    logger.info(f"Created MCP server '{SERVER_NAME}' (default model: {tools.get_info()})")
    return mcp


def run_server(
    default_model: str | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Create and run the HeLLMholtz MCP server.

    Args:
        default_model: Model used by tool calls that do not name one.
        transport: ``stdio`` (Claude Desktop default) or ``streamable-http``.
        host: Bind host for the HTTP transports.
        port: Bind port for the HTTP transports.
    """
    mcp = create_server(default_model)
    if transport != "stdio":
        # FastMCP reads host/port from its settings before run().
        mcp.settings.host = host
        mcp.settings.port = port
    logger.info(f"Starting HeLLMholtz MCP server (transport={transport})")
    mcp.run(transport=transport)
