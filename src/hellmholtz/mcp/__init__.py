"""MCP (Model Context Protocol) server for HeLLMholtz.

Exposes HeLLMholtz models as MCP tools so MCP-capable clients (Claude
Desktop, Claude Code, Cherry Studio, ...) can offload heavy tasks to the
Blablador endpoint.

The ``mcp`` package is an optional dependency. Install the extra with
``pip install "hellmholtz[mcp]"`` (or ``poetry install --extras mcp``)
before running the server. Tool logic in :mod:`hellmholtz.mcp.tools` does
not require ``mcp`` to be installed.
"""

from hellmholtz.mcp.tools import DEFAULT_MODEL_ENV_VAR, HellmTools

__all__ = ["DEFAULT_MODEL_ENV_VAR", "HellmTools", "create_server", "run_server"]


def __getattr__(name: str) -> object:  # lazy re-export (needs `mcp` extra)
    """Lazily expose server helpers so importing the package never needs `mcp`."""
    if name in ("create_server", "run_server"):
        from hellmholtz.mcp import server

        return getattr(server, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
