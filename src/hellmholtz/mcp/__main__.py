"""Allow running the MCP server directly: ``python -m hellmholtz.mcp``.

Useful from a source checkout when the ``hellm`` script is not installed.
Environment overrides for MCP clients:

- ``HELLM_MCP_MODEL``: default model for tool calls.
- ``HELLM_MCP_TRANSPORT``: ``stdio`` (default) or ``streamable-http``.
- ``HELLM_MCP_HOST`` / ``HELLM_MCP_PORT``: bind address for HTTP transport.
"""

import os

from hellmholtz.mcp.server import run_server


def main() -> None:
    run_server(
        default_model=os.getenv("HELLM_MCP_MODEL"),
        transport=os.getenv("HELLM_MCP_TRANSPORT", "stdio"),
        host=os.getenv("HELLM_MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("HELLM_MCP_PORT", "8765")),
    )


if __name__ == "__main__":
    main()
