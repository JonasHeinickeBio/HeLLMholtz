"""MCP command group: expose HeLLMholtz models as MCP tools."""

import typer

from hellmholtz.cli.common import handle_error


def register_mcp_commands(app: typer.Typer) -> None:
    """Register MCP server commands to the app."""

    @app.command()
    def mcp(
        model: str | None = typer.Option(
            None,
            "--model",
            help="Default model for tool calls without one "
            "(also HELLM_MCP_MODEL env var). See `hellm models`.",
        ),
        transport: str = typer.Option(
            "stdio",
            "--transport",
            help="MCP transport: stdio (Claude Desktop) or streamable-http",
        ),
        host: str = typer.Option("127.0.0.1", help="Bind host for HTTP transports"),
        port: int = typer.Option(8765, help="Bind port for HTTP transports"),
        print_config: bool = typer.Option(
            False,
            "--print-config",
            help="Print a ready-to-paste Claude Desktop MCP config and exit",
        ),
    ) -> None:
        """Run the HeLLMholtz MCP server (tools: ask_external, chat_external, list_models)."""
        _mcp_impl(model, transport, host, port, print_config)


def _mcp_impl(
    model: str | None,
    transport: str,
    host: str,
    port: int,
    print_config: bool,
) -> None:
    """Implementation for the mcp command."""
    from hellmholtz.mcp.tools import claude_desktop_config

    try:
        if print_config:
            typer.echo(claude_desktop_config(model=model))
            return

        # Stdio speaks MCP on stdout, so keep banners on stderr.
        banner = [
            "HeLLMholtz MCP server. Tools: ask_external, chat_external, list_models, get_info.",
            f"Default model: {model or 'auto (HELLM_MCP_MODEL / defaults)'}",
            "Claude Desktop: add via 'hellm mcp --print-config' or "
            "Settings > Developer > Edit Config.",
        ]
        for line in banner:
            typer.echo(line, err=True)

        from hellmholtz.mcp.server import run_server

        run_server(default_model=model, transport=transport, host=host, port=port)
    except RuntimeError as e:
        # Missing optional `mcp` extra surfaces as RuntimeError with a hint.
        handle_error(e, "MCP error")
    except Exception as e:
        handle_error(e, "MCP error")
