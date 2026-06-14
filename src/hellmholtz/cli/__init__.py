"""HeLLMholtz CLI - modularized command groups.

This package organizes CLI commands into logical groups:
- chat: Direct chat interactions
- benchmark: Benchmarking and reporting (bench, report, chart, analyze)
- models: Model management and monitoring (models, check, monitor)
- integrations: Third-party integrations (lm_eval, proxy, bench_throughput,
  anythingllm chat/upload/workspaces/ping)
- rag: Retrieval-Augmented Generation (rag ingest, rag chat)
"""

import typer

from hellmholtz.cli.benchmark import register_benchmark_commands
from hellmholtz.cli.chat import register_chat_commands
from hellmholtz.cli.common import configure_logging
from hellmholtz.cli.integrations import register_integration_commands
from hellmholtz.cli.models import register_models_commands
from hellmholtz.cli.rag import register_rag_commands

__all__ = ["app", "main"]


def create_app() -> typer.Typer:
    """Create and configure the main Typer application with all command groups."""
    app = typer.Typer(help="HeLLMholtz - Unified LLM access, benchmarking, and reporting suite")

    # Register all command groups
    register_chat_commands(app)
    register_benchmark_commands(app)
    register_models_commands(app)
    register_integration_commands(app)
    register_rag_commands(app)

    return app


app = create_app()


def main() -> None:
    """Main entry point for the HeLLMholtz CLI application.

    Configures logging and launches the Typer app.
    """
    configure_logging()
    app()


if __name__ == "__main__":
    main()
