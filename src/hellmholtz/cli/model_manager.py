"""CLI commands for Blablador Model Manager and Config Exporters."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
import typer

from hellmholtz.core.exporters import get_exporter, list_exporters
from hellmholtz.core.model_manager import BlabladorManager

manager_app = typer.Typer(help="Blablador Model Manager - Discover and configure AI tools")
console = Console()


@manager_app.command()
def list(
    api_base: str | None = typer.Option(None, help="Blablador API base URL"),
    api_key: str | None = typer.Option(None, help="Blablador API key"),
    search: str | None = typer.Option(None, "-s", "--search", help="Filter models"),
) -> None:
    """List available Blablador models."""
    manager = BlabladorManager(api_base=api_base, api_key=api_key)
    models = manager.fetch_models(use_cache=False)

    if search:
        models = manager.search_models(search)

    if not models:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title=f"Blablador Models ({len(models)} available)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Context", justify="right")
    table.add_column("Max Output", justify="right")
    table.add_column("Provider", style="dim")

    for model in models:
        ctx = f"{model.context_length:,}" if model.context_length else "-"
        max_out = f"{model.max_output_tokens:,}" if model.max_output_tokens else "-"
        table.add_row(model.id, model.name, ctx, max_out, model.provider)

    console.print(table)


@manager_app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    api_base: str | None = typer.Option(None, help="Blablador API base URL"),
    api_key: str | None = typer.Option(None, help="Blablador API key"),
) -> None:
    """Search models by name or description."""
    manager = BlabladorManager(api_base=api_base, api_key=api_key)
    manager.fetch_models()
    models = manager.search_models(query)

    if not models:
        console.print(f"[yellow]No models matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results for '{query}'")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Context", justify="right")
    table.add_column("Provider", style="dim")

    for model in models:
        ctx = f"{model.context_length:,}" if model.context_length else "-"
        table.add_row(model.id, model.name, ctx, model.provider)

    console.print(table)


@manager_app.command()
def info(
    model_id: str = typer.Argument(help="Model ID"),
    api_base: str | None = typer.Option(None, help="Blablador API base URL"),
    api_key: str | None = typer.Option(None, help="Blablador API key"),
) -> None:
    """Get detailed information about a model."""
    manager = BlabladorManager(api_base=api_base, api_key=api_key)
    manager.fetch_models()
    model = manager.get_model(model_id)

    if not model:
        console.print(f"[red]Model '{model_id}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Model: {model.id}[/bold cyan]")
    console.print(f"  Name: {model.name}")
    console.print(f"  Provider: {model.provider}")
    if model.description:
        console.print(f"  Description: {model.description}")
    if model.context_length:
        console.print(f"  Context Length: {model.context_length:,}")
    if model.max_output_tokens:
        console.print(f"  Max Output Tokens: {model.max_output_tokens:,}")


@manager_app.command()
def export(
    tool: str = typer.Argument(
        help="Target tool: opencode, claude-code, continue, aider, "
        "cursor, generic-openai, hermes, jan, langchain, gpt4all, pi"
    ),
    models: str = typer.Option(
        "alias-code",
        "-m",
        "--models",
        help="Comma-separated model IDs (first is primary)",
    ),
    api_base: str | None = typer.Option(None, help="Blablador API base URL"),
    api_key: str | None = typer.Option(None, help="Blablador API key"),
    output: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    no_merge: bool = typer.Option(False, "--no-merge", help="Don't merge with existing config"),
) -> None:
    """Export configuration for an AI tool."""
    try:
        exporter = get_exporter(tool)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    manager = BlabladorManager(api_base=api_base, api_key=api_key)
    manager.fetch_models()

    # Parse model IDs
    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    model_configs = []

    for model_id in model_ids:
        model = manager.get_model(model_id)
        if not model:
            console.print(f"[yellow]Warning: Model '{model_id}' not found, skipping[/yellow]")
            continue
        model_config = manager.create_model_config(model, api_key=api_key)
        model_configs.append(model_config)

    if not model_configs:
        console.print("[red]No valid models to export[/red]")
        raise typer.Exit(1)

    from pathlib import Path

    output_path = Path(output) if output else None
    result_path = exporter.export(model_configs, output_path=output_path, merge=not no_merge)

    console.print(f"[green]✓ Configuration exported to {result_path}[/green]")
    console.print(f"  Tool: {exporter.tool_name}")
    console.print(f"  Models: {', '.join(m.name for m in model_configs)}")


@manager_app.command()
def tools() -> None:
    """List all supported AI tools for export."""
    exporters = list_exporters()

    TOOL_DESCRIPTIONS = {
        "opencode": "OpenCode (JSON config)",
        "claude-code": "Claude Code (settings.json)",
        "continue": "Continue.dev (YAML config.yaml)",
        "aider": "Aider (.aider.conf.yml)",
        "cursor": "Cursor (.env file)",
        "generic-openai": "Generic OpenAI-compatible (JSON)",
        "hermes": "Hermes Agent (config.json)",
        "jan": "Jan.AI (models provider)",
        "langchain": "LangChain (env vars)",
        "gpt4all": "GPT4All (reference config)",
        "pi": "Pi Agent (models.json)",
    }

    table = Table(title="Supported AI Tools")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Config File", style="dim")

    for name in exporters:
        exporter = get_exporter(name)
        description = TOOL_DESCRIPTIONS.get(name, name)
        table.add_row(name, description, str(exporter.config_path))

    console.print(table)


def register_model_manager_commands(parent_app: typer.Typer) -> None:
    """Register model manager commands with the main app."""
    parent_app.add_typer(manager_app, name="manager", help="Blablador Model Manager commands")
