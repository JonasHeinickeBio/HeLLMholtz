"""Models command group: models, check, monitor, sync, available."""

import logging

import typer

from hellmholtz.cli.common import format_token_limit, handle_error

logger = logging.getLogger(__name__)


def register_models_commands(app: typer.Typer) -> None:
    """Register models commands to the app."""

    # Create models subcommand group
    models_app = typer.Typer(help="Model management and monitoring commands")

    @models_app.command(name="list")
    def models_list() -> None:
        """List all configured models from local config with availability status."""
        _list_impl()

    @models_app.command()
    def available() -> None:
        """List only models currently available from Blablador API."""
        _available_impl()

    @models_app.command()
    def check(
        model: str = typer.Argument(
            ..., help="Model to check (e.g., openai:gpt-4o, blablador:gpt-4o)"
        ),
    ) -> None:
        """Check if a model is available and can respond to requests."""
        _check_impl(model)

    @models_app.command()
    def monitor(
        test_accessibility: bool = typer.Option(
            False, help="Test actual accessibility of configured models (slower)"
        ),
        save_report: bool = typer.Option(True, help="Save report to file in reports/ directory"),
        auto_sync: bool = typer.Option(
            False,
            "--auto-sync",
            help="Automatically sync configuration with API models (add new, remove unavailable)",
        ),
    ) -> None:
        """Monitor Blablador model availability and configuration consistency.

        Checks which models are available in the API versus configured locally,
        identifies mismatches, and provides recommendations for keeping the
        configuration up-to-date.

        Use --auto-sync to automatically update the configuration file with
        new models from the API and remove models that are no longer available.
        """
        _monitor_impl(test_accessibility, save_report, auto_sync)

    @models_app.command("agent")
    def auto_agent(
        test_accessibility: bool = typer.Option(
            True, help="Test live accessibility/latency for each available model"
        ),
    ) -> None:
        """Run automatic model sync + token-limit + search-config workflow.

        This command is fully automatic and produces:
        - model status update,
        - search configuration YAML,
        - concise sync report.
        """
        _auto_agent_impl(test_accessibility)

    @models_app.command()
    def sync(
        dry_run: bool = typer.Option(
            False, help="Show what would change without making modifications"
        ),
        test_accessibility: bool = typer.Option(False, help="Test actual accessibility of models"),
    ) -> None:
        """Synchronize models between API and local configuration.

        Compares models available in the API with those in the local configuration
        and provides a report of differences.

        With --dry-run, shows what would change without making any modifications.
        """
        _sync_impl(dry_run, test_accessibility)

    # Add the models group to the main app
    app.add_typer(models_app, name="models", help="Model management and monitoring")


# ============================================================================
# Implementation Functions
# ============================================================================


def _available_impl() -> None:
    """Implementation for available command - shows only API models."""
    from rich.console import Console
    from rich.table import Table

    from hellmholtz.providers.blablador import list_models
    from hellmholtz.providers.blablador_config import get_token_limit

    try:
        console = Console()

        # Get currently available models from API
        try:
            api_models = list_models()
        except Exception as e:
            console.print(f"[yellow]⚠️  Warning: Could not fetch API models: {e}[/yellow]")
            api_models = []

        if not api_models:
            console.print("[yellow]No models available from API[/yellow]")
            return

        # Create a table for API-only models
        table = Table(title="Available Models from Blablador API", show_lines=False)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Name", style="magenta", width=45)
        table.add_column("Source", style="blue", width=12)
        table.add_column("Tokens", style="yellow", width=10)
        table.add_column("Description", style="white")

        for model in sorted(api_models, key=lambda m: (m.name or "")):
            token_limit = get_token_limit(model.name)
            token_display = format_token_limit(token_limit)

            desc = ""
            if hasattr(model, "description") and model.description:
                desc = model.description

            table.add_row(
                str(model.id)[:7],
                (model.name or "")[:43],
                (model.source or "")[:10] if hasattr(model, "source") else "",
                token_display,
                (desc or "")[:50],
            )

        console.print(table)
        console.print(f"\n[i]Total: {len(api_models)} models available from API[/i]")

    except Exception as e:
        handle_error(e, "Model availability check error")


def _list_impl() -> None:
    """Implementation for models command."""
    from rich.console import Console
    from rich.table import Table

    from hellmholtz.providers.blablador import list_models
    from hellmholtz.providers.blablador_config import get_model_by_name, get_token_limit

    try:
        console = Console()

        # Get configured models from config file
        from hellmholtz.providers.blablador_config import KNOWN_MODELS

        configured_names = {m.name for m in KNOWN_MODELS}

        # Get currently available models from API
        try:
            api_models = list_models()
            api_names = {m.name for m in api_models}
        except Exception as api_error:
            console.print(f"[yellow]⚠️  Warning: Could not fetch API models: {api_error}[/yellow]")
            api_models = []
            api_names = set()

        # Create a table with availability status
        table = Table(title="Models Status", show_lines=False)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Name", style="magenta", width=40)
        table.add_column("Source", style="blue", width=12)
        table.add_column("Tokens", style="yellow", width=10)
        table.add_column("Configured", style="cyan", width=12, justify="center")
        table.add_column("Available", style="green", width=12, justify="center")
        table.add_column("Description", style="white")

        # Merge: show all configured models + any API-only models
        all_model_names = configured_names | api_names

        for name in sorted(all_model_names):
            # Get model data
            config_model = get_model_by_name(name)
            api_model = next((m for m in api_models if m.name == name), None)

            # Determine ID (use config ID if available, otherwise API ID)
            model_id = ""
            if config_model and config_model.id:
                model_id = config_model.id
            elif api_model and api_model.id:
                model_id = api_model.id

            # Determine source
            source = (
                api_model.source if api_model else (config_model.source if config_model else "")
            )

            # Get token limit
            token_limit = get_token_limit(name)
            token_display = format_token_limit(token_limit)

            # Availability status
            is_configured = name in configured_names
            is_available = name in api_names

            configured_icon = "[green]✓[/green]" if is_configured else "[yellow]-[/yellow]"
            available_icon = "[green]✓[/green]" if is_available else "[red]✗[/red]"

            # Build description
            description = ""
            if config_model and config_model.description:
                description = config_model.description
            elif api_model and api_model.description:
                description = api_model.description

            table.add_row(
                model_id,
                name[:38],  # Truncate if too long
                source[:10],
                token_display,
                configured_icon,
                available_icon,
                description[:60],  # Truncate if too long
            )

        console.print(table)

        # Print legend
        console.print(
            "\n[i][green]✓[/green] = Available  [red]✗[/red] = Not Available  "
            "[yellow]-[/yellow] = Not Configured[/i]"
        )

    except Exception as e:
        handle_error(e, "Model list error")


def _check_impl(model: str) -> None:
    """Implementation for check command."""
    from hellmholtz.client import check_model_availability

    typer.echo(f"Checking availability of model: {model}")

    try:
        is_available = check_model_availability(model)
        if is_available:
            typer.echo("✅ Model is available and responding")
        else:
            typer.echo("❌ Model is not available or not responding", err=True)
            raise typer.Exit(1)
    except Exception as e:
        handle_error(e, "Model check error")


def _monitor_impl(test_accessibility: bool, save_report: bool, auto_sync: bool) -> None:
    """Implementation for monitor command."""
    from hellmholtz.monitoring import monitor_models

    try:
        # Use the enhanced monitor_models function that supports auto_sync
        report = monitor_models(
            test_accessibility=test_accessibility,
            save_report=save_report,
            auto_sync=auto_sync,
        )
        typer.echo(report)

    except Exception as e:
        handle_error(e, "Monitoring error")


def _auto_agent_impl(test_accessibility: bool) -> None:
    """Implementation for automatic model configuration agent."""
    from hellmholtz.monitoring import ModelAvailabilityMonitor

    try:
        monitor = ModelAvailabilityMonitor()
        result = monitor.run_auto_config_agent(test_accessibility=test_accessibility)
        typer.echo(result["report"])
        typer.echo(f"\n💾 Agent report saved to: {result['report_path']}")
    except Exception as e:
        handle_error(e, "Automatic model agent error")


def _sync_impl(dry_run: bool, test_accessibility: bool) -> None:
    """Implementation for sync command."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from hellmholtz.providers.blablador_config import sync_models

    console = Console()

    try:
        console.print(
            Panel(
                "[bold cyan]🔄 Synchronizing models between API and configuration...[/bold cyan]",
                expand=False,
            )
        )

        # Get sync results
        result = sync_models(
            api_key=None,
            api_base=None,
            auto_update=not dry_run,  # Update config if not in dry-run mode
            dry_run=dry_run,
        )

        # Display summary panel
        summary = result.get("summary", {})

        summary_text = Text()
        summary_text.append("API Models:    ", style="bold cyan")
        summary_text.append(str(summary.get("api_models_count", 0)), style="white")
        summary_text.append("\n")
        summary_text.append("Configured:    ", style="bold cyan")
        summary_text.append(str(summary.get("config_models_count", 0)), style="white")
        summary_text.append("\n")
        summary_text.append("New:           ", style="bold green")
        summary_text.append(str(summary.get("new_count", 0)), style="white")
        summary_text.append("\n")
        summary_text.append("Unavailable:   ", style="bold yellow")
        summary_text.append(str(summary.get("removed_count", 0)), style="white")
        summary_text.append("\n")
        summary_text.append("Unchanged:     ", style="bold blue")
        summary_text.append(str(summary.get("unchanged_count", 0)), style="white")

        console.print(Panel(summary_text, title="📊 Sync Summary", expand=False))

        # Show detailed comparison
        if result.get("api_models") or result.get("config_models"):
            # Get API model names
            api_model_names = set(result.get("api_models", []))

            # Get configured model names (from config_models key)
            config_model_names = set(result.get("config_models", []))

            # Build comparison table
            table = Table(title="_comparison of API vs Config")
            table.add_column("Status", style="bold", width=12)
            table.add_column("Model Name", style="magenta", width=40)
            table.add_column("Source", style="blue", width=10)

            # New models (in API but not in config)
            new_models = result.get("new_models", [])
            for model_id in new_models:
                # Extract model name from API ID
                parts = model_id.split(" - ")
                name = parts[1] if len(parts) >= 2 else model_id
                table.add_row("new", name[:38], "API")

            # Removed models (in config but not in API)
            removed_models = result.get("removed_models", [])
            for name in removed_models:
                table.add_row("unavailable", name[:38], "Config")

            # Unchanged models (in both)
            unchanged_models = result.get("unchanged_models", [])
            for model_id in unchanged_models:
                parts = model_id.split(" - ")
                name = parts[1] if len(parts) >= 2 else model_id
                table.add_row("same", name[:38], "Both")

            console.print(table)

        # Display detailed actions if available
        if result.get("actions"):
            console.print()
            console.print("[bold]📝 Actions:[/bold]")
            for action in result["actions"]:
                console.print(action)

        # Handle dry-run
        if dry_run:
            if result.get("would_update"):
                console.print()
                console.print(
                    Panel(
                        "[yellow]⚠️  This is a dry-run. No changes will be made.[/yellow]\n"
                        "[green]💡 Use 'hellm models sync' without --dry-run to apply changes[/green]",
                        expand=False,
                    )
                )
            else:
                console.print()
                console.print(
                    Panel(
                        "[green]✅ Configuration is up-to-date. No changes needed.[/green]",
                        expand=False,
                    )
                )
        else:
            console.print()
            if summary.get("sync_status") == "up-to-date":
                console.print(
                    Panel(
                        "[green]✅ Configuration is already up-to-date.[/green]",
                        expand=False,
                    )
                )
            else:
                console.print(
                    Panel(
                        "[yellow]⚠️  Run with --dry-run to preview changes before applying[/yellow]\n"
                        "[green]💡 Configuration sync completed.[/green]",
                        expand=False,
                    )
                )

    except Exception as e:
        handle_error(e, "Model sync error")
