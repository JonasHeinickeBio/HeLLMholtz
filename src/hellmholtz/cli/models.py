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
        """List available models from Blablador with token limits."""
        _models_impl()

    @models_app.command()
    def available() -> None:
        """List available models from Blablador API."""
        _models_impl()

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


def _models_impl() -> None:
    """Implementation for models command."""
    from hellmholtz.providers.blablador import list_models
    from hellmholtz.providers.blablador_config import get_token_limit

    try:
        models = list_models()
        header = (
            f"{'ID':<5} | {'Name':<35} | {'Alias':<10} | {'Source':<10} | "
            f"{'Tokens':<8} | {'Description'}"
        )
        typer.echo(header)
        typer.echo("-" * 110)
        for model in models:
            alias = model.alias if model.alias else ""
            token_limit = get_token_limit(model.name)
            token_display = format_token_limit(token_limit)

            # If ID is same as Name (fallback), just show Name
            if model.id == model.name:
                typer.echo(
                    f"{'':<5} | {model.name:<35} | {alias:<10} | "
                    f"{model.source:<10} | {token_display:<8} | {model.description}"
                )
            else:
                typer.echo(
                    f"{model.id:<5} | {model.name:<35} | {alias:<10} | "
                    f"{model.source:<10} | {token_display:<8} | {model.description}"
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

    from hellmholtz.providers.blablador_config import sync_models

    console = Console()

    try:
        console.print(
            "[bold cyan]🔄 Synchronizing models between API and configuration...[/bold cyan]\n"
        )

        # Get sync results
        result = sync_models(
            api_key=None,
            api_base=None,
            auto_update=not dry_run,  # Update config if not in dry-run mode
            dry_run=dry_run,
        )

        # Display summary
        summary = result.get("summary", {})
        console.print("[bold]📊 Sync Summary:[/bold]")
        console.print(f"  API Models: {summary.get('api_models_count', 0)}")
        console.print(f"  Configured Models: {summary.get('config_models_count', 0)}")
        console.print(f"  New Models: {summary.get('new_count', 0)}")
        console.print(f"  Models Marked Unavailable: {summary.get('removed_count', 0)}")
        console.print(f"  Unchanged Models: {summary.get('unchanged_count', 0)}")
        console.print(f"  Status: {summary.get('sync_status', 'unknown')}")
        console.print()

        # Display detailed actions
        if result.get("actions"):
            console.print("[bold]📝 Actions:[/bold]")
            for action in result["actions"]:
                console.print(action)
            console.print()

        # Handle dry-run
        if dry_run:
            if result.get("would_update"):
                console.print("[yellow]⚠️  This is a dry-run. No changes will be made.[/yellow]")
                console.print(
                    "[green]💡 Use 'hellm models sync' without --dry-run to apply changes[/green]"
                )
            else:
                console.print("[green]✅ Configuration is up-to-date. No changes needed.[/green]")
        else:
            if summary.get("sync_status") == "up-to-date":
                console.print("[green]✅ Configuration is already up-to-date.[/green]")
            else:
                console.print(
                    "[yellow]⚠️  Run with --dry-run to preview changes before applying[/yellow]"
                )
                console.print("[green]💡 Configuration sync completed.[/green]")

    except Exception as e:
        handle_error(e, "Model sync error")
