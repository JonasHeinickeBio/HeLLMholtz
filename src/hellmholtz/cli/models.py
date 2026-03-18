"""Models command group: models, check, monitor."""

import logging

import typer

from hellmholtz.cli.common import format_token_limit, handle_error

logger = logging.getLogger(__name__)


def register_models_commands(app: typer.Typer) -> None:
    """Register models commands to the app."""

    @app.command()
    def models() -> None:
        """List available models from Blablador with token limits."""
        _models_impl()

    @app.command()
    def check(
        model: str = typer.Argument(
            ..., help="Model to check (e.g., openai:gpt-4o, blablador:gpt-4o)"
        ),
    ) -> None:
        """Check if a model is available and can respond to requests."""
        _check_impl(model)

    @app.command()
    def monitor(
        test_accessibility: bool = typer.Option(
            False, help="Test actual accessibility of configured models (slower)"
        ),
        save_report: bool = typer.Option(True, help="Save report to file in reports/ directory"),
    ) -> None:
        """Monitor Blablador model availability and configuration consistency.

        Checks which models are available in the API versus configured locally,
        identifies mismatches, and provides recommendations for keeping the
        configuration up-to-date.
        """
        _monitor_impl(test_accessibility, save_report)


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


def _monitor_impl(test_accessibility: bool, save_report: bool) -> None:
    """Implementation for monitor command."""
    from hellmholtz.monitoring import ModelAvailabilityMonitor

    try:
        monitor = ModelAvailabilityMonitor()
        analysis = monitor.analyze_availability(test_accessibility=test_accessibility)
        report = monitor.generate_report(analysis, test_accessibility=test_accessibility)

        typer.echo(report)

        if save_report:
            filepath = monitor.save_report(report)
            typer.echo(f"\n💾 Report saved to: {filepath}")

    except Exception as e:
        handle_error(e, "Monitoring error")
