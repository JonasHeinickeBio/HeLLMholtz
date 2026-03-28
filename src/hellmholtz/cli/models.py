"""Models command group: models, check, monitor."""

import logging
import time

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
        update_status: bool = typer.Option(
            True, help="Run full availability check and update models_status.yaml"
        ),
    ) -> None:
        """Monitor Blablador model availability and configuration consistency.

        Checks which models are available in the API versus configured locally,
        identifies mismatches, and provides recommendations for keeping the
        configuration up-to-date. Saves a Markdown report to reports/.
        """
        _monitor_impl(test_accessibility, save_report, update_status)


# ============================================================================
# Implementation Functions
# ============================================================================


def _models_impl() -> None:
    """Implementation for models command."""
    from hellmholtz.providers.blablador import list_models
    from hellmholtz.providers.blablador_config import get_token_limit

    try:
        models = list_models()

        typer.echo(f"\n{'='*100}")
        typer.echo(f"{'Blablador Models':^100}")
        typer.echo(f"{'='*100}")

        col_id = 5
        col_name = 38
        col_alias = 12
        col_src = 10
        col_tok = 8
        col_desc = 24

        header = (
            f"{'ID':<{col_id}} | {'Name':<{col_name}} | {'Alias':<{col_alias}} | "
            f"{'Source':<{col_src}} | {'Tokens':<{col_tok}} | Description"
        )
        typer.echo(header)
        typer.echo("-" * 115)

        for model in models:
            alias = model.alias if model.alias else ""
            token_limit = get_token_limit(model.name)
            token_display = format_token_limit(token_limit)
            desc = model.description or ""
            if len(desc) > col_desc:
                desc = desc[: col_desc - 1] + "…"

            display_id = "" if model.id == model.name else model.id
            typer.echo(
                f"{display_id:<{col_id}} | {model.name:<{col_name}} | {alias:<{col_alias}} | "
                f"{model.source:<{col_src}} | {token_display:<{col_tok}} | {desc}"
            )

        typer.echo(f"{'='*100}")
        typer.echo(f"Total: {len(models)} models  |  Use 'blablador:<name>' in commands")
        typer.echo(f"{'='*100}\n")
    except Exception as e:
        handle_error(e, "Model list error")


def _check_impl(model: str) -> None:
    """Implementation for check command."""
    from hellmholtz.client import check_model_availability
    from hellmholtz.providers.blablador_config import get_model_by_name, get_token_limit

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Model Check: {model}")
    typer.echo(f"{'='*60}")

    # Show static metadata if available for blablador models
    model_name = model.split(":", 1)[-1] if ":" in model else model
    known = get_model_by_name(model_name)
    if known:
        typer.echo(f"  Name        : {known.name}")
        if known.alias:
            typer.echo(f"  Alias       : {known.alias}")
        typer.echo(f"  Description : {known.description or 'N/A'}")
        typer.echo(f"  Source      : {known.source or 'Blablador'}")
        typer.echo(f"  Max Tokens  : {format_token_limit(known.max_context_tokens)}")
    else:
        token_limit = get_token_limit(model)
        typer.echo(f"  Max Tokens  : {format_token_limit(token_limit)}")

    typer.echo(f"{'─'*60}")
    typer.echo("  Testing availability …")

    try:
        t0 = time.time()
        is_available = check_model_availability(model)
        elapsed = time.time() - t0

        if is_available:
            typer.echo(f"  Status      : ✅ Available  ({elapsed:.2f}s)")
            typer.echo(f"{'='*60}\n")
        else:
            typer.echo("  Status      : ❌ Not available or not responding")
            typer.echo(f"{'='*60}\n")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        handle_error(e, "Model check error")


def _monitor_impl(test_accessibility: bool, save_report: bool, update_status: bool) -> None:
    """Implementation for monitor command."""
    from hellmholtz.monitoring import ModelAvailabilityMonitor

    try:
        monitor = ModelAvailabilityMonitor()

        # Run full availability check (updates models_status.yaml)
        if update_status:
            typer.echo("🔄 Running full model availability check …")
            monitor.check_all_models_automatically()

        analysis = monitor.analyze_availability(test_accessibility=test_accessibility)

        # Print structured summary table to terminal
        _print_monitor_summary(analysis, test_accessibility)

        # Generate and save Markdown report
        if save_report:
            md_report = _generate_monitor_markdown(analysis, test_accessibility)
            filepath = monitor.save_report(md_report, filename="model_availability_report.md")
            typer.echo(f"\n💾 Markdown report saved to: {filepath}")

    except Exception as e:
        handle_error(e, "Monitoring error")


def _print_monitor_summary(analysis: dict, test_accessibility: bool) -> None:
    """Print a structured monitor summary to the terminal."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    typer.echo(f"\n{'='*70}")
    typer.echo(f"{'Blablador Model Availability Report':^70}")
    typer.echo(f"  Generated: {timestamp}")
    typer.echo(f"{'='*70}")

    typer.echo("\n📈 Summary")
    typer.echo(f"  API models found      : {analysis['api_models_count']}")
    typer.echo(f"  Configured models     : {analysis['configured_models_count']}")
    typer.echo(f"  Available & configured: {len(analysis['configured_and_available'])}")
    typer.echo(f"  Configured but missing: {len(analysis['configured_not_available'])}")
    typer.echo(f"  New (unconfigured)    : {len(analysis['available_not_configured'])}")

    _print_available_models(analysis, test_accessibility)
    _print_missing_models(analysis)
    _print_unconfigured_models(analysis)

    typer.echo("\n💡 Recommendations")
    if not analysis["configured_not_available"] and not analysis["available_not_configured"]:
        typer.echo("  ✅ Configuration is fully in sync with the API")
    else:
        if analysis["configured_not_available"]:
            typer.echo("  • Remove models no longer in the API from blablador_config.py")
        if analysis["available_not_configured"]:
            typer.echo("  • Add newly available models to blablador_config.py")
    typer.echo(f"{'='*70}\n")


def _print_available_models(analysis: dict, test_accessibility: bool) -> None:
    """Print the available & configured models section."""
    if not analysis["configured_and_available"]:
        return
    typer.echo("\n✅ Available & Configured Models")
    typer.echo(f"  {'Model Name':<40} {'Accessible':<12} {'Latency':>8}")
    typer.echo(f"  {'─'*40} {'─'*11} {'─'*8}")
    for _api_id, model in analysis["configured_and_available"]:
        accessible_str = ""
        latency_str = ""
        if test_accessibility and model.name in analysis["accessibility_results"]:
            result = analysis["accessibility_results"][model.name]
            accessible_str = "✅ yes" if result["accessible"] else "❌ no"
            latency_str = f"{result['latency']:.2f}s" if result.get("latency") else ""
        typer.echo(f"  {model.name:<40} {accessible_str:<12} {latency_str:>8}")


def _print_missing_models(analysis: dict) -> None:
    """Print the configured-but-missing models section."""
    if not analysis["configured_not_available"]:
        return
    typer.echo("\n⚠️  Configured but Not Available (consider removing from config)")
    for api_id, model in analysis["configured_not_available"]:
        typer.echo(f"  • {model.name}  (API ID: {api_id})")


def _print_unconfigured_models(analysis: dict) -> None:
    """Print the available-but-unconfigured models section."""
    if not analysis["available_not_configured"]:
        return
    typer.echo("\n🔍 Available but Not Configured (consider adding to blablador_config.py)")
    for _api_id, api_model in analysis["available_not_configured"]:
        model_id = api_model.get("id", "unknown")
        typer.echo(f"  • {model_id}")


def _generate_monitor_markdown(analysis: dict, test_accessibility: bool) -> str:
    """Generate a well-formatted Markdown availability report."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lines = [
        "# Blablador Model Availability Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| API models found | {analysis['api_models_count']} |",
        f"| Configured models | {analysis['configured_models_count']} |",
        f"| Available & configured | {len(analysis['configured_and_available'])} |",
        f"| Configured but missing | {len(analysis['configured_not_available'])} |",
        f"| New (unconfigured) | {len(analysis['available_not_configured'])} |",
        "",
    ]

    lines += _md_available_section(analysis, test_accessibility)
    lines += _md_missing_section(analysis)
    lines += _md_unconfigured_section(analysis)
    lines += _md_recommendations_section(analysis)
    lines.append("")

    return "\n".join(lines)


def _md_available_section(analysis: dict, test_accessibility: bool) -> list[str]:
    """Markdown section for available & configured models."""
    if not analysis["configured_and_available"]:
        return []
    lines: list[str] = ["## ✅ Available & Configured Models", ""]
    if test_accessibility:
        lines += [
            "| Model | API ID | Accessible | Latency |",
            "|-------|--------|------------|---------|",
        ]
        for api_id, model in analysis["configured_and_available"]:
            result = analysis["accessibility_results"].get(model.name, {})
            accessible = "✅ yes" if result.get("accessible") else "❌ no"
            latency = f"{result['latency']:.2f}s" if result.get("latency") else "—"
            lines.append(f"| {model.name} | `{api_id}` | {accessible} | {latency} |")
    else:
        lines += ["| Model | API ID | Description |", "|-------|--------|-------------|"]
        for api_id, model in analysis["configured_and_available"]:
            lines.append(f"| {model.name} | `{api_id}` | {model.description or ''} |")
    lines.append("")
    return lines


def _md_missing_section(analysis: dict) -> list[str]:
    """Markdown section for configured-but-missing models."""
    if not analysis["configured_not_available"]:
        return []
    lines: list[str] = [
        "## ⚠️ Configured but Not Available",
        "",
        "These models are in `blablador_config.py` but not returned by the API.",
        "Consider removing them from the configuration.",
        "",
        "| Model | API ID |",
        "|-------|--------|",
    ]
    for api_id, model in analysis["configured_not_available"]:
        lines.append(f"| {model.name} | `{api_id}` |")
    lines.append("")
    return lines


def _md_unconfigured_section(analysis: dict) -> list[str]:
    """Markdown section for available-but-unconfigured models."""
    if not analysis["available_not_configured"]:
        return []
    lines: list[str] = [
        "## 🔍 Available but Not Configured",
        "",
        "These models are returned by the API but not in `blablador_config.py`.",
        "Consider adding them with their token limits and descriptions.",
        "",
        "| API ID |",
        "|--------|",
    ]
    for _api_id, api_model in analysis["available_not_configured"]:
        model_id = api_model.get("id", "unknown")
        lines.append(f"| `{model_id}` |")
    lines.append("")
    return lines


def _md_recommendations_section(analysis: dict) -> list[str]:
    """Markdown recommendations section."""
    lines: list[str] = ["## 💡 Recommendations", ""]
    if not analysis["configured_not_available"] and not analysis["available_not_configured"]:
        lines.append("✅ Configuration is fully in sync with the Blablador API.")
    else:
        if analysis["configured_not_available"]:
            lines.append(
                "- Remove models that are no longer available from `blablador_config.py`."
            )
        if analysis["available_not_configured"]:
            lines.append(
                "- Add newly available models to `blablador_config.py` with correct metadata."
            )
    return lines
