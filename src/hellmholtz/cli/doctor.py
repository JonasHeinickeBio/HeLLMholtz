"""Doctor command: configuration and connectivity diagnostics (`hellm doctor`)."""

from __future__ import annotations

from rich.console import Console
import typer

from hellmholtz.cli.common import configure_logging, handle_error
from hellmholtz.diagnostics import (
    CheckResult,
    _http_get_with_headers,
    api_paths,
    check_chat,
    check_credentials,
    check_endpoint,
    check_mcp_extra,
    check_model_available,
    check_python,
    check_version,
)

console = Console()


def _render(results: list[CheckResult]) -> int:
    """Print the report and return the process exit code (0 = healthy)."""
    width = max(len(r.name) for r in results)
    failures = 0
    for r in results:
        style = "green" if r.ok else ("yellow" if r.warn else "red")
        console.print(f"[{style}]{r.status:<4}[/] {r.name:<{width}}  {r.detail}")
        if not r.ok and not r.warn:
            failures += 1
    if failures:
        console.print(
            f"\n[red]{failures} check(s) failed.[/] Run [bold]hellm setup[/] to configure."
        )
        return 1
    if any(not r.ok for r in results):
        console.print("\n[yellow]Healthy with warnings.[/]")
        return 0
    console.print("\n[green]All checks passed.[/]")
    return 0


def register_doctor_commands(app: typer.Typer) -> None:
    """Register diagnostics commands on ``app``."""

    @app.command()
    def doctor(
        model: str | None = typer.Option(
            None, "--model", help="Model for the live chat probe (default: your default model)"
        ),
        skip_chat: bool = typer.Option(
            False, "--skip-chat", help="Skip the live chat round-trip probe"
        ),
        timeout: float = typer.Option(5.0, "--timeout", help="Network timeout in seconds"),
    ) -> None:
        """Diagnose HeLLMholtz configuration and Blablador connectivity."""
        configure_logging()
        try:
            from hellmholtz.core.config import get_settings
            from hellmholtz.mcp.tools import resolve_default_model

            settings = get_settings()
            results = [
                check_python(),
                check_version(),
                check_mcp_extra(),
                check_credentials(settings.blablador_api_key),
            ]
            target_model = resolve_default_model(model)
            endpoint = check_endpoint(
                settings.blablador_base_url, timeout=timeout, api_key=settings.blablador_api_key
            )
            results.append(endpoint)
            if endpoint.ok and settings.blablador_base_url:
                models_url, _ = api_paths(settings.blablador_base_url)
                try:
                    body = _http_get_with_headers(
                        models_url,
                        timeout,
                        {"Authorization": f"Bearer {settings.blablador_api_key}"},
                    )
                    results.append(check_model_available(body, target_model))
                except Exception as exc:  # noqa: BLE001 - listing is best-effort
                    results.append(
                        CheckResult(
                            "model", True, f"could not list models: {_brief(exc)}", warn=True
                        )
                    )
            if skip_chat:
                results.append(
                    CheckResult("chat round-trip", True, "skipped (--skip-chat)", warn=True)
                )
            else:
                results.append(
                    check_chat(
                        settings.blablador_api_key,
                        settings.blablador_base_url,
                        target_model,
                        timeout=timeout,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - report, never traceback
            handle_error(exc, "while running diagnostics")
        raise typer.Exit(code=_render(results))


def _brief(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
