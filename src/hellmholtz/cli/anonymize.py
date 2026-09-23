"""Anonymizer command group: PII anonymization and prompt benchmarking."""

import json
import logging
from pathlib import Path
import sys

import typer

from hellmholtz.anonymizer import (
    DEFAULT_MODEL,
    PROMPTS,
    Anonymizer,
    compare_prompts,
    write_reports,
)

logger = logging.getLogger(__name__)

_PROMPT_CHOICES = typer.Option("default", help=f"Prompt variant: {', '.join(sorted(PROMPTS))}")
_MODEL_OPTION = typer.Option(DEFAULT_MODEL, help="Model name (provider:model)")
_PATH_ARGUMENT = typer.Argument(None, help="Input file; reads from stdin when omitted")
_OUT_OPTION = typer.Option(None, "--out", help="Output path (implies --write)")
_INPUTS_ARGUMENT = typer.Argument(
    ..., help="Input files and/or directories (recurses for text files)"
)
_GROUND_TRUTH_OPTION = typer.Option(
    None, "--ground-truth", help="JSON: {text_name: [{type, value}, ...]}"
)
_REPORT_DIR_OPTION = typer.Option(
    Path("reports/anonymizer"), "--report-dir", help="Report output directory"
)
_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".tex", ".csv", ".json", ".yaml", ".yml"}


def _read_input(path: Path | None) -> str:
    """Read text from a file argument or stdin."""
    if path is None:
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def register_anonymize_commands(app: typer.Typer) -> None:  # noqa: C901
    """Register anonymizer commands to the app."""

    @app.command(name="anonymize")
    def anonymize_cmd(  # noqa: C901
        path: Path | None = _PATH_ARGUMENT,
        model: str = _MODEL_OPTION,
        prompt: str = _PROMPT_CHOICES,
        temperature: float = typer.Option(0.0, help="Sampling temperature"),
        validate: bool = typer.Option(
            True, "--validate/--no-validate", help="Run leak validation"
        ),
        fuzzy: bool = typer.Option(True, "--fuzzy/--no-fuzzy", help="Enable fuzzy leak matching"),
        max_chunk_chars: int = typer.Option(20000, help="Soft chunk size limit in characters"),
        write: bool = typer.Option(False, help="Write anonymized text next to the input file"),
        out: Path | None = _OUT_OPTION,
        as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON"),
        show_entities: bool = typer.Option(
            False, help="Print the entity table after the anonymized text"
        ),
    ) -> None:
        """Anonymize PII in a document using the ShinrAI PII 1.3 model."""
        from hellmholtz.cli.common import handle_error

        try:
            text = _read_input(path)
            if not text.strip():
                handle_error(ValueError("Empty input"), "Nothing to anonymize")

            anon = Anonymizer(
                model=model,
                temperature=temperature,
                validate=validate,
                fuzzy=fuzzy,
                max_chunk_chars=max_chunk_chars,
                prompt=prompt,
            )
            result = anon.anonymize(text)

            if as_json:
                typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                typer.echo(result.text)
                if show_entities:
                    typer.echo("")
                    typer.echo(f"Entities ({len(result.entities)}):")
                    for e in result.entities:
                        conf = f" {e.confidence:.2f}" if e.confidence is not None else ""
                        typer.echo(f"  {e.entity_type}: {e.original} -> {e.replacement}{conf}")

            if (write or out is not None) and path is not None:
                name = f"{path.stem}.anonymized{path.suffix}"
                target = out if out is not None else path.with_name(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(result.text + "\n", encoding="utf-8")
                typer.echo(f"Anonymized text written to {target}", err=True)

            if result.validation is not None:
                if result.validation.is_anonymous:
                    typer.echo(
                        f"Validation: {result.validation.summary} in {result.duration_ms:.0f} ms",
                        err=True,
                    )
                else:
                    typer.echo(
                        f"WARNING: {result.validation.summary}",
                        err=True,
                    )
                    for leak in result.validation.leaks:
                        typer.echo(f"  {leak}", err=True)
        except typer.Exit:
            raise
        except Exception as e:
            handle_error(e, "Anonymization error")

    @app.command(name="anonymize-benchmark")
    def anonymize_benchmark_cmd(
        inputs: list[Path] = _INPUTS_ARGUMENT,
        model: str = _MODEL_OPTION,
        prompt: str = typer.Option(
            None, "--prompt", "-p", help="Prompt variant(s), comma-separated; default: all"
        ),
        iterations: int = typer.Option(1, help="Runs per variant and per text"),
        ground_truth: Path | None = _GROUND_TRUTH_OPTION,
        report_dir: Path = _REPORT_DIR_OPTION,
        validate: bool = typer.Option(True, "--validate/--no-validate"),
    ) -> None:
        """Benchmark and compare anonymizer prompt variants on real documents."""
        from hellmholtz.cli.common import handle_error

        try:
            texts: dict[str, str] = {}
            for item in inputs:
                if item.is_file():
                    candidates = [item]
                else:
                    candidates = sorted(
                        p
                        for p in item.rglob("*")
                        if p.is_file() and p.suffix.lower() in _TEXT_EXTENSIONS
                    )
                for p in candidates:
                    texts[p.stem] = p.read_text(encoding="utf-8")
            if not texts:
                handle_error(ValueError("No input text files found"), "Benchmark setup")
            typer.echo(f"Benchmarking {len(texts)} text(s): {', '.join(texts)}", err=True)

            prompt_names = [p.strip() for p in prompt.split(",") if p.strip()] if prompt else None

            gt: dict[str, list[dict[str, str]]] | None = None
            if ground_truth is not None:
                gt = json.loads(ground_truth.read_text(encoding="utf-8"))

            comparison = compare_prompts(
                texts,
                prompt_names,
                model=model,
                iterations=iterations,
                ground_truth=gt,
                validate=validate,
            )
            md_path, json_path = write_reports(comparison, out_dir=report_dir)
            typer.echo(f"Markdown report: {md_path}", err=True)
            typer.echo(f"JSON report:     {json_path}", err=True)
        except typer.Exit:
            raise
        except Exception as e:
            handle_error(e, "Anonymizer benchmark error")
