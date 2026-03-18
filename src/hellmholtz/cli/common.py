"""Shared CLI utilities, options, and helpers to reduce code duplication."""

import json
import logging
from pathlib import Path
from typing import Any, NoReturn

import typer

from hellmholtz.core.prompts import Message, Prompt

# Module-level logger
logger = logging.getLogger(__name__)

# ============================================================================
# Typer Options (DRY - defined once, reused across commands)
# ============================================================================

PROMPTS_FILE_OPTION = typer.Option(
    None, help="Path to prompts file (.txt for simple text, .json for structured prompts)"
)
PROMPTS_CATEGORY_OPTION = typer.Option(
    None, help="Category of prompts to use (reasoning, coding, creative, knowledge)"
)
ALL_PROMPTS_OPTION = typer.Option(False, help="Use all available prompts")
TEMPERATURES_OPTION = typer.Option(
    "0.1,0.7,1.0", help="Comma-separated temperature values to test"
)
MAX_TOKENS_OPTION = typer.Option(None, help="Maximum tokens for responses")
HTML_REPORT_OPTION = typer.Option(None, help="Generate HTML report at specified path")
RESULTS_FILE_ARGUMENT = typer.Argument(..., help="Path to evaluation results JSON file")


# ============================================================================
# Logging Configuration
# ============================================================================


def configure_logging() -> None:
    """Configure logging for the CLI application.

    Sets up INFO level logging with a formatted output including
    timestamp, logger name, level, and message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================================
# Formatting Helpers
# ============================================================================


def format_token_limit(token_limit: int) -> str:
    """Format token limit in human-readable form using binary (1024) units."""
    if token_limit < 1024:
        return str(token_limit)

    # Use binary units (1024-based) for consistency with model specs
    if token_limit < 1024 * 1024:
        return f"{token_limit // 1024}k"
    else:
        return f"{token_limit // (1024 * 1024)}M"


# ============================================================================
# Error Handling
# ============================================================================


def handle_error(error: Exception, context: str, exit_code: int = 1) -> NoReturn:
    """Handle and log an error with consistent formatting.

    Args:
        error: The exception that occurred
        context: Description of what was being attempted
        exit_code: Exit code to use on failure
    """
    logger.error(f"{context}: {error}")
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(exit_code) from error


# ============================================================================
# Prompt Loading (DRY - shared prompt loading logic)
# ============================================================================


def load_prompts_from_file(prompts_file: Path) -> list[Prompt]:
    """Load prompts from a file (JSON or plain text).

    Args:
        prompts_file: Path to prompts file

    Returns:
        List of loaded Prompt objects

    Raises:
        typer.Exit: On invalid format or file read errors
    """
    typer.echo(f"Loading prompts from file: {prompts_file}")

    if prompts_file.suffix.lower() == ".json":
        # Load structured prompts from JSON
        try:
            with open(prompts_file) as f:
                prompt_data = json.load(f)
            if isinstance(prompt_data, list):
                prompts = [Prompt(**p) for p in prompt_data]
            else:
                prompts = [Prompt(**prompt_data)]
            typer.echo(f"Loaded {len(prompts)} structured prompts from JSON")
            return prompts
        except (json.JSONDecodeError, ValueError) as e:
            handle_error(e, "Error parsing JSON prompts file")
    else:
        # Load simple text prompts (one per line)
        with open(prompts_file) as f:
            file_prompts = [line.strip() for line in f if line.strip()]
        prompts = [
            Prompt(
                id=f"custom_{i}",
                category="custom",
                messages=[Message(role="user", content=line, name=None)],
                description=None,
                expected_output=None,
            )
            for i, line in enumerate(file_prompts)
        ]
        typer.echo(f"Loaded {len(prompts)} custom prompts from text file")
        return prompts


def get_prompts_by_category_or_default(category: str | None) -> list[Prompt]:
    """Get prompts from a category with validation.

    Args:
        category: Category name, or None for default

    Returns:
        List of Prompt objects

    Raises:
        typer.Exit: If category is invalid
    """
    from hellmholtz.benchmark.prompts import PROMPTS, get_prompts_by_category

    if category is None:
        handle_error(
            ValueError("Category cannot be None"),
            "Category is required",
        )
    prompts = get_prompts_by_category(category)
    if not prompts:
        available_categories = set(prompt.category for prompt in PROMPTS)
        available_str = ", ".join(sorted(available_categories))
        handle_error(
            ValueError(f"Invalid category '{category}'"),
            f"Available categories: {available_str}",
        )
    typer.echo(f"Using {len(prompts)} prompts from category '{category}'")
    return prompts


def parse_temperatures(temperatures_str: str | None) -> list[float]:
    """Parse comma-separated temperature values.

    Args:
        temperatures_str: Comma-separated string of float values

    Returns:
        List of float temperature values

    Raises:
        typer.Exit: If parsing fails
    """
    if temperatures_str:
        try:
            return [float(t.strip()) for t in temperatures_str.split(",")]
        except ValueError as e:
            handle_error(e, "Invalid temperature values. Use comma-separated floats")
    return [0.1, 0.7, 1.0]


def parse_models(models_str: str | None) -> list[str]:
    """Parse comma-separated model names with optional Blablador fallback.

    Args:
        models_str: Comma-separated model names, or None for defaults

    Returns:
        List of model identifiers

    Raises:
        typer.Exit: If no models available
    """
    from hellmholtz.core.config import get_settings

    settings = get_settings()

    if models_str:
        return [m.strip() for m in models_str.split(",")]

    # Use all available models
    model_list = settings.default_models.copy()

    # Try to add Blablador models if available
    try:
        from hellmholtz.providers.blablador import list_models

        blablador_models = list_models()
        if blablador_models:
            # Add a few representative Blablador models
            blablador_to_add = []
            for model in blablador_models[:3]:  # Limit to first 3 to avoid too many
                identifier = model.name if model.name else model.id
                blablador_to_add.append(f"blablador:{identifier}")
            model_list.extend(blablador_to_add)
            typer.echo(f"Added {len(blablador_to_add)} Blablador models to test")
    except Exception as e:
        typer.echo(f"Could not load Blablador models: {e}")

    if not model_list:
        handle_error(ValueError("No models available."), "Model selection failed")

    return model_list


# ============================================================================
# Result File Handling (DRY - shared result management)
# ============================================================================


def generate_output_path(
    results: Any,
    format: str,
    timestamp: str | None = None,
    base_dir: str = "reports",
) -> Path:
    """Generate an output path for reports based on format and timestamp.

    Args:
        results: Results object containing timestamp info
        format: Output format (markdown, html, etc.)
        timestamp: Custom timestamp override
        base_dir: Base directory for reports

    Returns:
        Path object for output file
    """
    if not timestamp:
        timestamp = (
            results[0].timestamp.replace(":", "-").replace(".", "-") if results else "unknown"
        )

    if format.lower().startswith("html"):
        output_path = Path(f"{base_dir}/html/benchmark_report_{format.lower()}_{timestamp}.html")
    else:
        output_path = Path(f"{base_dir}/benchmark_report_{timestamp}.md")

    return output_path


def save_report_to_file(content: str, output_path: Path) -> None:
    """Save report content to file with directory creation.

    Args:
        content: Report content to write
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    typer.echo(f"Report saved to {output_path}")
