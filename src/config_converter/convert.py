"""JSON ↔ YAML conversion for AI tool configuration files.

Uses PyYAML with safe_dump/safe_load to avoid code execution risks.
Preserves key order and uses block style for human readability.
Supports masking secrets and resolving env var references.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .secrets import load_dotenv_files, mask_secrets, resolve_env_refs


def _is_file_path(input_str: str | Path) -> bool:
    """Check if input is a path to an existing file."""
    if isinstance(input_str, Path):
        return True
    if isinstance(input_str, str):
        return Path(input_str).is_file()
    return False


def json_to_yaml(
    json_input: str | Path,
    output_path: str | Path | None = None,
    indent: int = 2,
    sort_keys: bool = False,
    mask_api_keys: bool = False,
    env_files: list[str | Path] | None = None,
) -> str:
    """Convert JSON file or string to YAML.

    Args:
        json_input: JSON file path or JSON string
        output_path: Optional path to write YAML output
        indent: YAML indentation level (default: 2)
        sort_keys: Whether to sort keys alphabetically (default: False)
        mask_api_keys: Replace API key values with env var references
        env_files: Optional list of .env files to load for secret resolution

    Returns:
        YAML string
    """
    # Load .env files if provided
    if env_files:
        load_dotenv_files(*env_files)

    # Load JSON
    if _is_file_path(json_input):
        json_path = Path(json_input)
        data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        data = json.loads(json_input)

    # Mask secrets if requested
    if mask_api_keys:
        data = mask_secrets(data)

    # Convert to YAML
    yaml_str: str = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=sort_keys,
        allow_unicode=True,
        width=float("inf"),  # Prevent line folding
        indent=indent,
    )

    # Write output if path provided
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml_str, encoding="utf-8")

    return yaml_str


def yaml_to_json(
    yaml_input: str | Path,
    output_path: str | None = None,
    indent: int = 2,
    resolve_env: bool = False,
    env_files: list[str | Path] | None = None,
) -> str:
    """Convert YAML file or string to JSON.

    Args:
        yaml_input: YAML file path or YAML string
        output_path: Optional path to write JSON output
        indent: JSON indentation level (default: 2)
        resolve_env: Resolve ${VAR} references from environment
        env_files: Optional list of .env files to load for resolution

    Returns:
        JSON string
    """
    # Load .env files if provided
    if env_files:
        load_dotenv_files(*env_files)

    # Load YAML
    if _is_file_path(yaml_input):
        yaml_path = Path(yaml_input)
        data: Any = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(yaml_input)

    # Resolve env var references if requested
    if resolve_env:
        data = resolve_env_refs(data)

    # Convert to JSON
    json_str: str = json.dumps(data, indent=indent, ensure_ascii=False)

    # Write output if path provided
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_str, encoding="utf-8")

    return json_str


def convert_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    mask_api_keys: bool = False,
    resolve_env: bool = False,
    env_files: list[str | Path] | None = None,
) -> Path:
    """Auto-detect format and convert between JSON and YAML.

    Args:
        input_path: Input file path (.json or .yaml/.yml)
        output_path: Output file path (auto-detected from input if None)
        indent: Indentation level (default: 2)
        sort_keys: Whether to sort keys (default: False)
        mask_api_keys: Replace API key values with env var references (JSON→YAML)
        resolve_env: Resolve ${VAR} references (YAML→JSON)
        env_files: Optional list of .env files to load

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load .env files if provided
    if env_files:
        load_dotenv_files(*env_files)

    # Detect format from extension
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        # JSON → YAML
        output = output_path or input_path.with_suffix(".yaml")
        yaml_str = json_to_yaml(
            input_path,
            indent=indent,
            sort_keys=sort_keys,
            mask_api_keys=mask_api_keys,
        )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(yaml_str, encoding="utf-8")
    elif suffix in (".yaml", ".yml"):
        # YAML → JSON
        output = output_path or input_path.with_suffix(".json")
        json_str = yaml_to_json(
            input_path,
            indent=indent,
            resolve_env=resolve_env,
        )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json_str, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .json, .yaml, or .yml")

    return Path(output)
