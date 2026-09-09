"""Round-trip validation for JSON ↔ YAML conversion.

Ensures that converting JSON→YAML→JSON (or vice versa) produces equivalent data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .convert import json_to_yaml, yaml_to_json


def validate_roundtrip(
    input_path: str | Path,
    output_format: str = "auto",
) -> dict[str, Any]:
    """Validate that a file survives JSON→YAML→JSON or YAML→JSON→YAML roundtrip.

    Args:
        input_path: Path to the input file
        output_format: Target format for first conversion ("auto", "json", "yaml")

    Returns:
        Dict with validation results:
        - success: bool
        - input_format: str
        - output_format: str
        - data_match: bool (whether round-tripped data matches original)
        - errors: list of error messages (empty if successful)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        return {
            "success": False,
            "input_format": "unknown",
            "output_format": output_format,
            "data_match": False,
            "errors": [f"File not found: {input_path}"],
        }

    # Detect input format
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        input_format = "json"
    elif suffix in (".yaml", ".yml"):
        input_format = "yaml"
    else:
        return {
            "success": False,
            "input_format": "unknown",
            "output_format": output_format,
            "data_match": False,
            "errors": [f"Unsupported file extension: {suffix}"],
        }

    # Determine target format
    if output_format == "auto":
        target_format = "yaml" if input_format == "json" else "json"
    else:
        target_format = output_format

    errors: list[str] = []

    try:
        # Read original data
        original_content = input_path.read_text(encoding="utf-8")
        if input_format == "json":
            original_data: Any = json.loads(original_content)
        else:
            original_data = yaml.safe_load(original_content)

        # Roundtrip: convert to other format, then back to original format
        if input_format == "json":
            # JSON → YAML → JSON
            intermediate = json_to_yaml(original_content)
            roundtrip_content = yaml_to_json(intermediate)
            roundtrip_data: Any = json.loads(roundtrip_content)
        else:
            # YAML → JSON → YAML
            intermediate = yaml_to_json(original_content)
            roundtrip_content = json_to_yaml(intermediate)
            roundtrip_data = yaml.safe_load(roundtrip_content)

        # Compare data structures
        data_match = _deep_compare(original_data, roundtrip_data)

        return {
            "success": True,
            "input_format": input_format,
            "output_format": target_format,
            "data_match": data_match,
            "errors": [] if data_match else ["Data structures differ after roundtrip"],
        }

    except Exception as e:
        errors.append(f"Conversion error: {e}")
        return {
            "success": False,
            "input_format": input_format,
            "output_format": target_format,
            "data_match": False,
            "errors": errors,
        }


def _compare_scalars(a: Any, b: Any) -> bool | None:  # noqa: C901
    """Compare scalar values with type coercion. Returns None if not comparable as scalars."""
    # None comparisons
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Bool comparisons
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    if isinstance(a, bool) and isinstance(b, str):
        return str(a).lower() == b.lower()
    if isinstance(a, str) and isinstance(b, bool):
        return a.lower() == str(b).lower()

    # String comparisons
    if isinstance(a, str) and isinstance(b, str):
        return a == b

    # Numeric comparisons
    if isinstance(a, int | float) and isinstance(b, int | float):
        if isinstance(a, float) or isinstance(b, float):
            return abs(a - b) < 1e-10
        return a == b
    if isinstance(a, int | float) and isinstance(b, str):
        try:
            return bool(a == float(b))
        except ValueError:
            return False
    if isinstance(a, str) and isinstance(b, int | float):
        try:
            return bool(float(a) == b)
        except ValueError:
            return False

    return None  # Not a scalar comparison


def _deep_compare(a: Any, b: Any, path: str = "") -> bool:
    """Deep compare two data structures, handling YAML type coercions."""
    scalar_result = _compare_scalars(a, b)
    if scalar_result is not None:
        return scalar_result

    # Dict comparisons
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_compare(a[k], b[k], f"{path}.{k}") for k in a)

    # List comparisons
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(
            _deep_compare(ai, bi, f"{path}[{i}]")
            for i, (ai, bi) in enumerate(zip(a, b, strict=False))
        )

    # Fallback: direct equality
    result: bool = a == b
    return result


def print_validation_report(results: dict[str, Any]) -> str:
    """Format validation results as a readable report.

    Args:
        results: Results from validate_roundtrip()

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Round-Trip Validation Report")
    lines.append("=" * 50)
    lines.append(f"Input format:  {results['input_format']}")
    lines.append(f"Output format: {results['output_format']}")
    lines.append(f"Success:       {results['success']}")
    lines.append(f"Data match:    {results['data_match']}")

    if results["errors"]:
        lines.append("")
        lines.append("Errors:")
        for err in results["errors"]:
            lines.append(f"  ✗ {err}")
    else:
        lines.append("")
        lines.append("✓ All checks passed - data is preserved across formats")

    lines.append("=" * 50)
    return "\n".join(lines)
