"""Config Converter - JSON ↔ YAML translation for AI tool configurations.

Supports bidirectional conversion with round-trip validation,
and handles secrets via environment variables.
"""

from .convert import convert_file, json_to_yaml, yaml_to_json
from .secrets import (
    get_secret,
    load_dotenv_files,
    mask_secrets,
    resolve_env_refs,
)
from .validate import print_validation_report, validate_roundtrip

__all__ = [
    "convert_file",
    "json_to_yaml",
    "yaml_to_json",
    "mask_secrets",
    "resolve_env_refs",
    "load_dotenv_files",
    "get_secret",
    "validate_roundtrip",
    "print_validation_report",
]
