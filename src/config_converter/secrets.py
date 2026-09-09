"""Secrets management for config conversion.

Handles API keys and sensitive values by:
- Masking secrets in output (replacing with env var references)
- Resolving env var references when loading
- Loading secrets from .env files
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv

# Known secret field patterns (key names that typically hold secrets)
SECRET_PATTERNS = [
    r".*api[_-]?key$",
    r".*secret$",
    r".*token$",
    r".*password$",
    r".*credential$",
]

# Env var name mapping for known secrets
SECRET_ENV_MAP: dict[str, str] = {
    "apiKey": "BLABLADOR_API_KEY",
    "api_key": "BLABLADOR_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "GOOGLE_API_KEY": "GOOGLE_API_KEY",
}


def _is_secret_key(key: str) -> bool:
    """Check if a key name looks like it holds a secret."""
    key_lower = key.lower()
    return any(re.match(pattern, key_lower) for pattern in SECRET_PATTERNS)


def mask_secrets(
    data: Any,
    env_map: dict[str, str] | None = None,
    replacement: str = "",
) -> Any:
    """Replace secret values with env var references or placeholders.

    Args:
        data: Data structure (dict/list/scalar)
        env_map: Optional mapping of field names to env var names
        replacement: Value to replace secrets with (default: empty string)

    Returns:
        Data with secrets masked
    """
    if env_map is None:
        env_map = SECRET_ENV_MAP

    if isinstance(data, dict):
        masked: dict[str, Any] = {}
        for key, value in data.items():
            if _is_secret_key(key) and isinstance(value, str) and value:
                # Replace with env var reference
                env_var = env_map.get(key, key.upper())
                masked[key] = f"${{{env_var}}}"
            elif isinstance(value, dict | list):
                masked[key] = mask_secrets(value, env_map, replacement)
            else:
                masked[key] = value
        return masked

    if isinstance(data, list):
        return [mask_secrets(item, env_map, replacement) for item in data]

    return data


def resolve_env_refs(data: Any) -> Any:
    """Resolve ${VAR} references in data using environment variables.

    Args:
        data: Data structure (dict/list/scalar)

    Returns:
        Data with env var references resolved
    """
    if isinstance(data, dict):
        resolved: dict[str, Any] = {}
        for key, value in data.items():
            resolved[key] = resolve_env_refs(value)
        return resolved

    if isinstance(data, list):
        return [resolve_env_refs(item) for item in data]

    if isinstance(data, str):
        # Check for ${VAR} pattern
        match = re.match(r"^\$\{(\w+)\}$", data)
        if match:
            env_var = match.group(1)
            return os.environ.get(env_var, data)

        # Check for inline ${VAR} references
        def replace_ref(m: re.Match[str]) -> str:
            env_var = m.group(1)
            return os.environ.get(env_var, m.group(0))

        return re.sub(r"\$\{(\w+)\}", replace_ref, data)

    return data


def load_dotenv_files(
    *paths: str | Path,
    override: bool = True,
) -> None:
    """Load .env files into environment.

    Args:
        paths: Paths to .env files to load
        override: Whether to override existing env vars
    """
    for path in paths:
        p = Path(path)
        if p.exists():
            load_dotenv(p, override=override)


def get_secret(
    key: str,
    env_var: str | None = None,
    default: str | None = None,
) -> str | None:
    """Get a secret value from environment.

    Args:
        key: Field name (used to auto-detect env var name)
        env_var: Explicit env var name (overrides auto-detection)
        default: Default value if not found

    Returns:
        Secret value or default
    """
    if env_var is None:
        env_var = SECRET_ENV_MAP.get(key, key.upper())
    return os.environ.get(env_var, default)
