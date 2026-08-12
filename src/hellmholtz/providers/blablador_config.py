"""Blablador configuration - contains code logic for model management.

This file contains the configuration management code including:
- Model management functions (sync, available, check)
- Token limit utilities
- API integration functions

Model definitions are in `blablador_models.py`.
"""

import logging
import os
from pathlib import Path
import re
from typing import Any

from .models.blablador_models import (
    DEFAULT_TOKEN_LIMIT,
    KNOWN_MODELS,
    BlabladorModel,
)

# Re-export model classes for convenience
__all__ = [
    "sync_models",
    "get_available_models",
    "get_token_limit",
    "get_model_by_name",
    "get_model_display_string",
    "get_model_api_id",
    "get_models_dict",
    "clear_online_token_cache",
    # Re-export for convenience
    "BlabladorModel",
    "KNOWN_MODELS",
    "DEFAULT_TOKEN_LIMIT",
]

logger = logging.getLogger(__name__)

# Cache for online-fetched token limits to avoid repeated API calls
_ONLINE_TOKEN_CACHE: dict[str, int | None] = {}


def _extract_model_name_from_api_id(api_id: str) -> str:
    """Extract model name from API ID format.

    API IDs are formatted as "ID - Name - Description" or "ID - Name".
    This function extracts the name part.

    Args:
        api_id: The full API ID string

    Returns:
        The model name extracted from the ID
    """
    # Split by " - " and get the name part (second element)
    parts = api_id.split(" - ")
    if len(parts) >= 2:
        # Return the name part (between first " - " and optional " - description")
        return parts[1].strip()
    return api_id.strip()


def _extract_short_id(api_id: str) -> str | None:
    """Extract short ID from API ID format.

    API IDs start with a short ID like "01 - GPT-OSS-120b...".
    This function extracts just the "01" part.

    Args:
        api_id: The full API ID string

    Returns:
        The short ID (e.g., "01") or None if not found
    """
    # Split by " - " and get the first part
    parts = api_id.split(" - ")
    if parts:
        return parts[0].strip()
    return None


def sync_models(
    api_key: str | None = None,
    api_base: str | None = None,
    auto_update: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Synchronize models between API and local configuration.

    This function:
    1. Fetches current models from the API
    2. Compares with local configuration
    3. Optionally updates the configuration file

    Args:
        api_key: Blablador API key. If None, uses environment variable.
        api_base: API base URL. If None, uses environment variable.
        auto_update: If True, automatically update the configuration file.
        dry_run: If True, only show what would change without making changes.

    Returns:
        Dictionary with sync results including:
        - api_models: List of model names from API
        - config_models: List of model names in config
        - new_models: List of models in API but not in config
        - removed_models: List of models in config but not in API
        - unchanged_models: List of models in both
        - actions: List of actions that would be taken
    """
    import requests

    # Initialize result
    result: dict[str, Any] = {
        "api_models": [],
        "config_models": [],
        "new_models": [],
        "removed_models": [],
        "unchanged_models": [],
        "actions": [],
        "summary": {},
    }

    # Get API models
    api_base_url = api_base or os.getenv(
        "BLABLADOR_API_BASE", "https://api.helmholtz-blablador.fz-juelich.de/v1"
    )
    api_key = api_key or os.getenv("BLABLADOR_API_KEY", "")

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.get(f"{api_base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        api_models_data = data.get("data", [])
        result["api_models"] = [m.get("id", "") for m in api_models_data if isinstance(m, dict)]
        result["api_model_details"] = api_models_data

    except Exception as e:
        logger.error(f"Failed to fetch API models: {e}")
        result["error"] = f"Failed to fetch API models: {e}"
        return result

    # Get configured models from KNOWN_MODELS
    result["config_models"] = [model.name for model in KNOWN_MODELS]
    result["config_ids"] = [model.id for model in KNOWN_MODELS if model.id]

    # Compare models - need to handle formatted API IDs vs config model names
    # Parse API model IDs to extract short ID and name
    parsed_api_models: list[tuple[str, str, str]] = []  # (short_id, name, full_api_id)
    for api_id in result["api_models"]:
        short_id = _extract_short_id(api_id)
        name = _extract_model_name_from_api_id(api_id)
        parsed_api_models.append((short_id or "", name, api_id))

    # Create lookup sets
    api_names = {name for _, name, _ in parsed_api_models}
    api_short_ids = {short_id for short_id, _, _ in parsed_api_models}

    config_names_set = set(result["config_models"])
    config_ids_set = set(result["config_ids"])

    # Determine which models are new, removed, or unchanged
    new_api_models = []
    for short_id, name, full_api_id in parsed_api_models:
        # A model is "new" if its name is not in config names AND its short ID is not in config IDs
        if name not in config_names_set and short_id not in config_ids_set:
            new_api_models.append(full_api_id)

    removed_models = []
    for config_name in result["config_models"]:
        # A model is "removed" if its name is not in API names AND its ID is not in API short IDs
        if config_name not in api_names and config_name not in api_short_ids:
            removed_models.append(config_name)

    unchanged_models = []
    for short_id, name, full_api_id in parsed_api_models:
        # A model is "unchanged" if its name is in config names OR its short ID is in config IDs
        if name in config_names_set or short_id in config_ids_set:
            unchanged_models.append(full_api_id)

    result["new_models"] = sorted(new_api_models)
    result["removed_models"] = sorted(removed_models)
    result["unchanged_models"] = sorted(unchanged_models)

    # Generate actions
    actions = []

    if new_api_models:
        actions.append(f"  - Add {len(new_api_models)} new models from API")
        for model_id in sorted(new_api_models):
            actions.append(f"    - {model_id}")

    if removed_models:
        actions.append(f"  - Remove {len(removed_models)} models no longer in API")
        for model_name in sorted(removed_models):
            actions.append(f"    - {model_name}")

    if unchanged_models:
        actions.append(f"  - Keep {len(unchanged_models)} unchanged models")

    result["actions"] = actions

    # Summary
    result["summary"] = {
        "api_models_count": len(result["api_models"]),
        "config_models_count": len(result["config_models"]),
        "new_count": len(new_api_models),
        "removed_count": len(removed_models),
        "unchanged_count": len(unchanged_models),
        "sync_status": "up-to-date"
        if not new_api_models and not removed_models
        else "needs-update",
    }

    # Perform update if requested and not dry-run
    if auto_update and not dry_run:
        update_result = _update_config_file(Path(__file__).resolve(), api_models_data)
        result["update_result"] = update_result

    # Store dry-run result
    if dry_run:
        result["dry_run"] = True
        result["would_update"] = len(new_api_models) > 0 or len(removed_models) > 0

    return result


def _update_config_file(
    config_path: Path, api_models_data: list[dict[str, Any]]
) -> dict[str, Any]:
    """Update the configuration file with new models.

    This function:
    1. Creates a backup of the current config
    2. Updates KNOWN_MODELS with new API models
    3. Removes models that are no longer available
    4. Writes the updated config

    Args:
        config_path: Path to blablador_config.py
        api_models_data: List of model data from API

    Returns:
        Dictionary with update results
    """
    try:
        # Create backup
        backup_path = _backup_config(config_path)

        # Build new model configurations from API
        new_models_config: list[str] = []

        for model_data in api_models_data:
            model_id = model_data.get("id", "")
            if not model_id:
                continue

            # Extract model name from API ID
            # Format: "ID - Name - Description" or "ID - Name"
            model_name = _extract_model_name_from_api_id(model_id)

            # Check if model already exists in KNOWN_MODELS
            # Check by: name, short ID, or alias
            existing_names = {m.name for m in KNOWN_MODELS}
            existing_ids = {m.id for m in KNOWN_MODELS if m.id}

            # If this model ID is already known (matches by name, ID, or alias), skip it
            if model_name in existing_names or model_id in existing_ids:
                continue

            # Check if the short ID part exists in known IDs
            short_id = _extract_short_id(model_id)
            if short_id and short_id in existing_ids:
                continue

            # Extract model info from API response
            description = model_data.get("description", "")

            # Determine context length
            context_length = model_data.get("context_length")
            default_context = DEFAULT_TOKEN_LIMIT
            if context_length and isinstance(context_length, int):
                default_context = context_length

            # Generate a reasonable description if not provided
            if not description:
                description = "Model from Blablador API"

            # Create model entry
            model_entry = f"""    BlabladorModel(
        id="{model_id}",
        name="{model_name}",
        description="{description}",
        source="vllm",
        max_context_tokens={default_context},
    ),"""
            new_models_config.append(model_entry)

        # Read current file content
        content = config_path.read_text()

        # Find the KNOWN_MODELS list and update it
        # Pattern to find the start of KNOWN_MODELS list
        pattern = r"(KNOWN_MODELS:\s*list\[BlabladorModel\]\s*=\s*\[)(.*?)(\])"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            prefix = match.group(1)
            middle_content = match.group(2)

            # Keep existing models and add new ones
            # Find the position after the last existing model entry
            # For simplicity, we'll replace everything between [ and ]
            new_list_content = ""

            # Add existing models (without the last blank lines if any)
            lines = middle_content.strip().split("\n")
            # Keep only model definitions, not the trailing blank lines
            model_lines = [
                line for line in lines if "BlabladorModel(" in line or line.strip() == ""
            ]

            if model_lines:
                new_list_content = "\n".join(model_lines) + "\n"

            if new_models_config:
                new_list_content += "\n".join(new_models_config) + "\n    "

            # Reconstruct the file
            new_content = (
                content[: match.start()] + prefix + new_list_content + "]" + content[match.end() :]
            )

            # Write updated content
            config_path.write_text(new_content)

            return {
                "success": True,
                "backup_path": str(backup_path) if backup_path else None,
                "models_added": len(new_models_config),
            }
        else:
            return {
                "success": False,
                "error": "Could not find KNOWN_MODELS list in config file",
            }

    except Exception as e:
        logger.error(f"Failed to update config file: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def get_available_models(
    api_key: str | None = None,
    api_base: str | None = None,
) -> list[str]:
    """Get list of available model IDs from the API.

    Args:
        api_key: Blablador API key. If None, uses environment variable.
        api_base: API base URL. If None, uses environment variable.

    Returns:
        List of available model IDs
    """
    import requests

    api_base_url = api_base or os.getenv(
        "BLABLADOR_API_BASE", "https://api.helmholtz-blablador.fz-juelich.de/v1"
    )
    api_key = api_key or os.getenv("BLABLADOR_API_KEY", "")

    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.get(f"{api_base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        api_models_data = data.get("data", [])

        return [m.get("id", "") for m in api_models_data if isinstance(m, dict)]

    except Exception as e:
        logger.error(f"Failed to fetch API models: {e}")
        return []


# ============================================================================
# Configuration File Path Utilities
# ============================================================================


def _get_backup_path(config_path: Path) -> Path:
    """Get backup file path for configuration."""
    return config_path.parent / f".{config_path.name}.backup"


def _backup_config(config_path: Path) -> Path | None:
    """Create a backup of the current configuration file.

    Args:
        config_path: Path to the blablador_config.py file

    Returns:
        Path to backup file if successful, None otherwise
    """
    try:
        import shutil

        backup_path = _get_backup_path(config_path)
        if config_path.exists():
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup of config at: {backup_path}")
            return backup_path
    except Exception as e:
        logger.warning(f"Failed to create config backup: {e}")
    return None


def get_models_dict() -> dict[str, BlabladorModel]:
    """Get all known models as a dictionary keyed by name.

    Returns:
        Dictionary mapping model names to BlabladorModel instances
    """
    return {model.name: model for model in KNOWN_MODELS}


def get_model_by_name(model_name: str) -> BlabladorModel | None:
    """Get a BlabladorModel by name, ID, or alias.

    Args:
        model_name: The name, ID, or alias of the model

    Returns:
        The matching BlabladorModel or None if not found
    """
    for model in KNOWN_MODELS:
        if model.name == model_name or model.id == model_name or model.alias == model_name:
            return model
    return None


def get_model_display_string(model_name: str) -> str | None:
    """Get the display string for a model.

    Args:
        model_name: The name, ID, or alias of the model

    Returns:
        The display string or None if model not found
    """
    model = get_model_by_name(model_name)
    if model:
        return model.display_string
    return None


def get_model_api_id(model_name: str) -> str | None:
    """Get the API ID for a model.

    Args:
        model_name: The name, ID, or alias of the model

    Returns:
        The API ID or None if model not found
    """
    model = get_model_by_name(model_name)
    if model:
        return model.api_id
    return None


# ============================================================================
# Token Limit Functions
# ============================================================================


def _extract_existing_models_from_file(config_path: Path) -> dict[str, BlabladorModel]:
    """Extract models from existing config file by parsing it.

    This parses the file to get existing model configurations.

    Args:
        config_path: Path to blablador_config.py

    Returns:
        Dictionary mapping model names to BlabladorModel instances
    """
    models_by_name: dict[str, BlabladorModel] = {}

    try:
        content = config_path.read_text()

        # Parse known models from the file
        # Look for patterns like: BlabladorModel(name="...", ...)
        import re

        # Find all BlabladorModel declarations
        pattern = r"BlabladorModel\(([^)]+)\)"
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            # Extract name field
            name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', match)
            if name_match:
                name = name_match.group(1)

                # Extract other fields if present
                model = BlabladorModel(name=name)

                # Try to extract ID
                id_match = re.search(r'id\s*=\s*["\']([^"\']*)["\']', match)
                if id_match:
                    model.id = id_match.group(1)

                # Try to extract alias
                alias_match = re.search(r'alias\s*=\s*["\']([^"\']*)["\']', match)
                if alias_match:
                    model.alias = alias_match.group(1)

                # Try to extract description
                desc_match = re.search(r'description\s*=\s*["\']([^"\']*)["\']', match)
                if desc_match:
                    model.description = desc_match.group(1)

                # Try to extract max_context_tokens
                tokens_match = re.search(r"max_context_tokens\s*=\s*(\d+)", match)
                if tokens_match:
                    model.max_context_tokens = int(tokens_match.group(1))

                # Try to extract source
                source_match = re.search(r'source\s*=\s*["\']([^"\']*)["\']', match)
                if source_match:
                    model.source = source_match.group(1)

                models_by_name[name] = model

    except Exception as e:
        logger.warning(f"Failed to parse existing config file: {e}")

    return models_by_name


def _get_blablador_token_limit(model: str) -> int:
    """Get token limit for a Blablador model.

    Args:
        model: Model name or ID

    Returns:
        Token limit in tokens, or DEFAULT_TOKEN_LIMIT if not found
    """
    model_lower = model.lower()

    # Check KNOWN_MODELS for exact match
    for m in KNOWN_MODELS:
        if (
            m.name.lower() == model_lower
            or m.id == model
            or (m.alias and m.alias.lower() == model_lower)
        ):
            return m.max_context_tokens

    # Try to extract from model name patterns
    if "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower or "gpt-4-1106" in model_lower:
        return 128000
    elif "gpt-4" in model_lower:
        return 8192
    elif "gpt-3.5-turbo" in model_lower:
        return 16384

    # Check for context length patterns in model name
    match = re.search(r"(\d+)k?context", model_lower)
    if match:
        return int(match.group(1)) * 1024

    # Check for number patterns like -16k-, -32k-, -128k-
    match = re.search(r"(\d+)(?:k|k?)context|(\d+)k?(?:-context|ctx)", model_lower)
    if match:
        value = match.group(1) or match.group(2)
        return int(value) * 1024

    return DEFAULT_TOKEN_LIMIT


def _get_openai_token_limit(model: str) -> int:
    """Get token limits for OpenAI models."""
    model = model.lower()
    if "gpt-4o" in model:
        return 128000  # GPT-4o has 128k context
    elif "gpt-4-turbo" in model:
        return 128000  # GPT-4 Turbo has 128k context
    elif "gpt-4" in model:
        return 8192  # GPT-4 has 8k context
    elif "gpt-3.5-turbo" in model:
        return 16384  # GPT-3.5 Turbo has 16k context
    elif "text-davinci-003" in model:
        return 4096  # Legacy model
    elif "text-embedding-ada-002" in model:
        return 8192  # Embedding model
    else:
        return 4096  # Conservative default for unknown OpenAI models


def _get_anthropic_token_limit(model: str) -> int:
    """Get token limits for Anthropic models."""
    model = model.lower()
    if "claude-3-opus" in model:
        return 200000  # Claude 3 Opus has 200k context
    elif "claude-3-sonnet" in model:
        return 200000  # Claude 3 Sonnet has 200k context
    elif "claude-3-haiku" in model:
        return 200000  # Claude 3 Haiku has 200k context
    elif "claude-3" in model:
        return 200000  # Claude 3 family has 200k context
    elif "claude-2" in model:
        return 100000  # Claude 2 has 100k context
    else:
        return 100000  # Conservative default for unknown Anthropic models


def _get_google_token_limit(model: str) -> int:
    """Get token limits for Google models."""
    model = model.lower()
    if "gemini-pro" in model:
        return 1000000  # Gemini Pro has 1M context (theoretical)
    elif "gemini-flash" in model:
        return 1000000  # Gemini Flash has 1M context
    elif "gemini" in model:
        return 1000000  # Gemini family has large context
    else:
        return 32768  # Conservative default for unknown Google models


def _get_ollama_token_limit(model: str) -> int:
    """Get token limits for Ollama models."""
    model = model.lower()
    # Common Ollama models with known context sizes
    context_patterns = {
        "llama3": 8192,
        "llama2": 4096,
        "mistral": 8192,
        "mixtral": 32768,
        "codellama": 16384,
        "phi": 2048,
        "neural-chat": 8192,
        "starling": 8192,
        "dolphin": 16384,
        "yi": 4096,
        "qwen": 8192,
    }

    for pattern, context in context_patterns.items():
        if pattern in model:
            return context

    return 4096  # Conservative default for unknown Ollama models


def _get_provider_token_limit(provider: str, model: str) -> int:
    """Get token limit for a model from a specific provider.

    Args:
        provider: Provider name (blablador, openai, anthropic, google, ollama)
        model: Model name or ID

    Returns:
        Token limit in tokens
    """
    if provider == "openai":
        return _get_openai_token_limit(model)
    elif provider == "anthropic":
        return _get_anthropic_token_limit(model)
    elif provider == "google":
        return _get_google_token_limit(model)
    elif provider == "ollama":
        return _get_ollama_token_limit(model)
    elif provider == "blablador":
        limit = _get_blablador_token_limit(model)
        # If not found in static config, try online fetching for Blablador models
        if limit == DEFAULT_TOKEN_LIMIT:  # Default fallback value
            online_limit = _get_online_token_limit(model, "huggingface")
            if online_limit:
                limit = online_limit
        return limit
    else:
        # Unknown provider, try blablador as fallback
        limit = _get_blablador_token_limit(model)
        # If not found, try online fetching
        if limit == DEFAULT_TOKEN_LIMIT:  # Default fallback value
            online_limit = _get_online_token_limit(model, "huggingface")
            if online_limit:
                limit = online_limit
        return limit


def get_token_limit(model_name: str) -> int:
    """Get the maximum context token limit for a model.

    Supports all providers: Blablador, OpenAI, Anthropic, Google, Ollama.
    Falls back to reasonable defaults if model is unknown.

    Args:
        model_name: The name, ID, or alias of the model (with or without provider prefix)

    Returns:
        The maximum context tokens for the model
    """
    # Extract provider and model name if prefixed
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        provider = provider.lower()
    else:
        provider = "blablador"  # Default to blablador if no provider specified
        model = model_name

    return _get_provider_token_limit(provider, model)


def _get_online_token_limit(model: str, source: str = "huggingface") -> int | None:
    """Get token limit from online source (Hugging Face API).

    Args:
        model: Model name or ID
        source: Source to query (huggingface)

    Returns:
        Token limit if found, None otherwise
    """
    # Check cache first
    if model in _ONLINE_TOKEN_CACHE:
        cached = _ONLINE_TOKEN_CACHE[model]
        if cached is not None and not isinstance(cached, dict):
            return cached  # Return cached token limit directly
        # If cached value is a dict, extract token limit from it

    if source == "huggingface":
        # Try to fetch from Hugging Face
        model_info = _fetch_huggingface_model_info(model)
        if model_info:
            token_limit = _extract_context_length_from_hf_model(model_info)
            _ONLINE_TOKEN_CACHE[model] = token_limit
            return token_limit

    _ONLINE_TOKEN_CACHE[model] = None
    return None


def _fetch_huggingface_model_info(model_name: str) -> dict[str, Any] | None:
    """Fetch model information from Hugging Face API.

    Args:
        model_name: The model name to search for

    Returns:
        Model info dict if found, None otherwise
    """

    normalized_name = model_name.strip()
    for pattern in _build_hf_search_patterns(normalized_name):
        model_info = _fetch_hf_model_details(pattern)
        if model_info is not None:
            return model_info

    for term in _build_hf_search_terms(normalized_name):
        model_info = _search_hf_model_candidates(term)
        if model_info is not None:
            return model_info

    return None


def _build_hf_search_patterns(normalized_name: str) -> list[str]:
    """Build direct Hugging Face model lookup patterns."""
    clean_name = normalized_name.replace("/", "--").replace(" ", "-").lower()
    return [
        normalized_name,
        clean_name,
        normalized_name.lower(),
        f"microsoft/{normalized_name}",
        f"meta-llama/{normalized_name}",
        f"mistralai/{normalized_name}",
        f"Qwen/{normalized_name}",
    ]


def _build_hf_search_terms(normalized_name: str) -> list[str]:
    """Build search terms for Hugging Face model search."""
    # Extract base model name (remove provider prefixes, version numbers, etc.)
    clean = normalized_name.lower()

    # Remove common prefixes
    prefixes = ["hf-", "model-", "-"]
    for prefix in prefixes:
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]

    # Split on common separators and take first part
    for sep in ["-", ".", "_", "/"]:
        if sep in clean:
            clean = clean.split(sep)[0]
            break

    # Remove numbers at end
    clean = re.sub(r"\d+$", "", clean)

    return [clean] if clean else []


def _fetch_hf_model_details(model_name: str) -> dict[str, Any] | None:
    """Fetch details for a specific Hugging Face model.

    Args:
        model_name: Full model name (e.g., 'meta-llama/Llama-2-7b')

    Returns:
        Model info dict or None
    """
    import requests

    api_url = f"https://huggingface.co/api/models/{model_name}"

    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data: dict[str, Any] = response.json()
            return data
    except Exception as e:
        logger.debug(f"Failed to fetch model {model_name} from HF: {e}")

    return None


def _search_hf_model_candidates(term: str) -> dict[str, Any] | None:
    """Search Hugging Face for models matching a term.

    Args:
        term: Search term

    Returns:
        First matching model info or None
    """
    import requests

    search_url = "https://huggingface.co/api/models"
    params = {
        "search": term,
        "limit": 5,
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            models: list[dict[str, Any]] = response.json()
            if models:
                return models[0]
    except Exception as e:
        logger.debug(f"Failed to search HF for term '{term}': {e}")

    return None


def _get_model_family_context_length(model_id: str) -> int:
    """Get context length based on model family patterns.

    Args:
        model_id: Model identifier

    Returns:
        Context length in tokens
    """
    model_lower = model_id.lower()

    # Common model families and their context lengths
    families = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-1106": 128000,
        "gpt-4-32k": 32768,
        "gpt-4": 8192,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo": 16384,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2.1": 100000,
        "claude-2": 100000,
        "claude-1": 100000,
        "gemini-pro": 1000000,
        "gemini-ultra": 1000000,
        "llama-3": 8192,
        "llama-2": 4096,
        "llama-1": 2048,
        "mistral-large": 32768,
        "mistral-medium": 8192,
        "mistral-small": 8192,
        "mixtral": 32768,
        "codellama": 16384,
        "phi-3": 128000,
        "phi-2": 2048,
        "qwen-2.5": 32768,
        "qwen-2": 32768,
        "qwen-1.5": 32768,
        "qwen-1": 8192,
    }

    for pattern, length in families.items():
        if pattern in model_lower:
            return length

    # Default based on model size hints
    if "70b" in model_lower or "65b" in model_lower or "7b" in model_lower:
        return 4096
    elif "13b" in model_lower or "34b" in model_lower:
        return 8192
    elif "70b" in model_lower:
        return 16384

    return 4096


def _extract_context_length_from_hf_model(model_info: dict[str, Any]) -> int | None:
    """Extract context length from Hugging Face model information.

    Args:
        model_info: Model info dict from HF API

    Returns:
        Context length in tokens, or None if not found
    """
    # Check config.json for context length
    config = model_info.get("config", {})

    # Common config keys for context length
    context_keys = [
        "max_position_embeddings",
        "max_seq_len",
        "max_seq_length",
        "seq_length",
        "context_length",
        "n_positions",
        "model_max_length",
    ]

    for key in context_keys:
        if key in config and isinstance(config[key], int):
            length = int(config[key])
            if 1000 <= length <= 2000000:  # Reasonable token range
                return length

    # Check model card content for context information
    card_data = model_info.get("cardData", {})
    if "max_position_embeddings" in card_data:
        return int(card_data["max_position_embeddings"])

    # Try to extract from model description/tags
    tags = model_info.get("tags", [])
    for tag in tags:
        # Look for context length in tags like "context-length-131072"
        match = re.search(r"context[_-]?length[_-]?(\d+)", tag, re.IGNORECASE)
        if match:
            length = int(match.group(1))
            if 1000 <= length <= 2000000:
                return length

    # Fallback based on model family
    model_id = model_info.get("id", "").lower()
    return _get_model_family_context_length(model_id)


def clear_online_token_cache() -> None:
    """Clear the online token limit cache.

    This can be useful for testing or forcing fresh API calls.
    """
    global _ONLINE_TOKEN_CACHE
    _ONLINE_TOKEN_CACHE.clear()
    logger.info("Cleared online token limit cache")
