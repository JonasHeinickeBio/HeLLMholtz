from dataclasses import dataclass
import json
import logging
import re
from typing import Any, cast
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

# Cache for online-fetched token limits to avoid repeated API calls
_ONLINE_TOKEN_CACHE: dict[str, int | None] = {}

# Default token limit for unknown models
DEFAULT_TOKEN_LIMIT = 32768


@dataclass
class BaseModel:
    """Base model class with shared attributes for Blablador models.

    Attributes:
        name: Display name of the model
        alias: Short alias for the model (optional)
        description: Detailed description of model capabilities
        source: Provider source (default: 'Blablador')
    """

    name: str
    alias: str | None = None
    description: str = ""
    source: str = ""


@dataclass
class BlabladorModel(BaseModel):
    """Model configuration for Blablador models.

    Attributes:
        id: Model ID from the API
        original_api_id: Original API ID (if different from formatted ID)
        description_separator: Separator used in API ID formatting
        max_context_tokens: Maximum context window size in tokens
    """

    id: str = ""
    original_api_id: str | None = None
    description_separator: str = " - "  # Separator between name and description in API ID
    max_context_tokens: int = DEFAULT_TOKEN_LIMIT  # Default context window size

    @property
    def display_string(self) -> str:
        """Generate a human-readable display string for the model.

        Combines ID, name, alias, and description into a formatted string
        suitable for user-facing display.

        Returns:
            Formatted string like '1 - GPT-OSS-120b (alias) - description'
        """
        parts = [self.id, self.name]
        if self.alias:
            parts.append(f"({self.alias})")
        if self.description:
            parts.append(f"- {self.description}")
        return " - ".join(parts)

    @property
    def api_id(self) -> str:
        """Reconstructs the ID string expected by the API.

        Uses description_separator field to handle API formatting variations.
        Default separator is " - ", but some models (e.g., Qwen3 235) use ", ".

        For models with full formatted IDs (from API), returns ID as-is.
        For models with short IDs, formats as "ID - Name - Description"
        For models without numeric IDs (aliases, new models), uses just the name.
        """
        # Prefer the exact API identifier when available (e.g., from /models response).
        if self.original_api_id:
            return self.original_api_id

        # If ID contains spaces or commas, it's already a full formatted ID from API
        if " " in self.id or "," in self.id:
            return self.id
        # Otherwise, format it
        if self.id:  # Has short ID
            if self.description:
                return f"{self.id} - {self.name}{self.description_separator}{self.description}"
            return f"{self.id} - {self.name}"
        else:  # No numeric ID, use name directly
            return self.name


# Models known to have specific IDs and descriptions
KNOWN_MODELS: list[BlabladorModel] = [
    # Models currently returned by the Blablador API
    BlabladorModel(
        id="15",
        name="Apertus-8B-Instruct-2509",
        description="A new swiss model from September 2025",
        source="Blablador",
        max_context_tokens=32768,  # 32k context window typical for 8B models
    ),
    BlabladorModel(
        id="20",
        name="EVE-Instruct",
        description="Expert Earth Observation and Earth Science domains",
        source="Blablador",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="01",
        name="GPT-OSS-120b",
        description="Open model released by OpenAI in August 2025",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="01",
        name="MiniMax-M2.7",
        description="MiniMax best model as of April, 2026",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="02",
        name="Qwen3.5-122B-A10B-FP8",
        description="General-purpose large multimodal model",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="09",
        name="Qwen3-Coder-Next-FP8",
        description="Code model from Feb 2026",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="07",
        name="Qwen3.5-35B-A3B",
        description="Multimodal model from Feb 2026",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="999 - Mis",
        name="Mis",
        description="Internal miscellaneous model",
        source="Blablador",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="eve-instruct-4gpu",
        description="EVE instruct deployment on 4 GPUs",
        source="Blablador",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="faster-whisper-large-v3",
        description="Whisper large-v3 speech model",
        source="Blablador",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="nemotron-3-nano-omni-30b-bf16-262k-8gpu",
        description="Nemotron omni 30B model",
        source="Blablador",
        max_context_tokens=262144,
    ),
    BlabladorModel(
        id="08",
        name="Qwen3.6-35B-A3B-FP8",
        description="Multimodal model from Apr 2026",
        source="Blablador",
        max_context_tokens=131072,
    ),
    # Alias models for optimized routing
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-fast",
        alias="fast",
        description="Optimized for speed - fastest available model",
        source="Blablador",
        max_context_tokens=32768,  # Typically uses smaller models with 32k
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-large",
        alias="large",
        description="Optimized for capability - most capable available model",
        source="Blablador",
        max_context_tokens=131072,  # Large models typically have 128k
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-huge",
        alias="huge",
        description="Optimized for maximum capability - largest available model",
        source="Blablador",
        max_context_tokens=131072,  # Huge models typically have 128k+
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-code",
        alias="code",
        description="Optimized for coding tasks",
        source="Blablador",
        max_context_tokens=131072,  # Code models typically have 128k
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-apertus",
        alias="apertus",
        description="Alias for Apertus models",
        source="Blablador",
        max_context_tokens=32768,  # 32k for Apertus
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-eve",
        alias="eve",
        description="Alias for EVE-Instruct models",
        source="Blablador",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen3-8b-embeddings",
        alias="qwen3-8b-embeddings",
        description="Optimized for Qwen3 8B embeddings",
        source="Blablador",
        max_context_tokens=32000,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen35-35b-a3b",
        alias="qwen35-35b-a3b",
        description="Alias for Qwen3.5-35B-A3B",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen36-35b",
        alias="qwen36-35b",
        description="Alias for Qwen3.6-35B-A3B-FP8",
        source="Blablador",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-embeddings",
        alias="embeddings",
        description="Optimized for text embeddings",
        source="Blablador",
        max_context_tokens=8192,  # 8k typical for embeddings
    ),
    # Legacy OpenAI-compatible models
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="gpt-3.5-turbo",
        description="Legacy GPT-3.5 Turbo model",
        source="Blablador",
        max_context_tokens=16384,  # 16k context for GPT-3.5-turbo
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="text-davinci-003",
        description="Legacy text generation model",
        source="Blablador",
        max_context_tokens=4096,  # 4k context for legacy models
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="text-embedding-ada-002",
        description="Legacy text embedding model",
        source="Blablador",
        max_context_tokens=8192,  # 8k typical for embeddings
    ),
]


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
    """Build fallback Hugging Face search terms."""
    clean_name = normalized_name.replace("/", "--").replace(" ", "-").lower()
    return [normalized_name, clean_name, clean_name.replace("-", " ")]


def _fetch_hf_model_details(pattern: str) -> dict[str, Any] | None:
    """Fetch one Hugging Face model document directly by pattern."""
    try:
        url = f"https://huggingface.co/api/models/{pattern}"
        logger.debug(f"Trying to fetch model info from: {url}")

        with urllib.request.urlopen(url, timeout=5) as response:  # nosec B310
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return cast(dict[str, Any], data)

    except urllib.error.HTTPError as e:
        if e.code != 404:
            logger.debug(f"HTTP error fetching {pattern}: {e}")
    except Exception as e:
        logger.debug(f"Error fetching model info for {pattern}: {e}")

    return None


def _search_hf_model_candidates(term: str) -> dict[str, Any] | None:
    """Search Hugging Face and probe top candidate model documents."""
    try:
        quoted_term = urllib.parse.quote(term)
        search_url = f"https://huggingface.co/api/models?search={quoted_term}&limit=10"
        logger.debug(f"Searching Hugging Face models via: {search_url}")

        with urllib.request.urlopen(search_url, timeout=8) as response:  # nosec B310
            if response.status != 200:
                return None
            hits = json.loads(response.read().decode("utf-8"))

        if not isinstance(hits, list):
            return None

        candidate_ids = [
            hit.get("id")
            for hit in hits
            if isinstance(hit, dict) and isinstance(hit.get("id"), str)
        ]

        return _fetch_first_hf_candidate(candidate_ids[:5])
    except Exception as e:
        logger.debug(f"HF search fallback failed for {term}: {e}")
        return None


def _fetch_first_hf_candidate(candidate_ids: list[str]) -> dict[str, Any] | None:
    """Fetch the first valid Hugging Face candidate document."""
    for candidate_id in candidate_ids:
        detail_url = f"https://huggingface.co/api/models/{candidate_id}"
        try:
            with urllib.request.urlopen(detail_url, timeout=8) as response:  # nosec B310
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    if isinstance(data, dict):
                        return cast(dict[str, Any], data)
        except Exception as e:
            logger.debug(f"Error fetching HF candidate {candidate_id}: {e}")
    return None


def _get_model_family_context_length(model_id: str) -> int | None:
    """Get context length based on model family detection.

    Args:
        model_id: Lowercase model identifier

    Returns:
        Context length for known model families, None otherwise
    """
    # Known model families and their typical context lengths
    if (
        "llama-3.2" in model_id
        or "llama3.2" in model_id
        or "llama-3.1" in model_id
        or "llama3.1" in model_id
    ):
        return 131072
    elif "llama-3" in model_id or "llama3" in model_id:
        return 8192
    elif "mistral" in model_id:
        return 32768
    elif "qwen3" in model_id or "qwen-3" in model_id:
        return 131072
    elif "phi-4" in model_id or "phi4" in model_id:
        return 16384
    elif "phi-3" in model_id or "phi3" in model_id:
        return 4096
    elif "gpt-4" in model_id:
        return 128000
    elif "claude-3" in model_id:
        return 200000

    return None


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


def _get_online_token_limit(model_name: str, provider: str = "huggingface") -> int | None:
    """Fetch token limit from online sources for unknown models.

    Args:
        model_name: The model name to look up
        provider: The provider/source to check ("huggingface", etc.)

    Returns:
        Token limit if found, None otherwise
    """
    cache_key = f"{provider}:{model_name}"

    # Check cache first
    if cache_key in _ONLINE_TOKEN_CACHE:
        return _ONLINE_TOKEN_CACHE[cache_key]

    try:
        if provider == "huggingface":
            model_info = _fetch_huggingface_model_info(model_name)
            if model_info:
                context_length = _extract_context_length_from_hf_model(model_info)
                if context_length:
                    _ONLINE_TOKEN_CACHE[cache_key] = context_length
                    logger.debug(f"Fetched token limit for {model_name}: {context_length}")
                    return context_length

        # Could add other providers here (OpenAI API, Anthropic docs, etc.)

    except Exception as e:
        logger.debug(f"Failed to fetch online token limit for {model_name}: {e}")

    # Cache negative results too to avoid repeated failed requests
    _ONLINE_TOKEN_CACHE[cache_key] = None
    return None


def _get_provider_token_limit(provider: str, model: str) -> int:
    """Get token limit for a specific provider and model.

    Args:
        provider: The provider name (lowercase)
        model: The model name

    Returns:
        Token limit for the model
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
    # Common Ollama models and their typical context windows
    if "llama3.2" in model:
        return 131072  # Llama 3.2 has 128k context
    elif "llama3.1" in model:
        return 131072  # Llama 3.1 has 128k context
    elif "llama3" in model:
        return 8192  # Llama 3 has 8k context
    elif "mistral" in model:
        return 32768  # Mistral has 32k context
    elif "codellama" in model:
        return 16384  # Code Llama has 16k context
    elif "phi" in model:
        return 4096  # Phi models have smaller context
    else:
        return 4096  # Conservative default for unknown Ollama models


def _get_blablador_token_limit(model: str) -> int:
    """Get token limits for Blablador models."""
    blablador_model = get_model_by_name(model)
    if blablador_model:
        return blablador_model.max_context_tokens
    return 32768  # Default fallback


def get_all_provider_token_limits(include_online: bool = False) -> dict[str, dict[str, int]]:
    """Get token limits for all known models across all providers.

    Args:
        include_online: If True, include cached online-fetched models

    Returns:
        Dictionary mapping provider names to dictionaries of model_name -> token_limit
    """
    # Define all known models for each provider
    provider_models = {
        "openai": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "text-davinci-003",
            "text-embedding-ada-002",
        ],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-2.1",
            "claude-2",
        ],
        "google": ["gemini-pro", "gemini-pro-vision", "gemini-1.5-flash", "gemini-1.5-pro"],
        "ollama": [
            "llama3.2",
            "llama3.2:1b",
            "llama3.2:3b",
            "llama3.1",
            "llama3.1:8b",
            "llama3.1:70b",
            "llama3.1:405b",
            "llama3",
            "mistral",
            "codellama",
            "phi",
            "phi3",
        ],
    }

    limits: dict[str, dict[str, int]] = {"blablador": {}}

    # Build limits using provider-specific functions
    for provider, models in provider_models.items():
        limits[provider] = {}
        for model_name in models:
            limit = get_token_limit(f"{provider}:{model_name}")
            limits[provider][model_name] = limit

    # Add all Blablador models
    for model in KNOWN_MODELS:
        limits["blablador"][model.name] = model.max_context_tokens
        if model.alias:
            limits["blablador"][model.alias] = model.max_context_tokens

    # Optionally include cached online models
    if include_online:
        limits["online"] = {k: v for k, v in _ONLINE_TOKEN_CACHE.items() if v is not None}

    return limits


def clear_online_token_cache() -> None:
    """Clear the online token limit cache.

    This can be useful for testing or forcing fresh API calls.
    """
    global _ONLINE_TOKEN_CACHE
    _ONLINE_TOKEN_CACHE.clear()
    logger.info("Cleared online token limit cache")
