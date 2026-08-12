"""Blablador model list - auto-generated from API.

This file contains the list of known Blablador models.
Edit this file to add new models or update existing ones.
For automatic synchronization with the API, use: hellm models sync
"""

from dataclasses import dataclass

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
        max_context_tokens=32768,  # 32k context window typical for 8B models
    ),
    BlabladorModel(
        id="20",
        name="EVE-Instruct",
        description="Expert Earth Observation and Earth Science domains",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="01",
        name="GPT-OSS-120b",
        description="Open model released by OpenAI in August 2025",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="01",
        name="MiniMax-M2.7",
        description="MiniMax best model as of April, 2026",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="02",
        name="Qwen3.5-122B-A10B-FP8",
        description="General-purpose large multimodal model",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="09",
        name="Qwen3-Coder-Next-FP8",
        description="Code model from Feb 2026",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="07",
        name="Qwen3.5-35B-A3B",
        description="Multimodal model from Feb 2026",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="999 - Mis",
        name="Mis",
        description="Internal miscellaneous model",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="eve-instruct-4gpu",
        description="EVE instruct deployment on 4 GPUs",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="faster-whisper-large-v3",
        description="Whisper large-v3 speech model",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="nemotron-3-nano-omni-30b-bf16-262k-8gpu",
        description="Nemotron omni 30B model",
        max_context_tokens=262144,
    ),
    BlabladorModel(
        id="08",
        name="Qwen3.6-35B-A3B-FP8",
        description="Multimodal model from Apr 2026",
        max_context_tokens=131072,
    ),
    # Alias models for optimized routing
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-fast",
        alias="fast",
        description="Optimized for speed - fastest available model",
        max_context_tokens=32768,  # Typically uses smaller models with 32k
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-large",
        alias="large",
        description="Optimized for capability - most capable available model",
        max_context_tokens=131072,  # Large models typically have 128k
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-huge",
        alias="huge",
        description="Optimized for maximum capability - largest available model",
        max_context_tokens=131072,  # Huge models typically have 128k+
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-code",
        alias="code",
        description="Optimized for coding tasks",
        max_context_tokens=131072,  # Code models typically have 128k
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-apertus",
        alias="apertus",
        description="Alias for Apertus models",
        max_context_tokens=32768,  # 32k for Apertus
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-eve",
        alias="eve",
        description="Alias for EVE-Instruct models",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen3-8b-embeddings",
        alias="qwen3-8b-embeddings",
        description="Optimized for Qwen3 8B embeddings",
        max_context_tokens=32000,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen35-35b-a3b",
        alias="qwen35-35b-a3b",
        description="Alias for Qwen3.5-35B-A3B",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen36-35b",
        alias="qwen36-35b",
        description="Alias for Qwen3.6-35B-A3B-FP8",
        max_context_tokens=131072,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-embeddings",
        alias="embeddings",
        description="Optimized for text embeddings",
        max_context_tokens=8192,  # 8k typical for embeddings
    ),
    # Legacy OpenAI-compatible models
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="gpt-3.5-turbo",
        description="Legacy GPT-3.5 Turbo model",
        max_context_tokens=16384,  # 16k context for GPT-3.5-turbo
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="text-davinci-003",
        description="Legacy text generation model",
        max_context_tokens=4096,  # 4k context for legacy models
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="text-embedding-ada-002",
        description="Legacy text embedding model",
        max_context_tokens=8192,  # 8k typical for embeddings
    ),
]
