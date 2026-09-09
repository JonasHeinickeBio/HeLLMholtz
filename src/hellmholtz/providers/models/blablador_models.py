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
        available: Whether the model is available (online check succeeded)
    """

    id: str = ""
    original_api_id: str | None = None
    description_separator: str = " - "  # Separator between name and description in API ID
    max_context_tokens: int = DEFAULT_TOKEN_LIMIT  # Default context window size
    available: bool = True  # Whether the model is available (online check succeeded)

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
        available=False,
    ),
    BlabladorModel(
        id="09",
        name="Qwen3-Coder-Next-FP8",
        description="Code model from Feb 2026",
        source="Blablador",
        max_context_tokens=262144,  # 256k native, extensible up to 1M
        available=False,
    ),
    BlabladorModel(
        id="07",
        name="Qwen3.5-35B-A3B",
        description="Multimodal model from Feb 2026",
        source="Blablador",
        max_context_tokens=262144,  # 256k native, extensible up to 1M
        available=False,
    ),
    BlabladorModel(
        id="999 - Mis",
        name="Mis",
        description="Internal miscellaneous model",
        source="Blablador",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="eve-instruct-4gpu",
        description="EVE instruct deployment on 4 GPUs",
        source="Blablador",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="faster-whisper-large-v3",
        description="Whisper large-v3 speech model",
        source="Blablador",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="nemotron-3-nano-omni-30b-bf16-262k-8gpu",
        description="Nemotron omni 30B model",
        source="Blablador",
        max_context_tokens=262144,
        available=False,
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
        id="",
        name="alias-code",
        alias="code",
        description="Optimized for coding tasks (128k context, 32768 output)",
        source="Blablador",
        max_context_tokens=131072,
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
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-qwen35-35b-a3b",
        alias="qwen35-35b-a3b",
        description="Alias for Qwen3.5-35B-A3B",
        source="Blablador",
        max_context_tokens=131072,
        available=False,
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
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-function-call",
        alias="function-call",
        description="Optimized for function calling tasks",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for function calling
        available=False,
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
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Ministral-3-14B-Instruct-2512",
        description="Ministral 3 14B Instruct model from September 2025",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for ministral
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Qwen3 235",
        description="Qwen3 235B model",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for Qwen3
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Phi-4-multimodal-instruct",
        description="Phi-4 multimodal instruct model",
        source="Blablador",
        max_context_tokens=16384,  # 16k context for Phi
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="MiniMax-M2.1",
        description="MiniMax M2.1 model",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for MiniMax
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Qwen3-Coder-30B-A3B-Instruct",
        description="Qwen3 Coder 30B A3B Instruct model",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for Qwen3 Coder
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Devstral-Small-2-24B-Instruct-2512",
        description="Devstral Small 2 24B Instruct model from September 2025",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for Devstral
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Qwen3-Next",
        description="Qwen3 Next generation model",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for Qwen3 Next
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Qwen3-VL-32B-Instruct-FP8",
        description="Qwen3 VL 32B Instruct FP8 multimodal model",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for Qwen3 VL
        available=False,
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Tongyi-DeepResearch-30B-A3B",
        description="Tongyi DeepResearch 30B A3B model",
        source="Blablador",
        max_context_tokens=131072,  # 128k context for DeepResearch
        available=False,
    ),
    BlabladorModel(
        id="02 - Qwen3.5-122B-A10B-FP8, general purpose large model",
        name="Qwen3.5-122B-A10B-FP8, general purpose large model",
        description="Model from Blablador API",
        max_context_tokens=32768,
    ),
    BlabladorModel(
        id="09 - Qwen3-Coder-Next-FP8 from Feb 2026",
        name="Qwen3-Coder-Next-FP8 from Feb 2026",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="10 - Muse Glimmer 30b - the newest META model as of August 11, 2026",
        name="Muse Glimmer 30b",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="10 - Qwen3.5-397B-A17B",
        name="Qwen3.5-397B-A17B",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="90 - MiniMax-M3-AWQ-INT4 (strube1-booster)",
        name="MiniMax-M3-AWQ-INT4 (strube1-booster)",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="99 - GLM-5.2-AWQ-INT4",
        name="GLM-5.2-AWQ-INT4",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="DeepSeek-V4-Flash-0731",
        name="DeepSeek-V4-Flash-0731",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="alias-deepseek-v4-flash-0731",
        name="alias-deepseek-v4-flash-0731",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="alias-glm-huge",
        name="alias-glm-huge",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="alias-minimax-m3-awq-int4",
        name="alias-minimax-m3-awq-int4",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="alias-muse",
        name="alias-muse",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="alias-qwen-huge",
        name="alias-qwen-huge",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="Kimi-K3-1M",
        name="Kimi-K3-1M",
        description="Model from Blablador API",
        max_context_tokens=32768,
        available=False,
    ),
    BlabladorModel(
        id="alias-kimi-k3-1m",
        name="alias-kimi-k3-1m",
        description="Model from Blablador API",
        max_context_tokens=32768,
    ),
]
