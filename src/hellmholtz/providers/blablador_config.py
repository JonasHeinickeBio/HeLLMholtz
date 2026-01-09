from dataclasses import dataclass


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
    id: str = ""
    original_api_id: str | None = None
    description_separator: str = " - "  # Separator between name and description in API ID

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
    BlabladorModel(
        id="0 - Ministral-3-14B-Instruct-2512 - The latest Ministral from Dec.2.2025",
        name="Ministral-3-14B-Instruct-2512",
        description="The latest Ministral from Dec.2.2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="1 - GPT-OSS-120b - an open model released by OpenAI in August 2025",
        name="GPT-OSS-120b",
        description="an open model released by OpenAI in August 2025",
        source="Blablador",
    ),
    # Note: ID conflict exists - both GPT-OSS-120b and MiniMax-M2.1 use ID starting with "1 - "
    # This may cause routing issues. MiniMax-M2.1 may not be accessible.
    BlabladorModel(
        id="1 - MiniMax-M2.1 - our best model as of December 26, 2025",
        name="MiniMax-M2.1",
        description="our best model as of December 26, 2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="15 - Apertus-8B-Instruct-2509 - A new swiss model from September 2025",
        name="Apertus-8B-Instruct-2509",
        description="A new swiss model from September 2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="2 - Qwen3 235, a great model from Alibaba with a long context size",
        name="Qwen3 235",
        description="a great model from Alibaba with a long context size",
        source="Blablador",
        description_separator=", ",  # API uses comma separator for this model
    ),
    BlabladorModel(
        id="7 - Qwen3-Coder-30B-A3B-Instruct - A code model from August 2025",
        name="Qwen3-Coder-30B-A3B-Instruct",
        description="A code model from August 2025",
        source="Blablador",
    ),
    # New December 2025 models - using exact API IDs
    BlabladorModel(
        id="999 NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        name="NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        description="NVIDIA's efficient 30B parameter model",
        source="Blablador",
    ),
    BlabladorModel(
        id="9999 option-g-2T-step-47250",
        name="option-g-2T-step-47250",
        description="Experimental model checkpoint",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Devstral-Small-2-24B-Instruct-2512",
        description="New Devstral model from December 2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Phi-4-multimodal-instruct",
        description="Multimodal model with vision capabilities",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Qwen3-Next",
        description="Latest Qwen3 model with enhanced capabilities",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Qwen3-VL-32B-Instruct-FP8",
        description="Vision-language model with 32B parameters",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="Tongyi-DeepResearch-30B-A3B",
        description="Alibaba's deep research model",
        source="Blablador",
    ),
    # Alias models for optimized routing
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-fast",
        alias="fast",
        description="Optimized for speed - fastest available model",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-large",
        alias="large",
        description="Optimized for capability - most capable available model",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-huge",
        alias="huge",
        description="Optimized for maximum capability - largest available model",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-code",
        alias="code",
        description="Optimized for coding tasks",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-apertus",
        alias="apertus",
        description="Alias for Apertus models",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-function-call",
        alias="function-call",
        description="Optimized for function calling and tool use",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="alias-embeddings",
        alias="embeddings",
        description="Optimized for text embeddings",
        source="Blablador",
    ),
    # Legacy OpenAI-compatible models
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="gpt-3.5-turbo",
        description="Legacy GPT-3.5 Turbo model",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="text-davinci-003",
        description="Legacy text generation model",
        source="Blablador",
    ),
    BlabladorModel(
        id="",  # No numeric ID, uses name directly
        name="text-embedding-ada-002",
        description="Legacy text embedding model",
        source="Blablador",
    ),
]
