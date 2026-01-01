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
        """
        if self.description:
            return f"{self.id} - {self.name}{self.description_separator}{self.description}"
        if self.id and self.id != self.name:
            return f"{self.id} - {self.name}"
        return self.name


# Models known to have specific IDs and descriptions
KNOWN_MODELS: list[BlabladorModel] = [
    BlabladorModel(
        id="0",
        name="Ministral-3-14B-Instruct-2512",
        description="The latest Ministral from Dec.2.2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="1",
        name="GPT-OSS-120b",
        description="an open model released by OpenAI in August 2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="1",
        name="MiniMax-M2",
        description="our best model as of December 2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="15",
        name="Apertus-8B-Instruct-2509",
        description="A new swiss model from September 2025",
        source="Blablador",
    ),
    BlabladorModel(
        id="2",
        name="Qwen3 235",
        description="a great model from Alibaba with a long context size",
        source="Blablador",
        description_separator=", ",  # API uses comma separator for this model
    ),
    BlabladorModel(
        id="7",
        name="Qwen3-Coder-30B-A3B-Instruct",
        description="A code model from August 2025",
        source="Blablador",
    ),
]
