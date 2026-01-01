from dataclasses import dataclass


@dataclass
class BaseModel:
    name: str
    alias: str | None = None
    description: str = ""
    source: str = ""


@dataclass
class BlabladorModel(BaseModel):
    id: str = ""
    original_api_id: str | None = None

    @property
    def display_string(self) -> str:
        parts = [self.id, self.name]
        if self.alias:
            parts.append(f"({self.alias})")
        if self.description:
            parts.append(f"- {self.description}")
        return " - ".join(parts)

    @property
    def api_id(self) -> str:
        """Reconstructs the ID string expected by the API."""
        if self.description:
            # Special case for Qwen3 235 which uses comma separator
            if self.id == "2" and "Qwen3 235" in self.name:
                return f"{self.id} - {self.name}, {self.description}"
            return f"{self.id} - {self.name} - {self.description}"
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
    ),
    BlabladorModel(
        id="7",
        name="Qwen3-Coder-30B-A3B-Instruct",
        description="A code model from August 2025",
        source="Blablador",
    ),
]
