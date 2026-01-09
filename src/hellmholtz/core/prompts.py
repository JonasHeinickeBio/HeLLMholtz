"""
Core prompt structures for LLM interactions.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single message in a conversation, compatible with OpenAI format."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(min_length=1, description="Message content")
    name: str | None = Field(None, description="Optional name for the message")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed_roles = {"system", "user", "assistant"}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}, got '{v}'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None  # Convert empty strings to None
        return v


class Prompt(BaseModel):
    """A structured prompt for LLM benchmarking."""

    id: str = Field(..., description="Unique prompt identifier")
    category: str = Field(..., description="Prompt category (reasoning, coding, etc.)")
    messages: list[Message] = Field(min_length=1, description="Conversation messages")
    description: str | None = Field(None, description="Optional prompt description")
    expected_output: str | None = Field(None, description="Expected output format/type")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if not any(msg.role == "user" for msg in v):
            raise ValueError("Prompt must contain at least one user message")
        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt ID cannot be empty")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Category cannot be empty")
        return v.strip()

    @property
    def user_message(self) -> str:
        """Get the primary user message content.

        Returns the first user message in the conversation.
        Useful for backward compatibility with systems expecting
        a single user prompt string.

        Returns:
            Content of the first user message, or empty string if none exists.
        """
        user_msgs = [msg for msg in self.messages if msg.role == "user"]
        return user_msgs[0].content if user_msgs else ""

    @property
    def system_message(self) -> str | None:
        """Get the system message content if present.

        System messages provide instructions or context to the model
        about how to behave or what role to play.

        Returns:
            Content of the first system message, or None if no system message exists.
        """
        system_msgs = [msg for msg in self.messages if msg.role == "system"]
        return system_msgs[0].content if system_msgs else None

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Export messages in OpenAI API format."""
        return [msg.model_dump(exclude_none=True) for msg in self.messages]

    def to_dict(self) -> dict[str, Any]:
        """Export prompt as dictionary."""
        return dict(self.model_dump())

    def to_json(self, indent: int = 2) -> str:
        """Export prompt as JSON string."""
        return str(self.model_dump_json(indent=indent))

    def to_yaml(self) -> str:
        """Export prompt as YAML string."""
        try:
            import yaml

            return str(yaml.dump(self.model_dump(), default_flow_style=False))
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML export. Install with: pip install PyYAML"
            ) from None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prompt":
        """Create prompt from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Prompt":
        """Create prompt from JSON string."""
        import json

        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Prompt":
        """Create prompt from YAML string."""
        try:
            import yaml

            data = yaml.safe_load(yaml_str)
            return cls(**data)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML import. Install with: pip install PyYAML"
            ) from None


def load_prompts(
    file_path: str | Path, category: str | None = None, file_format: str | None = None
) -> list["Prompt"]:
    """Load prompts from a file.

    Supports both JSON and text formats. For JSON files, expects a list of
    prompt objects or a single prompt object. For text files, treats each
    non-empty line as a separate prompt.

    Args:
        file_path: Path to the prompts file
        category: Optional category filter (only applies to JSON format)
        file_format: Optional file format override ('json' or 'txt').
                    If not provided, inferred from file extension.

    Returns:
        List of loaded Prompt objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid or unsupported
        json.JSONDecodeError: If JSON parsing fails
    """
    import json

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {file_path}")

    # Determine file format
    if file_format is None:
        if path.suffix.lower() == ".json":
            file_format = "json"
        elif path.suffix.lower() in [".txt", ".md"]:
            file_format = "txt"
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .json, .txt, or .md")

    if file_format == "json":
        # Load structured prompts from JSON
        try:
            with open(path, encoding="utf-8") as f:
                prompt_data = json.load(f)

            if isinstance(prompt_data, list):
                prompts = [Prompt(**p) for p in prompt_data]
            else:
                prompts = [Prompt(**prompt_data)]

            # Apply category filter if specified
            if category:
                prompts = [p for p in prompts if p.category == category]

            return prompts

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts file {file_path}: {e}") from e

    elif file_format == "txt":
        # Load simple text prompts (one per line)
        with open(path, encoding="utf-8") as f:
            file_prompts = [line.strip() for line in f if line.strip()]

        prompts = [
            Prompt(
                id=f"prompt_{i:03d}",
                category=category or "custom",
                messages=[Message(role="user", content=line)],
            )
            for i, line in enumerate(file_prompts)
        ]
        return prompts

    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'json' or 'txt'")
