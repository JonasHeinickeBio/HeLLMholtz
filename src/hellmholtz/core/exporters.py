"""Configuration exporters for popular AI/agent tools.

Supports exporting Blablador configurations to:
- OpenCode (JSON)
- Claude Code (JSON settings.json)
- Continue.dev (YAML config.yaml)
- Aider (YAML .aider.conf.yml)
- Cursor (Environment variables)
- Generic OpenAI-compatible (JSON)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import yaml

from .model_manager import ModelConfig


class ConfigExporter(ABC):
    """Base class for configuration exporters."""

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the target tool."""
        ...

    @property
    @abstractmethod
    def config_path(self) -> Path:
        """Path to the configuration file."""
        ...

    @abstractmethod
    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """
        Export configuration for the target tool.

        Args:
            models: List of model configurations
            output_path: Optional output path (defaults to config_path)
            merge: Whether to merge with existing config

        Returns:
            Path to the exported config file
        """
        ...

    def _load_existing(self, path: Path) -> dict[str, Any] | None:
        """Load existing configuration if it exists."""
        if not path.exists():
            return None

        try:
            if path.suffix == ".json":
                result: dict[str, Any] = json.loads(path.read_text())
                return result
            elif path.suffix in (".yaml", ".yml"):
                result = yaml.safe_load(path.read_text())
                return result
        except (json.JSONDecodeError, yaml.YAMLError):
            pass

        return None


class OpenCodeExporter(ConfigExporter):
    """Export configuration for OpenCode."""

    @property
    def tool_name(self) -> str:
        return "opencode"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".config" / "opencode" / "opencode.json"

    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """Export OpenCode configuration."""
        output = output_path or self.config_path
        output.parent.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {}
        if merge:
            existing = self._load_existing(output)
            if existing:
                config = existing

        # Ensure provider structure
        if "provider" not in config:
            config["provider"] = {}

        # Add models under blablador provider
        if "blablador" not in config["provider"]:
            config["provider"]["blablador"] = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Blablador",
                "options": {
                    "baseURL": models[0].api_base if models else "",
                    "apiKey": models[0].api_key if models else "",
                },
                "models": {},
            }

        provider = config["provider"]["blablador"]

        # Ensure options structure
        if "options" not in provider:
            provider["options"] = {}

        # Update API key and base URL in options
        if models and models[0].api_key:
            provider["options"]["apiKey"] = models[0].api_key
        if models and models[0].api_base:
            provider["options"]["baseURL"] = models[0].api_base

        # Ensure models is a dict (OpenCode uses dict keyed by model ID)
        if not isinstance(provider.get("models"), dict):
            provider["models"] = {}

        # Update models dict
        for model_config in models:
            model_id = model_config.model
            model_entry: dict[str, Any] = {
                "name": model_config.name,
                "limit": {
                    "context": model_config.context_length or 98304,
                    "output": model_config.max_tokens or 32768,
                },
            }
            provider["models"][model_id] = model_entry

        # Write config
        output.write_text(json.dumps(config, indent=2))
        return output


class ClaudeCodeExporter(ConfigExporter):
    """Export configuration for Claude Code."""

    @property
    def tool_name(self) -> str:
        return "claude-code"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".claude" / "settings.json"

    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """Export Claude Code settings.json."""
        output = output_path or self.config_path
        output.parent.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {}
        if merge:
            existing = self._load_existing(output)
            if existing:
                config = existing

        # Ensure env structure
        if "env" not in config:
            config["env"] = {}

        # Set environment variables for Blablador
        if models:
            primary = models[0]
            config["env"]["ANTHROPIC_BASE_URL"] = primary.api_base
            config["env"]["ANTHROPIC_API_KEY"] = primary.api_key

            # Set model if single model
            if len(models) == 1:
                config["model"] = primary.model

        output.write_text(json.dumps(config, indent=2))
        return output


class ContinueExporter(ConfigExporter):
    """Export configuration for Continue.dev."""

    @property
    def tool_name(self) -> str:
        return "continue"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".continue" / "config.yaml"

    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """Export Continue config.yaml."""
        output = output_path or self.config_path
        output.parent.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {}
        if merge:
            existing = self._load_existing(output)
            if existing:
                config = existing

        # Set required fields
        config.setdefault("name", "Blablador Configuration")
        config.setdefault("version", "1.0.0")
        config.setdefault("schema", "v1")

        # Build models list
        continue_models = []
        for model_config in models:
            continue_model: dict[str, Any] = {
                "name": model_config.name,
                "provider": "openai",
                "model": model_config.model,
                "apiBase": model_config.api_base,
                "apiKey": model_config.api_key,
            }
            if model_config.context_length:
                continue_model["defaultCompletionOptions"] = {
                    "contextLength": model_config.context_length,
                    "maxTokens": model_config.max_tokens or 4096,
                }
            if model_config.roles:
                continue_model["roles"] = model_config.roles
            continue_models.append(continue_model)

        config["models"] = continue_models

        # Write YAML
        output.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        return output


class AiderExporter(ConfigExporter):
    """Export configuration for Aider."""

    @property
    def tool_name(self) -> str:
        return "aider"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".aider.conf.yml"

    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """Export .aider.conf.yml."""
        output = output_path or self.config_path

        config: dict[str, Any] = {}
        if merge:
            existing = self._load_existing(output)
            if existing:
                config = existing

        if models:
            primary = models[0]
            # Set primary model with openai/ prefix for OpenAI-compatible
            config["model"] = f"openai/{primary.model}"
            config["openai-api-key"] = primary.api_key
            config["openai-api-base"] = primary.api_base

            # Add aliases for additional models
            if len(models) > 1:
                aliases = []
                for m in models[1:]:
                    aliases.append(f"{m.name}:openai/{m.model}")
                config["alias"] = aliases

            # Set weak model if available
            if len(models) > 1:
                config["weak-model"] = f"openai/{models[-1].model}"

        output.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        return output


class CursorExporter(ConfigExporter):
    """Export environment variables for Cursor."""

    @property
    def tool_name(self) -> str:
        return "cursor"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".cursor" / ".env"

    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """Export .env file for Cursor."""
        output = output_path or self.config_path
        output.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        if merge and output.exists():
            existing_lines = output.read_text().splitlines()
            # Keep non-OpenAI lines
            lines = [
                line
                for line in existing_lines
                if not line.startswith("OPENAI_API_KEY=")
                and not line.startswith("OPENAI_BASE_URL=")
            ]

        if models:
            primary = models[0]
            lines.append(f"OPENAI_API_KEY={primary.api_key}")
            lines.append(f"OPENAI_BASE_URL={primary.api_base}")

        output.write_text("\n".join(lines) + "\n")
        return output


class GenericOpenAIExporter(ConfigExporter):
    """Export generic OpenAI-compatible configuration."""

    @property
    def tool_name(self) -> str:
        return "generic-openai"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".config" / "hellmholtz" / "openai-compatible.json"

    def export(
        self,
        models: list[ModelConfig],
        output_path: Path | None = None,
        merge: bool = True,
    ) -> Path:
        """Export generic OpenAI-compatible JSON config."""
        output = output_path or self.config_path
        output.parent.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {}
        if merge:
            existing = self._load_existing(output)
            if existing:
                config = existing

        if models:
            primary = models[0]
            config["api_base"] = primary.api_base
            config["api_key"] = primary.api_key
            config["models"] = [m.to_dict() for m in models]

        output.write_text(json.dumps(config, indent=2))
        return output


# Registry of all exporters
EXPORTERS: dict[str, type[ConfigExporter]] = {
    "opencode": OpenCodeExporter,
    "claude-code": ClaudeCodeExporter,
    "continue": ContinueExporter,
    "aider": AiderExporter,
    "cursor": CursorExporter,
    "generic-openai": GenericOpenAIExporter,
}


def get_exporter(tool_name: str) -> ConfigExporter:
    """
    Get an exporter for a specific tool.

    Args:
        tool_name: Name of the target tool

    Returns:
        ConfigExporter instance

    Raises:
        ValueError: If tool_name is not supported
    """
    if tool_name not in EXPORTERS:
        supported = ", ".join(EXPORTERS.keys())
        raise ValueError(f"Unsupported tool: {tool_name}. Supported: {supported}")
    return EXPORTERS[tool_name]()


def list_exporters() -> list[str]:
    """
    List all supported tools.

    Returns:
        List of supported tool names
    """
    return list(EXPORTERS.keys())
