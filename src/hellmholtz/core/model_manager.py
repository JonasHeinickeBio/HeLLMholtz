"""Blablador Model Manager - Discovery and configuration for OpenAI-compatible APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import requests


@dataclass
class Model:
    """Represents an LLM model from Blablador API."""

    id: str
    name: str
    description: str = ""
    context_length: int | None = None
    max_output_tokens: int | None = None
    provider: str = ""
    pricing: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "provider": self.provider,
            "pricing": self.pricing,
        }


@dataclass
class ModelConfig:
    """Configuration for a model in an AI tool."""

    name: str
    provider: str = "blablador"
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    context_length: int | None = None
    max_tokens: int | None = None
    roles: list[str] = field(default_factory=lambda: ["chat"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
        }
        if self.api_base:
            result["apiBase"] = self.api_base
        if self.api_key:
            result["apiKey"] = self.api_key
        if self.context_length:
            result["contextLength"] = self.context_length
        if self.max_tokens:
            result["maxTokens"] = self.max_tokens
        if self.roles:
            result["roles"] = self.roles
        return result


class BlabladorManager:
    """Manages Blablador API models and configurations."""

    DEFAULT_API_BASE = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the Blablador Manager.

        Args:
            api_base: Base URL for Blablador API
            api_key: API key for authentication
        """
        self.api_base = api_base or self.DEFAULT_API_BASE
        self.api_key = api_key or os.getenv("BLABLADOR_API_KEY", "")
        self._models: list[Model] = []
        self._cache_file = Path.home() / ".cache" / "hellmholtz" / "models.json"

    def fetch_models(self, use_cache: bool = True) -> list[Model]:
        """
        Fetch available models from Blablador API.

        Args:
            use_cache: Whether to use cached models

        Returns:
            List of available models
        """
        if use_cache and self._cache_file.exists():
            try:
                cache_data = json.loads(self._cache_file.read_text())
                if cache_data.get("api_base") == self.api_base:
                    self._models = [Model(**m) for m in cache_data.get("models", [])]
                    return self._models
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.api_base}/models",
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            models_data = data.get("data", [])

            self._models = []
            for m in models_data:
                model = Model(
                    id=m.get("id", ""),
                    name=m.get("id", "").split("/")[-1]
                    if "/" in m.get("id", "")
                    else m.get("id", ""),
                    description=m.get("description", ""),
                    context_length=m.get("context_length"),
                    max_output_tokens=m.get("max_output_tokens"),
                    provider=m.get("owned_by", ""),
                )
                self._models.append(model)

            # Cache the results
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "api_base": self.api_base,
                "models": [m.to_dict() for m in self._models],
            }
            self._cache_file.write_text(json.dumps(cache_data, indent=2))

            return self._models

        except requests.RequestException as e:
            print(f"Error fetching models: {e}")
            return self._models

    def search_models(self, query: str) -> list[Model]:
        """
        Search models by name or description.

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        if not self._models:
            self.fetch_models()

        query_lower = query.lower()
        return [
            m
            for m in self._models
            if query_lower in m.id.lower()
            or query_lower in m.name.lower()
            or query_lower in m.description.lower()
        ]

    def get_model(self, model_id: str) -> Model | None:
        """
        Get a specific model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Model if found, None otherwise
        """
        if not self._models:
            self.fetch_models()

        for m in self._models:
            if m.id == model_id or m.name == model_id:
                return m
        return None

    def create_model_config(
        self,
        model: Model,
        api_key: str | None = None,
        roles: list[str] | None = None,
    ) -> ModelConfig:
        """
        Create a ModelConfig for a specific model.

        Args:
            model: Model to configure
            api_key: API key to use
            roles: Model roles

        Returns:
            ModelConfig for the model
        """
        return ModelConfig(
            name=model.name,
            provider="blablador",
            model=model.id,
            api_base=self.api_base,
            api_key=api_key or self.api_key,
            context_length=model.context_length,
            max_tokens=model.max_output_tokens,
            roles=roles or ["chat"],
        )
