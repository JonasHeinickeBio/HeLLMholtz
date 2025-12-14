import os
from typing import Any

from aisuite.provider import LLMError, Provider
from aisuite.providers.message_converter import OpenAICompliantMessageConverter
import openai

from hellmholtz.providers.blablador_config import KNOWN_MODELS


class BlabladorProvider(Provider):
    def __init__(self, **config: str | None) -> None:
        """
        Initialize the Blablador provider with the given configuration.
        """
        # Ensure API key and base URL are provided
        config.setdefault("api_key", os.getenv("BLABLADOR_API_KEY"))
        config.setdefault("base_url", os.getenv("BLABLADOR_API_BASE"))

        if not config["api_key"]:
            raise ValueError(
                "Blablador API key is missing. Please provide it in the config "
                "or set the BLABLADOR_API_KEY environment variable."
            )
        if not config["base_url"]:
            raise ValueError(
                "Blablador Base URL is missing. Please provide it in the config "
                "or set the BLABLADOR_API_BASE environment variable."
            )

        # Pass the config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(
        self, model: str, messages: list[dict[str, Any]], **kwargs: dict[str, Any]
    ) -> object:
        try:
            # Resolve model name
            resolved_model = model
            for m in KNOWN_MODELS:
                if m.id == model or m.name == model or m.alias == model:
                    resolved_model = m.api_id
                    break

            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=transformed_messages,
                **kwargs,
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
