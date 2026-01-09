import logging
import os
from typing import Any

from aisuite.provider import LLMError, Provider
import openai

from hellmholtz.providers.blablador_config import KNOWN_MODELS

logger = logging.getLogger(__name__)


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
        logger.info("BlabladorProvider initialized")

    def chat_completions_create(
        self, model: str, messages: list[dict[str, Any]], **kwargs: dict[str, Any]
    ) -> object:
        logger.debug(f"Chat completion request for {model}")
        try:
            # Resolve model name
            resolved_model = model
            for m in KNOWN_MODELS:
                if m.id == model or m.name == model or m.alias == model:
                    resolved_model = m.api_id
                    break

            response = self.client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                **kwargs,
            )
            return response
        except openai.APIConnectionError as e:
            # Handle connection errors (e.g., server down, DNS issues, or localhost redirects)
            error_msg = str(e)
            if "localhost" in error_msg or "127.0.0.1" in error_msg:
                logger.error(f"Server configuration error for {model}: {e}")
                raise LLMError(
                    f"Server configuration error: The API is redirecting to localhost. "
                    f"This is likely a server-side misconfiguration for model '{model}'. "
                    f"Details: {e}"
                ) from e
            logger.error(f"Connection error: {e}")
            raise LLMError(f"Connection error: {e}") from e
        except openai.APIStatusError as e:
            # Handle API status errors (e.g., 400, 500)
            logger.error(f"API status error: {e}")

            # Check for server-side connection errors reflected in the 500 response
            error_msg = str(e)
            if "localhost" in error_msg or "127.0.0.1" in error_msg:
                raise LLMError(
                    f"Server configuration error: The API returned an error indicating it failed "
                    f"to reach an internal service at localhost. "
                    f"This is a server-side misconfiguration for model '{model}'. "
                    f"Details: {e}"
                ) from e

            if e.status_code == 400:
                raise LLMError(f"Bad Request: {e.message}") from e
            raise LLMError(f"API Error ({e.status_code}): {e.message}") from e
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise LLMError(f"An error occurred: {e}") from e
