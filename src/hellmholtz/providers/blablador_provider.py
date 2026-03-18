import logging
import os
import time
from typing import Any

from aisuite.provider import LLMError, Provider
import openai

from hellmholtz.providers.blablador import list_models
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
        self._available_models: list[str] | None = None
        self._models_cache_time: float | None = None
        self._cache_ttl = 300  # 5 minutes TTL for model list
        logger.info("BlabladorProvider initialized")

    def _get_available_models(self) -> list[str]:
        """Get list of available model API IDs from the API."""
        current_time = time.time()
        if (
            self._available_models is None
            or self._models_cache_time is None
            or current_time - self._models_cache_time > self._cache_ttl
        ):
            try:
                models = list_models()
                self._available_models = [m.api_id for m in models]
                self._models_cache_time = current_time
                logger.debug(f"Cached {len(self._available_models)} available models")
            except Exception as e:
                logger.warning(
                    f"Failed to fetch available models: {e}. Using known models as fallback."
                )
                # Fallback to known models if API fetch fails
                self._available_models = [m.api_id for m in KNOWN_MODELS]
                self._models_cache_time = current_time
        return self._available_models

    def check_model_availability(self, model: str) -> bool:
        """Check if a model is available by making a minimal test request.

        Args:
            model: Model identifier (name, alias, or ID)

        Returns:
            True if the model is available and can respond to requests
        """
        try:
            # Resolve model name to API ID
            resolved_model = model
            for m in KNOWN_MODELS:
                if m.id == model or m.name == model or m.alias == model:
                    resolved_model = m.api_id
                    break

            # Make a minimal test request
            test_messages = [{"role": "user", "content": "test"}]
            self.client.chat.completions.create(
                model=resolved_model,
                messages=test_messages,
                max_tokens=1,  # Minimal response
                temperature=0,  # Deterministic
            )

            # If we get here, the model is available
            logger.debug(f"Model {model} (resolved to {resolved_model}) is available")
            return True

        except Exception as e:
            logger.debug(f"Model {model} availability check failed: {e}")
            return False

    def chat_completions_create(  # noqa: C901
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

            # Check if the resolved model is available
            available_models = self._get_available_models()
            if resolved_model not in available_models:
                available_names = []
                for m in KNOWN_MODELS:
                    if m.api_id in available_models:
                        names = [m.name]
                        if m.alias:
                            names.append(m.alias)
                        available_names.extend(names)

                error_msg = (
                    f"Model '{model}' (resolved to '{resolved_model}') is not currently available."
                    f"This may indicate the model has been removed or renamed by the API provider."
                    f"Available models: {', '.join(sorted(set(available_names)))}"
                )
                logger.error(error_msg)
                raise LLMError(error_msg)

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
