from collections.abc import Mapping, Sequence
import logging
from typing import Any, cast

import aisuite as ai

from hellmholtz.core.config import get_settings

logger = logging.getLogger(__name__)


class ClientManager:
    """Lazy singleton for aisuite Client."""

    _default_instance: ai.Client | None = None

    @classmethod
    def get_client(cls, model: str) -> tuple[ai.Client, str]:
        """
        Returns the appropriate client and the model name.
        """
        # We rely on aisuite's "provider:model" parsing.
        # We just ensure the provider is registered in _get_default_client.
        return cls._get_default_client(), model

    @classmethod
    def _get_default_client(cls) -> ai.Client:
        if cls._default_instance is None:
            # Register custom provider
            # 1. Inject module so importlib.import_module("aisuite.providers.blablador_provider")
            # works
            import sys

            import hellmholtz.providers.blablador_provider as blablador_provider

            sys.modules["aisuite.providers.blablador_provider"] = blablador_provider

            # 2. Monkey-patch ProviderFactory.get_supported_providers to include 'blablador'
            from aisuite.provider import ProviderFactory

            original_get_supported_providers = ProviderFactory.get_supported_providers

            @classmethod
            def patched_get_supported_providers(cls: Any) -> set[str]:
                # Clear cache if possible to ensure we get a fresh set if needed,
                # though strictly not required if we just add to the result.
                if hasattr(original_get_supported_providers, "cache_clear"):
                    original_get_supported_providers.cache_clear()

                providers = cast(set[str], original_get_supported_providers())
                return providers | {"blablador"}

            ProviderFactory.get_supported_providers = patched_get_supported_providers

            # Standard client using env vars
            # We configure blablador provider here so it's available in the default client
            settings = get_settings()
            config = {
                "blablador": {
                    "api_key": settings.blablador_api_key,
                    "base_url": settings.blablador_base_url,
                }
            }
            cls._default_instance = ai.Client(provider_configs=config)
            logger.info("Initialized aisuite Client with Blablador provider")
        return cls._default_instance


def chat(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    """High-level helper that returns the model's text content as a string."""
    client, effective_model = ClientManager.get_client(model)

    # Prepare kwargs
    call_args = kwargs.copy()
    if temperature is not None:
        call_args["temperature"] = temperature
    if max_tokens is not None:
        call_args["max_tokens"] = max_tokens

    logger.debug(f"Chat request to {model}: {messages}")
    try:
        response = client.chat.completions.create(
            model=effective_model, messages=messages, **call_args
        )

        # Log token usage if available
        if hasattr(response, "usage") and response.usage:
            logger.debug(
                f"Token usage for {model}: "
                f"prompt={response.usage.prompt_tokens}, "
                f"completion={response.usage.completion_tokens}, "
                f"total={response.usage.total_tokens}"
            )
    except Exception as e:
        logger.error(f"Chat completion failed for {model}: {e}")
        raise

    # Extract content - aisuite returns a standard response object
    content = response.choices[0].message.content
    return str(content) if content is not None else ""


def chat_raw(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> Any:
    """Low-level call that returns the full aisuite response."""
    client, effective_model = ClientManager.get_client(model)
    logger.debug(f"Raw chat request to {model}")
    try:
        return client.chat.completions.create(model=effective_model, messages=messages, **kwargs)
    except Exception as e:
        logger.error(f"Raw chat completion failed for {model}: {e}")
        raise


def check_model_availability(model: str) -> bool:
    """Check if a model is available by making a minimal test request.

    Args:
        model: Model identifier (e.g., "openai:gpt-4o", "blablador:gpt-4o")

    Returns:
        True if the model is available and can respond to requests
    """
    try:
        # For all models, try a minimal request using the chat function
        test_messages = [{"role": "user", "content": "test"}]
        chat(
            model=model,
            messages=test_messages,
            max_tokens=1,  # Minimal response
            temperature=0,  # Deterministic
        )
        logger.debug(f"Model {model} is available")
        return True

    except Exception as e:
        logger.debug(f"Model {model} availability check failed: {e}")
        return False
