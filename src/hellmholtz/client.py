from collections.abc import Mapping, Sequence
import logging
from typing import Any

import aisuite as ai
from hellmholtz.core.config import get_settings
from hellmholtz.providers.blablador_config import get_model_by_name

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

                providers = original_get_supported_providers()
                return providers | {"blablador"}

            ProviderFactory.get_supported_providers = patched_get_supported_providers  # type: ignore[assignment]

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


def _resolve_systemone_model(model: str) -> str | None:
    """Return the model name if *model* refers to a System-One model.

    Accepts plain names/aliases ("alias-laya", "laya") as well as
    provider-prefixed identifiers ("blablador:alias-laya").

    Args:
        model: Model identifier.

    Returns:
        The System-One model name, or None if *model* is not a System-One
        model.
    """
    name = model.split(":", 1)[1] if ":" in model else model
    known = get_model_by_name(name)
    if known is not None and known.model_kind == "systemone":
        return known.name
    return None


def check_model_availability(model: str) -> bool:
    """Check if a model is available by making a minimal test request.

    System-One typed-decision models (e.g. "alias-laya") have no chat
    surface, so they are probed through the dedicated /v1/systemone
    endpoint instead of a chat completion.

    Args:
        model: Model identifier (e.g., "openai:gpt-4o", "blablador:gpt-4o",
            "alias-laya")

    Returns:
        True if the model is available and can respond to requests
    """
    try:
        # System-One models are probed via their typed-decision endpoint
        from hellmholtz.providers import systemone

        systemone_model = _resolve_systemone_model(model)
        if systemone_model is not None:
            return systemone.check_availability(systemone_model)

        # For all other models, try a minimal request using the chat function
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
