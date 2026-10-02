from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import logging
from typing import Any

import aisuite as ai
from hellmholtz.core.config import get_settings
from hellmholtz.core.logging_utils import failures_handled_downstream, log_failure, short
from hellmholtz.providers import ollama
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
            config: dict[str, dict[str, Any]] = {
                "blablador": {
                    "api_key": settings.blablador_api_key,
                    "base_url": settings.blablador_base_url,
                }
            }
            if settings.ollama_base_url:
                config["ollama"] = {"base_url": settings.ollama_base_url}
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
        log_failure(logger, f"Chat completion failed for {model}: {e}")
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
        log_failure(logger, f"Raw chat completion failed for {model}: {e}")
        raise


#: Provider prefixes always recognised when parsing a fallback entry, on top of
#: whatever aisuite reports (see :func:`_known_providers`). Anything else is
#: treated as a bare Ollama model name (which may itself contain a colon).
KNOWN_PROVIDERS = ("blablador", "openai", "anthropic", "google", "ollama")

#: Fallback entry that means "whatever local Ollama model is available".
AUTO_OLLAMA = "ollama"


class FallbackError(RuntimeError):
    """Raised when the primary model and every fallback failed.

    Attributes:
        attempts: ``(model, error message)`` for each model tried, in order.
    """

    def __init__(self, attempts: Sequence[tuple[str, str]]) -> None:
        self.attempts = list(attempts)
        detail = "; ".join(f"{model}: {error}" for model, error in self.attempts)
        super().__init__(f"All models failed ({detail})")


@dataclass
class FallbackResult:
    """Outcome of :func:`chat_with_fallback_detailed`.

    Attributes:
        text: The response text.
        model: The model that actually produced it.
        attempts: ``(model, error message)`` for each model that failed first.
    """

    text: str
    model: str
    attempts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def used_fallback(self) -> bool:
        """True if the response did not come from the primary model."""
        return bool(self.attempts)


def _chat_target(model: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> str:
    """Run :func:`chat`, explaining unreachable Ollama servers.

    aisuite only reports "Connection error" when the local server cannot be
    reached; for Ollama models the real cause (wrong host or port, server not
    running) is worth spelling out, including how to configure it.
    """
    try:
        return chat(model, messages, **kwargs)
    except Exception as e:
        if model.startswith(ollama.PROVIDER_PREFIX) and not ollama.is_available():
            raise ollama.OllamaUnavailableError(
                f"Ollama is not reachable at {ollama.get_base_url()} "
                "(set OLLAMA_API_URL if the server uses another host or port)"
            ) from e
        raise


def _known_providers() -> set[str]:
    """Provider prefixes aisuite can route, plus ours."""
    from aisuite.provider import ProviderFactory

    return set(ProviderFactory.get_supported_providers()) | set(KNOWN_PROVIDERS)


def _qualify_fallback(entry: str) -> str:
    """Normalise a fallback entry to a ``provider:model`` identifier.

    A leading provider prefix always wins, so ``mistral:7b`` means the Mistral
    API; write ``ollama:mistral:7b`` for the local model of the same name.
    """
    if entry == AUTO_OLLAMA or entry.split(":", 1)[0] in _known_providers():
        return entry
    return ollama.qualify(entry)


def _fallback_chain(primary: str, fallbacks: str | Sequence[str] | None) -> list[str]:
    """Build the ordered, de-duplicated list of models to try after *primary*.

    Precedence: *fallbacks* argument > ``HELLM_FALLBACK_MODELS`` > local Ollama.
    """
    if isinstance(fallbacks, str):
        fallbacks = [fallbacks]
    entries = list(fallbacks) if fallbacks is not None else get_settings().fallback_models
    chain: list[str] = []
    for entry in entries or [AUTO_OLLAMA]:
        qualified = _qualify_fallback(entry)
        if qualified != primary and qualified not in chain:
            chain.append(qualified)
    return chain


def chat_with_fallback_detailed(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    fallbacks: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> FallbackResult:
    """Like :func:`chat_with_fallback`, but also reports which model answered.

    Args:
        model: Primary model (``provider:model``).
        messages: Chat messages.
        fallbacks: Models to try, in order, if the primary fails. Entries are
            ``provider:model`` identifiers, bare Ollama model names
            (``llama3.2:3b``), or ``"ollama"`` for the preferred/first local
            model. Defaults to ``HELLM_FALLBACK_MODELS``, then local Ollama.
        **kwargs: Passed through to :func:`chat` (``temperature`` etc.).

    Raises:
        FallbackError: If every model failed.
    """
    attempts: list[tuple[str, str]] = []
    for candidate in [model, *_fallback_chain(model, fallbacks)]:
        try:
            target = ollama.resolve_model() if candidate == AUTO_OLLAMA else candidate
            with failures_handled_downstream():
                text = _chat_target(target, messages, **kwargs)
        except Exception as e:
            logger.warning(f"Model {candidate} failed: {short(e)}")
            attempts.append((candidate, str(e)))
            continue
        if attempts:
            logger.warning(f"Fell back to {target} after {len(attempts)} failed attempt(s)")
        return FallbackResult(text=text, model=target, attempts=attempts)
    raise FallbackError(attempts)


def chat_with_fallback(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    fallbacks: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> str:
    """Chat with *model*, falling back to other models (default: local Ollama).

    Any exception from a model moves on to the next one in the chain. See
    :func:`chat_with_fallback_detailed` for the arguments, and for a variant
    that reports which model answered.

    Raises:
        FallbackError: If every model failed.
    """
    return chat_with_fallback_detailed(model, messages, fallbacks=fallbacks, **kwargs).text


def ollama_chat(
    prompt: str | Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    **kwargs: Any,
) -> str:
    """Chat with a local Ollama model.

    Args:
        prompt: A user message, or a full list of chat messages.
        model: Ollama model name (``llama3.2:3b`` or ``ollama:llama3.2:3b``).
            Defaults to ``HELLM_OLLAMA_MODEL``, then the first installed model.
        **kwargs: Passed through to :func:`chat` (``temperature`` etc.).

    Raises:
        OllamaUnavailableError: If no model was given and none can be found.
    """
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    return _chat_target(ollama.resolve_model(model), messages, **kwargs)


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
