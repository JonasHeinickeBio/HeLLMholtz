"""Local Ollama discovery helpers.

Ollama is served by aisuite's built-in ``ollama`` provider, so this module only
adds what aisuite does not: probing whether a local server is running, listing
the models installed on it, and choosing one to use as a fallback target.

Probes use the native ``/api/tags`` endpoint with the standard library only, and
never raise: failures are reported through return values or
:class:`OllamaUnavailableError`.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from hellmholtz.core.config import get_settings

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "OllamaUnavailableError",
    "get_base_url",
    "is_available",
    "list_models",
    "qualify",
    "resolve_model",
]

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 3.0

PROVIDER_PREFIX = "ollama:"


class OllamaUnavailableError(RuntimeError):
    """Raised when no usable local Ollama server or model can be found."""


def get_base_url() -> str:
    """Return the Ollama server root (no trailing slash, no ``/v1`` suffix)."""
    url = get_settings().ollama_base_url or DEFAULT_BASE_URL
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def qualify(model: str) -> str:
    """Return *model* as an aisuite identifier (``ollama:<name>``).

    Ollama model names themselves contain colons (``llama3.2:3b``), so the
    prefix is added unless it is already present.
    """
    return model if model.startswith(PROVIDER_PREFIX) else f"{PROVIDER_PREFIX}{model}"


def list_models(timeout: float = DEFAULT_TIMEOUT) -> list[str]:
    """Return the names of locally installed Ollama models.

    Raises:
        OllamaUnavailableError: If the server cannot be reached or answers with
            something that is not an Ollama model listing.
    """
    url = f"{get_base_url()}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
        return [entry["name"] for entry in payload.get("models", [])]
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise OllamaUnavailableError(f"Ollama is not reachable at {get_base_url()}: {e}") from e
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise OllamaUnavailableError(f"Unexpected response from {url}: {e}") from e


def is_available(timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Return True if an Ollama server answers at the configured URL."""
    try:
        list_models(timeout=timeout)
    except OllamaUnavailableError:
        return False
    return True


def resolve_model(preferred: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Pick the Ollama model to use and return it as ``ollama:<name>``.

    Precedence: *preferred* argument > ``HELLM_OLLAMA_MODEL`` > the first
    installed model. An explicitly requested model is returned as-is without
    consulting the server, so that a missing model surfaces as a normal chat
    error rather than being silently swapped for a different one.

    Raises:
        OllamaUnavailableError: If nothing was requested and the server is
            unreachable or has no models installed.
    """
    requested = preferred or get_settings().ollama_model
    if requested:
        return qualify(requested)

    installed = list_models(timeout=timeout)
    if not installed:
        raise OllamaUnavailableError(
            f"Ollama is running at {get_base_url()} but has no models installed "
            "(try `ollama pull llama3.2`)"
        )
    return qualify(installed[0])
