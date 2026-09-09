"""Stub implementation of ``aisuite.provider`` required for the test suite.

The real ``aisuite`` library provides a rich plugin system for LLM providers.
For our purposes we only need the following symbols:

* ``Provider`` – a minimal base class that user‑defined providers subclass.
* ``ProviderFactory`` – a factory with a ``get_supported_providers`` classmethod.
* ``LLMError`` – exception type raised by providers.

The functions are deliberately simple; the test suite mocks the behavior of the
client and provider methods, so these definitions merely need to exist and be
importable.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Provider", "ProviderFactory", "LLMError"]


class LLMError(Exception):
    """Exception raised for provider‑specific errors."""


class Provider:
    """Base class for LLM providers.

    Concrete providers (e.g., ``BlabladorProvider``) inherit from this class.
    No functionality is required for the tests; the class exists solely for
    type checking and ``isinstance`` checks.
    """

    def __init__(
        self, **config: Any
    ) -> None:  # pragma: no cover – instantiated in tests via mocks
        self.config = config

    # The real library defines many methods; we provide a placeholder that can be
    # overridden by subclasses.
    def chat_completions_create(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class ProviderFactory:
    """Factory that reports supported provider names.

    The ``ClientManager`` in ``hellmholtz.client`` monkey‑patches the
    ``get_supported_providers`` method to include ``"blablador"``.  The default
    implementation simply returns a set with the common providers.
    """

    @classmethod
    def get_supported_providers(cls) -> set[str]:  # pragma: no cover – overridden in tests
        return {"openai", "anthropic", "google", "ollama"}
