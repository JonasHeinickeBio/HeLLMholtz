"""Lightweight stub for the ``aisuite`` package used in tests.

Only the symbols required by the project are provided:

* ``Client`` – a simple wrapper that stores ``provider_configs``.
* ``provider`` submodule exposing ``Provider``, ``ProviderFactory`` and ``LLMError``.
"""

from __future__ import annotations

from typing import Any

from .provider import LLMError, Provider, ProviderFactory


class Client:
    """Minimal ``aisuite.Client`` implementation.

    The real client would handle authentication and request routing. For the
    test suite we only need to store the ``provider_configs`` dictionary and expose
    a ``chat`` attribute with a ``completions`` namespace that can be mocked.
    """

    def __init__(self, provider_configs: dict[str, Any] | None = None):
        self.provider_configs = provider_configs or {}

        # ``chat.completions.create`` will be accessed in tests via mocks, so we
        # provide a placeholder object.
        class _Completions:
            def __init__(self, outer: Client) -> None:
                self._outer = outer

            def create(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
                raise NotImplementedError("This method should be mocked in tests")

        class _Chat:
            def __init__(self, outer: Client) -> None:
                self.completions = _Completions(outer)

        self.chat = _Chat(self)
