import pytest
from unittest import mock

from hellmholtz.providers.blablador_config import (
    get_all_provider_token_limits,
    _ONLINE_TOKEN_CACHE,
    clear_online_token_cache,
)


def test_missing_openai_model_is_suppressed(monkeypatch):
    """Ensure that an exception raised for a specific OpenAI model is suppressed.

    The get_all_provider_token_limits function uses ``contextlib.suppress`` to ignore
    ``Exception`` while fetching token limits. By patching ``get_token_limit`` to raise
    for a known model name we can verify that the resulting dict simply omits that
    model rather than propagating the error.
    """

    def fake_get_token_limit(model_id: str):
        # Simulate failure for a specific model name
        if model_id == "openai:gpt-4":
            raise RuntimeError("simulated fetch error")
        # For all other models return a dummy integer
        return 12345

    monkeypatch.setattr(
        "hellmholtz.providers.blablador_config.get_token_limit",
        fake_get_token_limit,
    )

    result = get_all_provider_token_limits(include_online=False)
    # The OpenAI section should exist but not contain the failing model
    openai_models = result.get("openai", {})
    assert "gpt-4" not in openai_models
    # Other models should be present with the dummy value
    for name in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "text-davinci-003", "text-embedding-ada-002"]:
        if name == "gpt-4":
            continue
        assert openai_models.get(name) == 12345


def test_online_token_cache_filtering(monkeypatch):
    """Verify that ``include_online`` correctly filters out ``None`` values.

    The function should only copy entries where the cached limit is not ``None``.
    """

    # Prepare a fake cache with a valid entry and a ``None`` entry
    clear_online_token_cache()
    _ONLINE_TOKEN_CACHE.update({"valid-model": 1024, "none-model": None})

    result = get_all_provider_token_limits(include_online=True)
    online = result.get("online", {})
    assert "valid-model" in online and online["valid-model"] == 1024
    # ``none-model`` should be omitted because its value is ``None``
    assert "none-model" not in online
