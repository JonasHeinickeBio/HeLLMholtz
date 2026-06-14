"""Tests for BlabladorProvider (hellmholtz.providers.blablador_provider)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import openai
import pytest

from aisuite.provider import LLMError

from hellmholtz.providers.blablador_provider import BlabladorProvider
from hellmholtz.providers.blablador_config import KNOWN_MODELS, BlabladorModel


def _make_provider(**overrides) -> BlabladorProvider:
    defaults = {"api_key": "test-key", "base_url": "https://test.example.com/v1"}
    defaults.update(overrides)
    return BlabladorProvider(**defaults)


# ── __init__ tests ────────────────────────────────────────────────────────────


class TestInit:
    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_init_with_config(self, mock_openai_cls: MagicMock) -> None:
        p = _make_provider(api_key="k", base_url="https://x.com")
        assert p._available_models is None
        mock_openai_cls.assert_called_once_with(api_key="k", base_url="https://x.com")

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch.dict("os.environ", {"BLABLADOR_API_KEY": "env-key", "BLABLADOR_API_BASE": "https://env.com"})
    def test_init_with_env_vars(self, mock_openai_cls: MagicMock) -> None:
        p = BlabladorProvider()
        assert p._available_models is None

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key is missing"):
            BlabladorProvider(base_url="https://x.com")

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Base URL is missing"):
            BlabladorProvider(api_key="k")

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_both_raises(self, mock_openai_cls: MagicMock) -> None:
        with pytest.raises(ValueError, match="API key is missing"):
            BlabladorProvider()


# ── _get_available_models tests ──────────────────────────────────────────────


class TestGetAvailableModels:
    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_cache_miss_fetches_models(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        m1 = MagicMock(api_id="model-1")
        m2 = MagicMock(api_id="model-2")
        mock_list.return_value = [m1, m2]
        p = _make_provider()
        result = p._get_available_models()
        assert result == ["model-1", "model-2"]
        mock_list.assert_called_once()
        assert p._models_cache_time is not None

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_cache_hit_returns_cached(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        m1 = MagicMock(api_id="m1")
        mock_list.return_value = [m1]
        p = _make_provider()
        p._get_available_models()
        p._get_available_models()
        assert mock_list.call_count == 1

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_cache_expired_refetches(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        m1 = MagicMock(api_id="m1")
        mock_list.return_value = [m1]
        p = _make_provider()
        p._get_available_models()
        p._models_cache_time = time.time() - 400  # expire
        p._get_available_models()
        assert mock_list.call_count == 2

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_api_failure_fallback(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.side_effect = Exception("API down")
        p = _make_provider()
        result = p._get_available_models()
        assert isinstance(result, list)
        assert len(result) > 0


# ── check_model_availability tests ───────────────────────────────────────────


class TestCheckModelAvailability:
    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_success(self, mock_openai_cls: MagicMock) -> None:
        p = _make_provider()
        p.client.chat.completions.create.return_value = MagicMock()
        assert p.check_model_availability("gpt-4") is True

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_failure_returns_false(self, mock_openai_cls: MagicMock) -> None:
        p = _make_provider()
        p.client.chat.completions.create.side_effect = Exception("model not found")
        assert p.check_model_availability("nonexistent") is False

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_model_name_resolution(self, mock_openai_cls: MagicMock) -> None:
        p = _make_provider()
        p.client.chat.completions.create.return_value = MagicMock()
        known = KNOWN_MODELS[0]
        p.check_model_availability(known.name)
        call_args = p.client.chat.completions.create.call_args
        assert call_args[1]["model"] == known.api_id or call_args[0][0] == known.api_id


# ── chat_completions_create tests ────────────────────────────────────────────


class TestChatCompletionsCreate:
    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_success(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        p.client.chat.completions.create.return_value = MagicMock(id="resp-1")
        msgs = [{"role": "user", "content": "hi"}]
        result = p.chat_completions_create("alias-code", msgs)
        assert result.id == "resp-1"

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_model_not_available_raises(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="other-model")]
        p = _make_provider()
        with pytest.raises(LLMError, match="not currently available"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_api_connection_error(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        p.client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        with pytest.raises(LLMError, match="Connection error"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_localhost_redirect(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        err = openai.APIConnectionError(
            message="Redirect to localhost detected", request=MagicMock()
        )
        p.client.chat.completions.create.side_effect = err
        with pytest.raises(LLMError, match="localhost"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_api_status_error_400(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        resp = MagicMock(status_code=400, headers={})
        p.client.chat.completions.create.side_effect = openai.APIStatusError(
            message="Bad request", response=resp, body=None
        )
        with pytest.raises(LLMError, match="Bad Request"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_api_status_error_500(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        resp = MagicMock(status_code=500, headers={})
        p.client.chat.completions.create.side_effect = openai.APIStatusError(
            message="Server error", response=resp, body=None
        )
        with pytest.raises(LLMError, match="API Error \\(500\\)"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_api_status_error_localhost(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        resp = MagicMock(status_code=500, headers={})
        err = openai.APIStatusError(message="localhost error", response=resp, body=None)
        p.client.chat.completions.create.side_effect = err
        with pytest.raises(LLMError, match="localhost"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_general_exception(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("something broke")
        with pytest.raises(LLMError, match="An error occurred"):
            p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}])

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    @patch("hellmholtz.providers.blablador_provider.list_models")
    def test_kwargs_forwarded(self, mock_list: MagicMock, mock_openai_cls: MagicMock) -> None:
        mock_list.return_value = [MagicMock(api_id="alias-code")]
        p = _make_provider()
        p.client.chat.completions.create.return_value = MagicMock()
        p.chat_completions_create("alias-code", [{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=100)
        call_kwargs = p.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100
