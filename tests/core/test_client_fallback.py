"""Tests for chat fallback and Ollama convenience wrappers (no network)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz import client
from hellmholtz.client import (
    ClientManager,
    FallbackError,
    chat_with_fallback,
    chat_with_fallback_detailed,
    ollama_chat,
)
from hellmholtz.providers.ollama import OllamaUnavailableError

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OLLAMA_API_URL", "HELLM_OLLAMA_MODEL", "HELLM_FALLBACK_MODELS"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _ollama_up() -> Any:
    """Default to a reachable server so failure tests never probe the network."""
    with patch.object(client.ollama, "is_available", return_value=True) as probe:
        yield probe


def _fake_chat(outcomes: dict[str, Any]) -> MagicMock:
    """chat() stand-in: value -> returned, exception -> raised, per model."""

    def side_effect(model: str, messages: Any, **kwargs: Any) -> str:
        outcome = outcomes[model]
        if isinstance(outcome, BaseException):
            raise outcome
        return str(outcome)

    return MagicMock(side_effect=side_effect)


class TestFallbackChain:
    def test_defaults_to_auto_ollama(self) -> None:
        assert client._fallback_chain("blablador:a", None) == ["ollama"]

    def test_env_fallbacks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HELLM_FALLBACK_MODELS", "openai:gpt-4o, llama3.2:3b")
        assert client._fallback_chain("blablador:a", None) == [
            "openai:gpt-4o",
            "ollama:llama3.2:3b",
        ]

    def test_argument_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HELLM_FALLBACK_MODELS", "openai:gpt-4o")
        assert client._fallback_chain("blablador:a", "mistral") == ["ollama:mistral"]

    def test_bare_name_with_colon_is_ollama(self) -> None:
        assert client._fallback_chain("x:y", ["llama3.2:3b"]) == ["ollama:llama3.2:3b"]

    def test_known_provider_kept(self) -> None:
        assert client._fallback_chain("x:y", ["anthropic:claude"]) == ["anthropic:claude"]

    def test_dedupes_and_drops_primary(self) -> None:
        chain = client._fallback_chain("ollama:m", ["m", "ollama:m", "other", "other"])
        assert chain == ["ollama:other"]


REAL_AISUITE_PROVIDERS = {"openai", "anthropic", "google", "ollama", "groq", "mistral", "azure"}


class TestProviderPrefixDetection:
    @pytest.fixture(autouse=True)
    def _aisuite_providers(self) -> Any:
        from aisuite.provider import ProviderFactory

        with patch.object(
            ProviderFactory, "get_supported_providers", return_value=REAL_AISUITE_PROVIDERS
        ):
            yield

    @pytest.mark.parametrize(
        "entry", ["groq:llama-3.1-8b", "azure:gpt-4o", "mistral:small", "openai:gpt-4o"]
    )
    def test_aisuite_providers_are_not_sent_to_ollama(self, entry: str) -> None:
        assert client._qualify_fallback(entry) == entry

    def test_blablador_kept_even_if_aisuite_does_not_list_it(self) -> None:
        assert client._qualify_fallback("blablador:alias-fast") == "blablador:alias-fast"

    def test_bare_ollama_names_still_qualified(self) -> None:
        assert client._qualify_fallback("llama3.2:3b") == "ollama:llama3.2:3b"
        assert client._qualify_fallback("phi3.5") == "ollama:phi3.5"

    def test_provider_prefix_wins_over_ollama_model_name(self) -> None:
        assert client._qualify_fallback("mistral:7b") == "mistral:7b"

    def test_explicit_ollama_prefix_forces_local_model(self) -> None:
        assert client._qualify_fallback("ollama:mistral:7b") == "ollama:mistral:7b"

    def test_static_floor_when_aisuite_reports_nothing(self) -> None:
        from aisuite.provider import ProviderFactory

        with patch.object(ProviderFactory, "get_supported_providers", return_value=set()):
            assert client._qualify_fallback("openai:gpt-4o") == "openai:gpt-4o"

    def test_chain_keeps_groq_entry(self) -> None:
        assert client._fallback_chain("a:b", ["groq:llama-3.1-8b", "llama3.2:3b"]) == [
            "groq:llama-3.1-8b",
            "ollama:llama3.2:3b",
        ]


class TestChatWithFallback:
    def test_primary_success_skips_fallbacks(self) -> None:
        fake = _fake_chat({"blablador:a": "primary"})
        with patch.object(client, "chat", fake):
            result = chat_with_fallback_detailed("blablador:a", MSGS)
        assert (result.text, result.model, result.used_fallback) == (
            "primary",
            "blablador:a",
            False,
        )
        assert fake.call_count == 1

    def test_falls_back_to_resolved_local_model(self) -> None:
        fake = _fake_chat({"blablador:a": RuntimeError("503"), "ollama:llama3.2": "local"})
        with (
            patch.object(client, "chat", fake),
            patch.object(client.ollama, "list_models", return_value=["llama3.2"]),
        ):
            result = chat_with_fallback_detailed("blablador:a", MSGS)
        assert result.text == "local"
        assert result.model == "ollama:llama3.2"
        assert result.attempts == [("blablador:a", "503")]
        assert result.used_fallback

    def test_walks_chain_in_order(self) -> None:
        fake = _fake_chat(
            {
                "blablador:a": RuntimeError("one"),
                "openai:gpt-4o": RuntimeError("two"),
                "ollama:mistral": "third",
            }
        )
        with patch.object(client, "chat", fake):
            result = chat_with_fallback_detailed(
                "blablador:a", MSGS, fallbacks=["openai:gpt-4o", "mistral"]
            )
        assert result.text == "third"
        assert [m for m, _ in result.attempts] == ["blablador:a", "openai:gpt-4o"]

    def test_kwargs_forwarded(self) -> None:
        fake = _fake_chat({"a:b": "ok"})
        with patch.object(client, "chat", fake):
            chat_with_fallback("a:b", MSGS, temperature=0.2, max_tokens=5)
        fake.assert_called_once_with("a:b", MSGS, temperature=0.2, max_tokens=5)

    def test_returns_text(self) -> None:
        with patch.object(client, "chat", _fake_chat({"a:b": "hello"})):
            assert chat_with_fallback("a:b", MSGS) == "hello"

    def test_all_fail_raises_with_attempts(self) -> None:
        fake = _fake_chat({"a:b": RuntimeError("down"), "ollama:m": ValueError("no model")})
        with patch.object(client, "chat", fake), pytest.raises(FallbackError) as exc:
            chat_with_fallback("a:b", MSGS, fallbacks="m")
        assert exc.value.attempts == [("a:b", "down"), ("ollama:m", "no model")]
        assert "a:b: down" in str(exc.value)

    def test_ollama_down_is_recorded_not_raised_raw(self) -> None:
        fake = _fake_chat({"a:b": RuntimeError("down")})
        with (
            patch.object(client, "chat", fake),
            patch.object(
                client.ollama,
                "resolve_model",
                side_effect=OllamaUnavailableError("not reachable"),
            ),
            pytest.raises(FallbackError) as exc,
        ):
            chat_with_fallback("a:b", MSGS)
        assert exc.value.attempts == [("a:b", "down"), ("ollama", "not reachable")]

    def test_keyboard_interrupt_not_swallowed(self) -> None:
        fake = _fake_chat({"a:b": KeyboardInterrupt()})
        with patch.object(client, "chat", fake), pytest.raises(KeyboardInterrupt):
            chat_with_fallback("a:b", MSGS)

    def test_primary_equals_only_fallback(self) -> None:
        fake = _fake_chat({"ollama:m": RuntimeError("down")})
        with patch.object(client, "chat", fake), pytest.raises(FallbackError) as exc:
            chat_with_fallback("ollama:m", MSGS, fallbacks="m")
        assert len(exc.value.attempts) == 1
        assert fake.call_count == 1


class TestOllamaChat:
    def test_string_prompt_becomes_user_message(self) -> None:
        fake = MagicMock(return_value="out")
        with patch.object(client, "chat", fake):
            assert ollama_chat("hello", model="llama3.2:3b", temperature=0) == "out"
        fake.assert_called_once_with(
            "ollama:llama3.2:3b", [{"role": "user", "content": "hello"}], temperature=0
        )

    def test_messages_passthrough(self) -> None:
        fake = MagicMock(return_value="out")
        with patch.object(client, "chat", fake):
            ollama_chat(MSGS, model="m")
        fake.assert_called_once_with("ollama:m", MSGS)

    def test_default_model_discovered(self) -> None:
        fake = MagicMock(return_value="out")
        with (
            patch.object(client, "chat", fake),
            patch.object(client.ollama, "list_models", return_value=["first", "second"]),
        ):
            ollama_chat("hi")
        assert fake.call_args.args[0] == "ollama:first"

    def test_default_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HELLM_OLLAMA_MODEL", "qwen3:8b")
        fake = MagicMock(return_value="out")
        with patch.object(client, "chat", fake):
            ollama_chat("hi")
        assert fake.call_args.args[0] == "ollama:qwen3:8b"

    def test_unavailable_raises(self) -> None:
        with (
            patch.object(client.ollama, "list_models", side_effect=OllamaUnavailableError("x")),
            pytest.raises(OllamaUnavailableError),
        ):
            ollama_chat("hi")


class TestUnreachableOllamaMessage:
    def test_ollama_chat_explains_unreachable_server(
        self, _ollama_up: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_API_URL", "http://gpu-box:11434")
        _ollama_up.return_value = False
        fake = _fake_chat({"ollama:m": RuntimeError("Connection error.")})
        with (
            patch.object(client, "chat", fake),
            pytest.raises(OllamaUnavailableError, match="gpu-box:11434.*OLLAMA_API_URL") as exc,
        ):
            ollama_chat("hi", model="m")
        assert isinstance(exc.value.__cause__, RuntimeError)

    def test_reachable_server_keeps_original_error(self) -> None:
        fake = _fake_chat({"ollama:m": RuntimeError("model 'm' not found")})
        with patch.object(client, "chat", fake), pytest.raises(RuntimeError, match="not found"):
            ollama_chat("hi", model="m")

    def test_fallback_attempt_records_explained_error(self, _ollama_up: MagicMock) -> None:
        _ollama_up.return_value = False
        fake = _fake_chat(
            {"a:b": RuntimeError("down"), "ollama:m": RuntimeError("Connection error.")}
        )
        with patch.object(client, "chat", fake), pytest.raises(FallbackError) as exc:
            chat_with_fallback("a:b", MSGS, fallbacks="m")
        assert "OLLAMA_API_URL" in exc.value.attempts[1][1]

    def test_non_ollama_failure_never_probes(self, _ollama_up: MagicMock) -> None:
        _ollama_up.return_value = False
        fake = _fake_chat({"a:b": RuntimeError("down")})
        with patch.object(client, "chat", fake), pytest.raises(RuntimeError, match="down"):
            client._chat_target("a:b", MSGS)
        _ollama_up.assert_not_called()


class TestClientRegistersOllamaUrl:
    def _configs(self) -> dict[str, Any]:
        ClientManager._default_instance = None
        with patch("hellmholtz.client.ai.Client") as cls:
            ClientManager._get_default_client()
        ClientManager._default_instance = None
        return dict(cls.call_args.kwargs["provider_configs"])

    def test_not_configured_by_default(self) -> None:
        assert "ollama" not in self._configs()

    def test_configured_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_API_URL", "http://gpu-box:11434")
        assert self._configs()["ollama"] == {"base_url": "http://gpu-box:11434"}
