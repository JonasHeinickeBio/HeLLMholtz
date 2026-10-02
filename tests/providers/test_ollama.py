"""Tests for local Ollama discovery helpers (no network)."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.providers import ollama


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OLLAMA_API_URL", "HELLM_OLLAMA_MODEL", "HELLM_FALLBACK_MODELS"):
        monkeypatch.delenv(var, raising=False)


def _response(body: str) -> MagicMock:
    resp = MagicMock()
    resp.read.return_value = body.encode("utf-8")
    resp.__enter__.return_value = resp
    return resp


def _tags(*names: str) -> str:
    return json.dumps({"models": [{"name": n} for n in names]})


class TestBaseUrl:
    def test_default(self) -> None:
        assert ollama.get_base_url() == "http://localhost:11434"

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_API_URL", "http://gpu-box:11434/")
        assert ollama.get_base_url() == "http://gpu-box:11434"

    def test_strips_v1_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_API_URL", "http://gpu-box:11434/v1")
        assert ollama.get_base_url() == "http://gpu-box:11434"


class TestQualify:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("llama3.2", "ollama:llama3.2"),
            ("llama3.2:3b", "ollama:llama3.2:3b"),
            ("ollama:llama3.2:3b", "ollama:llama3.2:3b"),
        ],
    )
    def test_qualify(self, name: str, expected: str) -> None:
        assert ollama.qualify(name) == expected


class TestListModels:
    def test_lists_names(self) -> None:
        with patch("urllib.request.urlopen", return_value=_response(_tags("a:1b", "b"))) as m:
            assert ollama.list_models() == ["a:1b", "b"]
        assert m.call_args.args[0] == "http://localhost:11434/api/tags"

    def test_empty(self) -> None:
        with patch("urllib.request.urlopen", return_value=_response(_tags())):
            assert ollama.list_models() == []

    def test_unreachable(self) -> None:
        with (
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")),
            pytest.raises(ollama.OllamaUnavailableError, match="not reachable"),
        ):
            ollama.list_models()

    def test_timeout(self) -> None:
        with (
            patch("urllib.request.urlopen", side_effect=TimeoutError("slow")),
            pytest.raises(ollama.OllamaUnavailableError),
        ):
            ollama.list_models()

    @pytest.mark.parametrize("body", ["not json", "[]", '{"models": [{"id": "x"}]}'])
    def test_unexpected_payload(self, body: str) -> None:
        with (
            patch("urllib.request.urlopen", return_value=_response(body)),
            pytest.raises(ollama.OllamaUnavailableError, match="Unexpected response"),
        ):
            ollama.list_models()

    def test_http_error_is_unavailable(self) -> None:
        err = urllib.error.HTTPError("u", 500, "boom", {}, io.BytesIO(b""))  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(ollama.OllamaUnavailableError),
        ):
            ollama.list_models()


class TestIsAvailable:
    def test_true(self) -> None:
        with patch("urllib.request.urlopen", return_value=_response(_tags())):
            assert ollama.is_available() is True

    def test_false(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            assert ollama.is_available() is False


class TestResolveModel:
    def test_preferred_wins_without_probing(self) -> None:
        with patch("urllib.request.urlopen") as m:
            assert ollama.resolve_model("mistral") == "ollama:mistral"
        m.assert_not_called()

    def test_env_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HELLM_OLLAMA_MODEL", "qwen3:8b")
        with patch("urllib.request.urlopen") as m:
            assert ollama.resolve_model() == "ollama:qwen3:8b"
        m.assert_not_called()

    def test_preferred_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HELLM_OLLAMA_MODEL", "qwen3:8b")
        assert ollama.resolve_model("phi4") == "ollama:phi4"

    def test_first_installed(self) -> None:
        with patch("urllib.request.urlopen", return_value=_response(_tags("llama3.2:3b", "x"))):
            assert ollama.resolve_model() == "ollama:llama3.2:3b"

    def test_no_models_installed(self) -> None:
        with (
            patch("urllib.request.urlopen", return_value=_response(_tags())),
            pytest.raises(ollama.OllamaUnavailableError, match="no models installed"),
        ):
            ollama.resolve_model()

    def test_server_down(self) -> None:
        with (
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")),
            pytest.raises(ollama.OllamaUnavailableError),
        ):
            ollama.resolve_model()
