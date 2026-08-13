"""Tests for hellmholtz.core.model_manager — target >80% coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from hellmholtz.core.model_manager import (
    BlabladorManager,
    Model,
    ModelConfig,
)

# ---------------------------------------------------------------------------
# Model dataclass
# ---------------------------------------------------------------------------


class TestModel:
    def test_creation_minimal(self):
        m = Model(id="gpt-4", name="gpt-4")
        assert m.id == "gpt-4"
        assert m.name == "gpt-4"
        assert m.description == ""
        assert m.context_length is None
        assert m.max_output_tokens is None
        assert m.provider == ""
        assert m.pricing == {}

    def test_creation_all_fields(self):
        m = Model(
            id="claude/opus",
            name="Claude Opus",
            description="A large model",
            context_length=200000,
            max_output_tokens=4096,
            provider="anthropic",
            pricing={"prompt": 0.01, "completion": 0.03},
        )
        assert m.context_length == 200000
        assert m.pricing == {"prompt": 0.01, "completion": 0.03}

    def test_to_dict_all_fields(self):
        m = Model(
            id="x",
            name="X",
            description="desc",
            context_length=1024,
            max_output_tokens=512,
            provider="prov",
            pricing={"p": 1.0},
        )
        d = m.to_dict()
        assert d == {
            "id": "x",
            "name": "X",
            "description": "desc",
            "context_length": 1024,
            "max_output_tokens": 512,
            "provider": "prov",
            "pricing": {"p": 1.0},
        }

    def test_to_dict_defaults(self):
        m = Model(id="a", name="a")
        d = m.to_dict()
        assert d["context_length"] is None
        assert d["max_output_tokens"] is None
        assert d["provider"] == ""
        assert d["pricing"] == {}


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_creation_minimal(self):
        c = ModelConfig(name="test")
        assert c.name == "test"
        assert c.provider == "blablador"
        assert c.model == ""
        assert c.api_base == ""
        assert c.api_key == ""
        assert c.context_length is None
        assert c.max_tokens is None
        assert c.roles == ["chat"]

    def test_creation_all_fields(self):
        c = ModelConfig(
            name="my-model",
            provider="openai",
            model="gpt-4",
            api_base="https://example.com",
            api_key="sk-123",
            context_length=8192,
            max_tokens=4096,
            roles=["chat", "completion"],
        )
        assert c.model == "gpt-4"
        assert c.roles == ["chat", "completion"]

    def test_to_dict_all_fields(self):
        c = ModelConfig(
            name="m",
            provider="prov",
            model="id",
            api_base="https://base",
            api_key="key",
            context_length=100,
            max_tokens=50,
            roles=["chat"],
        )
        d = c.to_dict()
        assert d == {
            "name": "m",
            "provider": "prov",
            "model": "id",
            "apiBase": "https://base",
            "apiKey": "key",
            "contextLength": 100,
            "maxTokens": 50,
            "roles": ["chat"],
        }

    def test_to_dict_omits_empty_optionals(self):
        c = ModelConfig(name="m")
        d = c.to_dict()
        assert "apiBase" not in d
        assert "apiKey" not in d
        assert "contextLength" not in d
        assert "maxTokens" not in d
        assert d.get("roles") == ["chat"]

    def test_to_dict_empty_roles(self):
        c = ModelConfig(name="m", roles=[])
        d = c.to_dict()
        assert "roles" not in d

    def test_to_dict_zero_max_tokens_included(self):
        c = ModelConfig(name="m", max_tokens=0)
        d = c.to_dict()
        assert "maxTokens" not in d


# ---------------------------------------------------------------------------
# BlabladorManager.__init__
# ---------------------------------------------------------------------------


class TestBlabladorManagerInit:
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults(self):
        mgr = BlabladorManager()
        assert mgr.api_base == BlabladorManager.DEFAULT_API_BASE
        assert mgr.api_key == ""
        assert mgr._models == []

    def test_custom_api_base(self):
        mgr = BlabladorManager(api_base="https://custom.example.com")
        assert mgr.api_base == "https://custom.example.com"

    @patch.dict(os.environ, {"BLABLADOR_API_KEY": "env-key-123"})
    def test_api_key_from_env(self):
        mgr = BlabladorManager()
        assert mgr.api_key == "env-key-123"

    def test_explicit_api_key_overrides_env(self):
        mgr = BlabladorManager(api_key="explicit")
        assert mgr.api_key == "explicit"

    def test_cache_file_path(self):
        mgr = BlabladorManager()
        assert mgr._cache_file == Path.home() / ".cache" / "hellmholtz" / "models.json"


# ---------------------------------------------------------------------------
# BlabladorManager.fetch_models
# ---------------------------------------------------------------------------


class TestFetchModels:
    def _make_response(self, models, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"data": models}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = Exception("HTTP Error")
        return resp

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_success(self, mock_get, tmp_path):
        mock_get.return_value = self._make_response([
            {
                "id": "gpt-4",
                "description": "A big model",
                "context_length": 8192,
                "max_output_tokens": 4096,
                "owned_by": "openai",
            },
            {
                "id": "org/small",
                "description": "Small model",
            },
        ])

        mgr = BlabladorManager()
        mgr._cache_file = tmp_path / "models.json"
        models = mgr.fetch_models(use_cache=False)

        assert len(models) == 2
        assert models[0].id == "gpt-4"
        assert models[0].name == "gpt-4"
        assert models[0].context_length == 8192
        assert models[0].provider == "openai"
        assert models[1].name == "small"

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_sends_auth_header(self, mock_get, tmp_path):
        mock_get.return_value = self._make_response([{"id": "m"}])

        mgr = BlabladorManager(api_key="test-key")
        mgr._cache_file = tmp_path / "models.json"
        mgr.fetch_models(use_cache=False)

        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"

    @patch("hellmholtz.core.model_manager.requests.get")
    @patch.dict(os.environ, {}, clear=True)
    def test_fetch_no_auth_when_no_key(self, mock_get, tmp_path):
        mock_get.return_value = self._make_response([])

        mgr = BlabladorManager()
        mgr._cache_file = tmp_path / "models.json"
        mgr.fetch_models(use_cache=False)

        _, kwargs = mock_get.call_args
        assert "Authorization" not in kwargs["headers"]

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_request_exception(self, mock_get, tmp_path):
        import requests as _requests
        mock_get.side_effect = _requests.RequestException("Network error")

        mgr = BlabladorManager()
        mgr._cache_file = tmp_path / "models.json"
        result = mgr.fetch_models(use_cache=False)

        assert result == []

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_caches_result(self, mock_get, tmp_path):
        mock_get.return_value = self._make_response([{"id": "m1"}])

        mgr = BlabladorManager(api_base="https://api.test.com/v1")
        mgr._cache_file = tmp_path / "models.json"
        mgr.fetch_models(use_cache=False)

        cache_file = tmp_path / "models.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["api_base"] == "https://api.test.com/v1"
        assert len(data["models"]) == 1

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_returns_cached(self, mock_get, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({
            "api_base": BlabladorManager.DEFAULT_API_BASE,
            "models": [{"id": "cached", "name": "cached", "description": "", "context_length": None, "max_output_tokens": None, "provider": "", "pricing": {}}],
        }))

        mgr = BlabladorManager()
        mgr._cache_file = cache_file
        models = mgr.fetch_models(use_cache=True)

        mock_get.assert_not_called()
        assert len(models) == 1
        assert models[0].id == "cached"

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_cache_wrong_api_base(self, mock_get, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps({
            "api_base": "https://other.example.com",
            "models": [{"id": "old", "name": "old", "description": "", "context_length": None, "max_output_tokens": None, "provider": "", "pricing": {}}],
        }))

        mock_get.return_value = self._make_response([{"id": "fresh"}])

        mgr = BlabladorManager()
        mgr._cache_file = cache_file
        models = mgr.fetch_models(use_cache=True)

        mock_get.assert_called_once()
        assert models[0].id == "fresh"

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_corrupt_json_cache(self, mock_get, tmp_path):
        cache_file = tmp_path / "models.json"
        cache_file.write_text("not valid json {{{")

        mock_get.return_value = self._make_response([{"id": "recovered"}])

        mgr = BlabladorManager()
        mgr._cache_file = cache_file
        models = mgr.fetch_models(use_cache=True)

        mock_get.assert_called_once()
        assert models[0].id == "recovered"

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_empty_response(self, mock_get, tmp_path):
        mock_get.return_value = self._make_response([])

        mgr = BlabladorManager()
        mgr._cache_file = tmp_path / "models.json"
        models = mgr.fetch_models(use_cache=False)

        assert models == []

    @patch("hellmholtz.core.model_manager.requests.get")
    def test_fetch_id_with_slash(self, mock_get, tmp_path):
        mock_get.return_value = self._make_response([
            {"id": "provider/model-name", "description": "d"},
        ])

        mgr = BlabladorManager()
        mgr._cache_file = tmp_path / "models.json"
        models = mgr.fetch_models(use_cache=False)

        assert models[0].id == "provider/model-name"
        assert models[0].name == "model-name"


# ---------------------------------------------------------------------------
# BlabladorManager.search_models
# ---------------------------------------------------------------------------


class TestSearchModels:
    def _make_manager_with_models(self):
        mgr = BlabladorManager()
        mgr._models = [
            Model(id="gpt-4o", name="GPT-4o", description="Fast GPT"),
            Model(id="claude-3", name="Claude 3", description="Anthropic model"),
        ]
        return mgr

    def test_search_by_id(self):
        mgr = self._make_manager_with_models()
        results = mgr.search_models("gpt-4o")
        assert len(results) == 1
        assert results[0].id == "gpt-4o"

    def test_search_by_name(self):
        mgr = self._make_manager_with_models()
        results = mgr.search_models("Claude")
        assert len(results) == 1
        assert results[0].name == "Claude 3"

    def test_search_by_description(self):
        mgr = self._make_manager_with_models()
        results = mgr.search_models("Anthropic")
        assert len(results) == 1
        assert results[0].id == "claude-3"

    def test_search_case_insensitive(self):
        mgr = self._make_manager_with_models()
        results = mgr.search_models("GPT")
        assert len(results) == 1

    def test_search_no_matches(self):
        mgr = self._make_manager_with_models()
        results = mgr.search_models("nonexistent")
        assert results == []

    @patch.object(BlabladorManager, "fetch_models")
    def test_search_empty_triggers_fetch(self, mock_fetch):
        mock_fetch.return_value = []
        mgr = BlabladorManager()
        mgr._models = []

        results = mgr.search_models("anything")
        mock_fetch.assert_called_once()
        assert results == []


# ---------------------------------------------------------------------------
# BlabladorManager.get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    def _make_manager_with_models(self):
        mgr = BlabladorManager()
        mgr._models = [
            Model(id="gpt-4o", name="GPT-4o"),
            Model(id="claude-3", name="Claude 3"),
        ]
        return mgr

    def test_get_by_id(self):
        mgr = self._make_manager_with_models()
        m = mgr.get_model("gpt-4o")
        assert m is not None
        assert m.id == "gpt-4o"

    def test_get_by_name(self):
        mgr = self._make_manager_with_models()
        m = mgr.get_model("Claude 3")
        assert m is not None
        assert m.id == "claude-3"

    def test_get_not_found(self):
        mgr = self._make_manager_with_models()
        assert mgr.get_model("nonexistent") is None

    @patch.object(BlabladorManager, "fetch_models")
    def test_get_empty_triggers_fetch(self, mock_fetch):
        mock_fetch.return_value = []
        mgr = BlabladorManager()
        mgr._models = []

        result = mgr.get_model("anything")
        mock_fetch.assert_called_once()
        assert result is None


# ---------------------------------------------------------------------------
# BlabladorManager.create_model_config
# ---------------------------------------------------------------------------


class TestCreateModelConfig:
    @patch.dict(os.environ, {}, clear=True)
    def test_create_config_defaults(self):
        mgr = BlabladorManager()
        model = Model(id="gpt-4", name="GPT-4", context_length=8192, max_output_tokens=4096)
        cfg = mgr.create_model_config(model)

        assert cfg.name == "GPT-4"
        assert cfg.provider == "blablador"
        assert cfg.model == "gpt-4"
        assert cfg.api_base == BlabladorManager.DEFAULT_API_BASE
        assert cfg.api_key == ""
        assert cfg.context_length == 8192
        assert cfg.max_tokens == 4096
        assert cfg.roles == ["chat"]

    def test_create_config_with_api_key_override(self):
        mgr = BlabladorManager(api_key="default-key")
        model = Model(id="m", name="M")
        cfg = mgr.create_model_config(model, api_key="override-key")
        assert cfg.api_key == "override-key"

    def test_create_config_uses_manager_key_when_no_override(self):
        mgr = BlabladorManager(api_key="manager-key")
        model = Model(id="m", name="M")
        cfg = mgr.create_model_config(model)
        assert cfg.api_key == "manager-key"

    def test_create_config_with_custom_roles(self):
        mgr = BlabladorManager()
        model = Model(id="m", name="M")
        cfg = mgr.create_model_config(model, roles=["completion", "embedding"])
        assert cfg.roles == ["completion", "embedding"]

    def test_create_config_default_roles_when_none(self):
        mgr = BlabladorManager()
        model = Model(id="m", name="M")
        cfg = mgr.create_model_config(model, roles=None)
        assert cfg.roles == ["chat"]

    def test_create_config_model_without_optional_fields(self):
        mgr = BlabladorManager(api_base="https://custom.api/v1")
        model = Model(id="x", name="X")
        cfg = mgr.create_model_config(model)
        assert cfg.context_length is None
        assert cfg.max_tokens is None
        assert cfg.api_base == "https://custom.api/v1"
