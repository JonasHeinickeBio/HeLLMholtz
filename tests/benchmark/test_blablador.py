"""Tests for hellmholtz.providers.blablador module."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from hellmholtz.providers.blablador import (
    _build_known_model_indexes,
    _enrich_model_from_known_data,
    _find_best_known_match,
    _parse_raw_model_id,
    list_models,
    parse_api_model_ids,
)
from hellmholtz.providers.blablador_config import BlabladorModel, KNOWN_MODELS


# ---------------------------------------------------------------------------
# _build_known_model_indexes
# ---------------------------------------------------------------------------
class TestBuildKnownModelIndexes:
    def test_returns_two_dicts(self):
        by_id, by_name = _build_known_model_indexes()
        assert isinstance(by_id, dict)
        assert isinstance(by_name, dict)

    def test_known_by_id_keys_are_ids(self):
        by_id, _ = _build_known_model_indexes()
        for model in KNOWN_MODELS:
            if model.id:
                assert model.id in by_id

    def test_known_by_id_groups_models_with_same_id(self):
        by_id, _ = _build_known_model_indexes()
        assert len(by_id["01"]) == 2
        names = {m.name for m in by_id["01"]}
        assert "GPT-OSS-120b" in names
        assert "MiniMax-M2.7" in names

    def test_known_by_name_keys_are_names(self):
        _, by_name = _build_known_model_indexes()
        for model in KNOWN_MODELS:
            assert model.name in by_name

    def test_known_by_name_values_are_models(self):
        _, by_name = _build_known_model_indexes()
        for model in KNOWN_MODELS:
            assert by_name[model.name] is model

    def test_empty_id_models_grouped_under_empty_key(self):
        by_id, _ = _build_known_model_indexes()
        empty_id_models = [m for m in KNOWN_MODELS if m.id == ""]
        assert len(by_id[""]) == len(empty_id_models)


# ---------------------------------------------------------------------------
# _parse_raw_model_id
# ---------------------------------------------------------------------------
class TestParseRawModelId:
    def test_full_format_three_parts(self):
        result = _parse_raw_model_id("15 - Model Name - Description")
        assert result.id == "15"
        assert result.name == "Model Name"
        assert result.description == "Description"
        assert result.source == "Blablador"
        assert result.original_api_id == "15 - Model Name - Description"

    def test_full_format_two_parts(self):
        result = _parse_raw_model_id("15 - Model Name")
        assert result.id == "15"
        assert result.name == "Model Name"
        assert result.description == ""
        assert result.source == "Blablador"
        assert result.original_api_id == "15 - Model Name"

    def test_no_number_prefix(self):
        result = _parse_raw_model_id("alias-code")
        assert result.id == "alias-code"
        assert result.name == "alias-code"
        assert result.description == ""
        assert result.source == "Blablador"
        assert result.original_api_id == "alias-code"

    def test_description_with_hyphens(self):
        result = _parse_raw_model_id("10 - My Model - A multi-hyphen description")
        assert result.id == "10"
        assert result.name == "My Model"
        assert result.description == "A multi-hyphen description"

    def test_name_with_extra_spaces(self):
        result = _parse_raw_model_id("5 - Model  Name - Desc")
        assert result.id == "5"
        assert result.name == "Model  Name"
        assert result.description == "Desc"

    def test_empty_description_after_last_separator(self):
        result = _parse_raw_model_id("7 - Foo - ")
        assert result.id == "7"
        assert result.name == "Foo"
        assert result.description == ""

    def test_single_word_no_separators(self):
        result = _parse_raw_model_id("gpt-4o")
        assert result.id == "gpt-4o"
        assert result.name == "gpt-4o"
        assert result.description == ""


# ---------------------------------------------------------------------------
# _find_best_known_match
# ---------------------------------------------------------------------------
class TestFindBestKnownMatch:
    def _indexes(self):
        return _build_known_model_indexes()

    def test_exact_id_and_name_match(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="20", name="EVE-Instruct")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is not None
        assert result.name == "EVE-Instruct"

    def test_id_match_fuzzy_name_contains_known(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="20", name="EVE-Instruct-extended")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is not None
        assert result.name == "EVE-Instruct"

    def test_id_match_fuzzy_name_known_contains_model(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="20", name="EVE")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is not None
        assert result.name == "EVE-Instruct"

    def test_id_match_single_candidate_no_name_match(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="20", name="SomethingElse")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is not None
        assert result.name == "EVE-Instruct"

    def test_id_match_multiple_candidates_no_name_match_falls_through(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="01", name="CompletelyDifferent")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is None

    def test_no_id_match_fallback_to_known_by_name(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="9999", name="GPT-OSS-120b")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is not None
        assert result.name == "GPT-OSS-120b"

    def test_no_match_at_all(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="9999", name="NonexistentModel")
        result = _find_best_known_match(model, by_id, by_name)
        assert result is None


# ---------------------------------------------------------------------------
# _enrich_model_from_known_data
# ---------------------------------------------------------------------------
class TestEnrichModelFromKnownData:
    def _indexes(self):
        return _build_known_model_indexes()

    def test_match_found_merges_metadata(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="20", name="EVE-Instruct", description="", source="")
        result = _enrich_model_from_known_data(model, by_id, by_name)
        assert result.description != ""
        assert result.name == "EVE-Instruct"
        assert result.source == "Blablador"

    def test_match_found_keeps_existing_description(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(
            id="20", name="EVE-Instruct", description="Custom description", source="Other"
        )
        result = _enrich_model_from_known_data(model, by_id, by_name)
        assert result.description == "Custom description"
        assert result.name == "EVE-Instruct"
        assert result.source == "Blablador"

    def test_no_match_returns_unchanged(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="9999", name="Unknown", description="desc", source="src")
        result = _enrich_model_from_known_data(model, by_id, by_name)
        assert result.id == "9999"
        assert result.name == "Unknown"
        assert result.description == "desc"
        assert result.source == "src"

    def test_match_sets_alias(self):
        by_id, by_name = self._indexes()
        model = BlabladorModel(id="", name="alias-fast")
        result = _enrich_model_from_known_data(model, by_id, by_name)
        assert result.alias == "fast"


# ---------------------------------------------------------------------------
# parse_api_model_ids
# ---------------------------------------------------------------------------
class TestParseApiModelIds:
    def test_empty_list(self):
        assert parse_api_model_ids([]) == []

    def test_known_model_enriched(self):
        models = parse_api_model_ids(["20 - EVE-Instruct - Expert Earth"])
        assert len(models) == 1
        m = models[0]
        assert m.id == "20"
        assert m.name == "EVE-Instruct"
        assert m.source == "Blablador"

    def test_unknown_model_passes_through(self):
        models = parse_api_model_ids(["unknown-model"])
        assert len(models) == 1
        m = models[0]
        assert m.id == "unknown-model"
        assert m.name == "unknown-model"

    def test_mix_known_and_unknown(self):
        raw = [
            "20 - EVE-Instruct - Expert",
            "some-random-id",
            "15 - Apertus-8B-Instruct-2509 - A new swiss model",
        ]
        models = parse_api_model_ids(raw)
        assert len(models) == 3
        assert models[0].name == "EVE-Instruct"
        assert models[1].id == "some-random-id"
        assert models[2].name == "Apertus-8B-Instruct-2509"

    def test_multiple_raw_ids(self):
        raw = ["01 - GPT-OSS-120b - Open model", "02 - Qwen3.5-122B-A10B-FP8 - General"]
        models = parse_api_model_ids(raw)
        assert len(models) == 2
        assert models[0].name == "GPT-OSS-120b"
        assert models[1].name == "Qwen3.5-122B-A10B-FP8"

    def test_fuzzy_name_match_enrichment(self):
        models = parse_api_model_ids(["20 - EVE"])
        assert len(models) == 1
        m = models[0]
        assert m.name == "EVE-Instruct"
        assert m.source == "Blablador"


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------
class TestListModels:
    def _settings(self, key="test-key", url="https://api.test.com"):
        from hellmholtz.core.config import Settings

        return Settings(blablador_api_key=key, blablador_base_url=url, timeout_seconds=10.0)

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_success(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "20 - EVE-Instruct - Expert Earth"},
                {"id": "15 - Apertus-8B-Instruct-2509 - A new swiss model"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models()
        assert len(models) == 2
        assert models[0].name == "EVE-Instruct"
        assert models[1].name == "Apertus-8B-Instruct-2509"

    @patch("hellmholtz.providers.blablador.get_settings")
    def test_missing_api_key_raises(self, mock_settings):
        mock_settings.return_value = self._settings(key=None)
        with pytest.raises(ValueError, match="Blablador API key"):
            list_models()

    @patch("hellmholtz.providers.blablador.get_settings")
    def test_missing_base_url_raises(self, mock_settings):
        mock_settings.return_value = self._settings(url=None)
        with pytest.raises(ValueError, match="Blablador API key"):
            list_models()

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_http_error_raises_runtime(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=MagicMock(status_code=404)
        )
        with pytest.raises(RuntimeError, match="Failed to fetch models"):
            list_models()

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_non_dict_data_filtered(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "20 - EVE-Instruct - Expert Earth"},
                "not-a-dict",
                42,
                {"id": "15 - Apertus-8B-Instruct-2509 - A new swiss model"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models()
        assert len(models) == 2

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_empty_data_list(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models()
        assert models == []

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_missing_data_key(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models()
        assert models == []

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_url_trailing_slash_stripped(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings(url="https://api.test.com/")
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        list_models()
        called_url = mock_get.call_args[0][0]
        assert called_url == "https://api.test.com/models"

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_timeout_from_settings(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_settings.return_value.timeout_seconds = 60.0
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        list_models()
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 60.0

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_connection_error_raises_runtime(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        with pytest.raises(RuntimeError, match="Failed to fetch models"):
            list_models()

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_no_data_key_in_json(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings()
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "something"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models()
        assert models == []

    @patch("httpx.get")
    @patch("hellmholtz.providers.blablador.get_settings")
    def test_authorization_header(self, mock_settings, mock_get):
        mock_settings.return_value = self._settings(key="my-secret-key")
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        list_models()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer my-secret-key"
