"""Comprehensive tests for hellmholtz.providers.blablador_config."""

import json
import urllib.error
import urllib.request
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.providers.blablador_config import (
    DEFAULT_TOKEN_LIMIT,
    KNOWN_MODELS,
    BaseModel,
    BlabladorModel,
    _ONLINE_TOKEN_CACHE,
    _build_hf_search_patterns,
    _build_hf_search_terms,
    _extract_context_length_from_hf_model,
    _fetch_first_hf_candidate,
    _fetch_hf_model_details,
    _fetch_huggingface_model_info,
    _get_blablador_token_limit,
    _get_google_token_limit,
    _get_model_family_context_length,
    _get_online_token_limit,
    _get_ollama_token_limit,
    _get_openai_token_limit,
    _get_anthropic_token_limit,
    _get_provider_token_limit,
    _search_hf_model_candidates,
    clear_online_token_cache,
    get_all_provider_token_limits,
    get_model_by_name,
    get_token_limit,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear online token cache before each test."""
    clear_online_token_cache()
    yield
    clear_online_token_cache()


# ---------------------------------------------------------------------------
# BaseModel tests
# ---------------------------------------------------------------------------
class TestBaseModel:
    def test_create_minimal(self):
        m = BaseModel(name="test")
        assert m.name == "test"
        assert m.alias is None
        assert m.description == ""
        assert m.source == ""

    def test_create_with_all_fields(self):
        m = BaseModel(name="M", alias="a", description="d", source="s")
        assert m.name == "M"
        assert m.alias == "a"
        assert m.description == "d"
        assert m.source == "s"

    def test_is_dataclass(self):
        assert hasattr(BaseModel, "__dataclass_fields__")

    def test_fields_count(self):
        assert len(fields(BaseModel)) == 4


# ---------------------------------------------------------------------------
# BlabladorModel tests
# ---------------------------------------------------------------------------
class TestBlabladorModel:
    def test_create_minimal(self):
        m = BlabladorModel(name="test")
        assert m.name == "test"
        assert m.id == ""
        assert m.original_api_id is None
        assert m.description_separator == " - "
        assert m.max_context_tokens == DEFAULT_TOKEN_LIMIT

    def test_create_with_all_fields(self):
        m = BlabladorModel(
            name="M", id="01", original_api_id="orig",
            description_separator=", ", max_context_tokens=8192,
            alias="a", description="d", source="s",
        )
        assert m.name == "M"
        assert m.id == "01"
        assert m.original_api_id == "orig"
        assert m.description_separator == ", "
        assert m.max_context_tokens == 8192

    def test_inherits_base_model(self):
        m = BlabladorModel(name="M", alias="a", description="d", source="s")
        assert isinstance(m, BaseModel)

    def test_default_max_context_tokens(self):
        m = BlabladorModel(name="M")
        assert m.max_context_tokens == 32768


# ---------------------------------------------------------------------------
# BlabladorModel.display_string
# ---------------------------------------------------------------------------
class TestDisplayString:
    def test_basic(self):
        m = BlabladorModel(id="01", name="GPT-OSS-120b")
        assert m.display_string == "01 - GPT-OSS-120b"

    def test_with_alias(self):
        m = BlabladorModel(id="01", name="GPT-OSS-120b", alias="fast")
        assert m.display_string == "01 - GPT-OSS-120b - (fast)"

    def test_with_description(self):
        m = BlabladorModel(id="01", name="GPT-OSS-120b", description="Open model")
        assert m.display_string == "01 - GPT-OSS-120b - - Open model"

    def test_with_alias_and_description(self):
        m = BlabladorModel(id="01", name="GPT-OSS-120b", alias="fast", description="Open model")
        assert m.display_string == "01 - GPT-OSS-120b - (fast) - - Open model"

    def test_empty_id(self):
        m = BlabladorModel(id="", name="alias-fast")
        assert m.display_string == " - alias-fast"

    def test_no_alias(self):
        m = BlabladorModel(id="02", name="Qwen3", description="desc")
        assert m.display_string == "02 - Qwen3 - - desc"


# ---------------------------------------------------------------------------
# BlabladorModel.api_id
# ---------------------------------------------------------------------------
class TestApiId:
    def test_original_api_id_preferred(self):
        m = BlabladorModel(id="01", name="M", original_api_id="01 - M - desc")
        assert m.api_id == "01 - M - desc"

    def test_id_with_spaces(self):
        m = BlabladorModel(id="999 - Mis", name="Mis")
        assert m.api_id == "999 - Mis"

    def test_id_with_commas(self):
        m = BlabladorModel(id="1,2", name="M")
        assert m.api_id == "1,2"

    def test_short_id_with_description(self):
        m = BlabladorModel(id="01", name="GPT-OSS-120b", description="Open model")
        assert m.api_id == "01 - GPT-OSS-120b - Open model"

    def test_short_id_with_custom_separator(self):
        m = BlabladorModel(id="02", name="Qwen3", description="desc", description_separator=", ")
        assert m.api_id == "02 - Qwen3, desc"

    def test_short_id_no_description(self):
        m = BlabladorModel(id="01", name="GPT-OSS-120b")
        assert m.api_id == "01 - GPT-OSS-120b"

    def test_empty_id_uses_name(self):
        m = BlabladorModel(id="", name="alias-fast")
        assert m.api_id == "alias-fast"


# ---------------------------------------------------------------------------
# KNOWN_MODELS
# ---------------------------------------------------------------------------
class TestKnownModels:
    def test_non_empty(self):
        assert len(KNOWN_MODELS) > 0

    def test_gpt_oss_120b_exists(self):
        names = [m.name for m in KNOWN_MODELS]
        assert "GPT-OSS-120b" in names

    def test_alias_code_exists(self):
        names = [m.name for m in KNOWN_MODELS]
        assert "alias-code" in names

    def test_alias_fast_exists(self):
        names = [m.name for m in KNOWN_MODELS]
        assert "alias-fast" in names

    def test_alias_large_exists(self):
        names = [m.name for m in KNOWN_MODELS]
        assert "alias-large" in names

    def test_all_are_blablador_model(self):
        for m in KNOWN_MODELS:
            assert isinstance(m, BlabladorModel)

    def test_mis_model_uses_space_in_id(self):
        mis = [m for m in KNOWN_MODELS if m.name == "Mis"]
        assert len(mis) == 1
        assert " " in mis[0].id

    def test_models_with_zero_context_tokens(self):
        assert all(m.max_context_tokens > 0 for m in KNOWN_MODELS)

    def test_alias_models_have_descriptions(self):
        alias_models = [m for m in KNOWN_MODELS if m.alias]
        for m in alias_models:
            assert m.description, f"Alias model {m.name} missing description"


# ---------------------------------------------------------------------------
# get_model_by_name
# ---------------------------------------------------------------------------
class TestGetModelByName:
    def test_find_by_name(self):
        m = get_model_by_name("GPT-OSS-120b")
        assert m is not None
        assert m.name == "GPT-OSS-120b"

    def test_find_by_id(self):
        m = get_model_by_name("15")
        assert m is not None
        assert m.name == "Apertus-8B-Instruct-2509"

    def test_find_by_alias(self):
        m = get_model_by_name("fast")
        assert m is not None
        assert m.name == "alias-fast"

    def test_not_found(self):
        m = get_model_by_name("nonexistent-model")
        assert m is None

    def test_find_apertus_by_name(self):
        m = get_model_by_name("Apertus-8B-Instruct-2509")
        assert m is not None
        assert m.id == "15"

    def test_find_eve_instruct_by_name(self):
        m = get_model_by_name("EVE-Instruct")
        assert m is not None
        assert m.max_context_tokens == 32768


# ---------------------------------------------------------------------------
# _build_hf_search_patterns
# ---------------------------------------------------------------------------
class TestBuildHfSearchPatterns:
    def test_basic(self):
        patterns = _build_hf_search_patterns("LLaMA-3.1-8B")
        assert "LLaMA-3.1-8B" in patterns
        assert "microsoft/LLaMA-3.1-8B" in patterns
        assert "meta-llama/LLaMA-3.1-8B" in patterns
        assert "mistralai/LLaMA-3.1-8B" in patterns
        assert "Qwen/LLaMA-3.1-8B" in patterns

    def test_lowercased(self):
        patterns = _build_hf_search_patterns("MyModel")
        assert "mymodel" in patterns

    def test_slash_replaced(self):
        patterns = _build_hf_search_patterns("org/model-name")
        assert "org--model-name" in patterns

    def test_spaces_replaced(self):
        patterns = _build_hf_search_patterns("My Model Name")
        assert "my-model-name" in patterns

    def test_count(self):
        patterns = _build_hf_search_patterns("test")
        assert len(patterns) == 7


# ---------------------------------------------------------------------------
# _build_hf_search_terms
# ---------------------------------------------------------------------------
class TestBuildHfSearchTerms:
    def test_basic(self):
        terms = _build_hf_search_terms("LLaMA-3.1-8B")
        assert "LLaMA-3.1-8B" in terms
        assert "llama-3.1-8b" in terms

    def test_space_replacement(self):
        terms = _build_hf_search_terms("My Model")
        assert "my model" in terms

    def test_count(self):
        terms = _build_hf_search_terms("test")
        assert len(terms) == 3

    def test_slash_replaced(self):
        terms = _build_hf_search_terms("org/model")
        assert "org--model" in terms


# ---------------------------------------------------------------------------
# _fetch_hf_model_details
# ---------------------------------------------------------------------------
class TestFetchHfModelDetails:
    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        response_data = {"id": "meta-llama/Llama-3.1-8B"}
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        result = _fetch_hf_model_details("meta-llama/Llama-3.1-8B")
        assert result == response_data

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_404_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs={}, fp=None
        )
        result = _fetch_hf_model_details("nonexistent")
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_500_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=500, msg="Internal Server Error", hdrs={}, fp=None
        )
        result = _fetch_hf_model_details("model")
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_generic_exception_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network error")
        result = _fetch_hf_model_details("model")
        assert result is None


# ---------------------------------------------------------------------------
# _search_hf_model_candidates
# ---------------------------------------------------------------------------
class TestSearchHfModelCandidates:
    @patch("hellmholtz.providers.blablador_config._fetch_first_hf_candidate")
    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_success_with_candidates(self, mock_urlopen, mock_fetch):
        hits = [{"id": "model-a"}, {"id": "model-b"}]
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(hits).encode("utf-8")
        mock_urlopen.return_value = mock_resp
        mock_fetch.return_value = {"id": "model-a", "config": {}}

        result = _search_hf_model_candidates("llama")
        assert result is not None
        mock_fetch.assert_called_once()

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_empty_list(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([]).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        result = _search_hf_model_candidates("nonexistent")
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_non_list_response(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"error": "bad"}).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        result = _search_hf_model_candidates("model")
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_non_200_returns_none(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 403
        mock_resp.read.return_value = b""
        mock_urlopen.return_value = mock_resp

        result = _search_hf_model_candidates("model")
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_exception_returns_none(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("connection failed")
        result = _search_hf_model_candidates("model")
        assert result is None

    @patch("hellmholtz.providers.blablador_config._fetch_first_hf_candidate")
    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_filters_non_dict_hits(self, mock_urlopen, mock_fetch):
        hits = [{"id": "model-a"}, "not-a-dict", {"no-id": True}]
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(hits).encode("utf-8")
        mock_urlopen.return_value = mock_resp
        mock_fetch.return_value = None

        _search_hf_model_candidates("test")
        mock_fetch.assert_called_once_with(["model-a"])


# ---------------------------------------------------------------------------
# _fetch_first_hf_candidate
# ---------------------------------------------------------------------------
class TestFetchFirstHfCandidate:
    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_valid_candidate(self, mock_urlopen):
        data = {"id": "model-a", "config": {"max_seq_len": 4096}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        result = _fetch_first_hf_candidate(["model-a"])
        assert result == data

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_non_dict_ignored(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps("not-a-dict").encode("utf-8")
        mock_urlopen.return_value = mock_resp

        result = _fetch_first_hf_candidate(["bad-model"])
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_exception_skips_candidate(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("network error")

        result = _fetch_first_hf_candidate(["bad", "also-bad"])
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_non_200_skips_candidate(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 404
        mock_urlopen.return_value = mock_resp

        result = _fetch_first_hf_candidate(["missing-model"])
        assert result is None

    @patch("hellmholtz.providers.blablador_config.urllib.request.urlopen")
    def test_fallback_to_second_candidate(self, mock_urlopen):
        call_count = [0]

        def side_effect(url, timeout=5):
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            call_count[0] += 1
            if call_count[0] == 1:
                mock_resp.status = 404
                mock_resp.read.return_value = b""
            else:
                mock_resp.status = 200
                mock_resp.read.return_value = json.dumps({"id": "second"}).encode("utf-8")
            return mock_resp

        mock_urlopen.side_effect = side_effect

        result = _fetch_first_hf_candidate(["first", "second"])
        assert result is not None
        assert result["id"] == "second"


# ---------------------------------------------------------------------------
# _fetch_huggingface_model_info
# ---------------------------------------------------------------------------
class TestFetchHuggingfaceModelInfo:
    @patch("hellmholtz.providers.blablador_config._search_hf_model_candidates")
    @patch("hellmholtz.providers.blablador_config._fetch_hf_model_details")
    def test_direct_fetch_succeeds(self, mock_details, mock_search):
        mock_details.return_value = {"id": "meta-llama/Llama-3.1-8B"}
        result = _fetch_huggingface_model_info("Llama-3.1-8B")
        assert result == {"id": "meta-llama/Llama-3.1-8B"}
        mock_search.assert_not_called()

    @patch("hellmholtz.providers.blablador_config._search_hf_model_candidates")
    @patch("hellmholtz.providers.blablador_config._fetch_hf_model_details")
    def test_falls_back_to_search(self, mock_details, mock_search):
        mock_details.return_value = None
        mock_search.return_value = {"id": "found-model"}
        result = _fetch_huggingface_model_info("unknown")
        assert result == {"id": "found-model"}

    @patch("hellmholtz.providers.blablador_config._search_hf_model_candidates")
    @patch("hellmholtz.providers.blablador_config._fetch_hf_model_details")
    def test_all_fail_returns_none(self, mock_details, mock_search):
        mock_details.return_value = None
        mock_search.return_value = None
        result = _fetch_huggingface_model_info("nonexistent")
        assert result is None

    def test_strips_whitespace(self):
        with patch("hellmholtz.providers.blablador_config._fetch_hf_model_details") as mock_details:
            mock_details.return_value = None
            _fetch_huggingface_model_info("  model  ")
            mock_details.assert_any_call("  model  ".strip())


# ---------------------------------------------------------------------------
# _get_model_family_context_length
# ---------------------------------------------------------------------------
class TestGetModelFamilyContextLength:
    def test_llama32(self):
        assert _get_model_family_context_length("llama-3.2-8b") == 131072

    def test_llama32_no_dash(self):
        assert _get_model_family_context_length("llama3.2") == 131072

    def test_llama31(self):
        assert _get_model_family_context_length("llama-3.1-8b") == 131072

    def test_llama31_no_dash(self):
        assert _get_model_family_context_length("llama3.1") == 131072

    def test_llama3(self):
        assert _get_model_family_context_length("llama-3") == 8192

    def test_llama3_no_dash(self):
        assert _get_model_family_context_length("llama3") == 8192

    def test_mistral(self):
        assert _get_model_family_context_length("mistral-7b") == 32768

    def test_qwen3(self):
        assert _get_model_family_context_length("qwen3-72b") == 131072

    def test_qwen3_with_dash(self):
        assert _get_model_family_context_length("qwen-3-72b") == 131072

    def test_phi4(self):
        assert _get_model_family_context_length("phi-4") == 16384

    def test_phi4_no_dash(self):
        assert _get_model_family_context_length("phi4") == 16384

    def test_phi3(self):
        assert _get_model_family_context_length("phi-3") == 4096

    def test_phi3_no_dash(self):
        assert _get_model_family_context_length("phi3") == 4096

    def test_gpt4(self):
        assert _get_model_family_context_length("gpt-4") == 128000

    def test_claude3(self):
        assert _get_model_family_context_length("claude-3-opus") == 200000

    def test_unknown_returns_none(self):
        assert _get_model_family_context_length("some-random-model") is None


# ---------------------------------------------------------------------------
# _extract_context_length_from_hf_model
# ---------------------------------------------------------------------------
class TestExtractContextLengthFromHfModel:
    def test_max_position_embeddings(self):
        info = {"config": {"max_position_embeddings": 8192}}
        assert _extract_context_length_from_hf_model(info) == 8192

    def test_max_seq_len(self):
        info = {"config": {"max_seq_len": 4096}}
        assert _extract_context_length_from_hf_model(info) == 4096

    def test_max_seq_length(self):
        info = {"config": {"max_seq_length": 16384}}
        assert _extract_context_length_from_hf_model(info) == 16384

    def test_seq_length(self):
        info = {"config": {"seq_length": 2048}}
        assert _extract_context_length_from_hf_model(info) == 2048

    def test_context_length(self):
        info = {"config": {"context_length": 32768}}
        assert _extract_context_length_from_hf_model(info) == 32768

    def test_n_positions(self):
        info = {"config": {"n_positions": 1024}}
        assert _extract_context_length_from_hf_model(info) == 1024

    def test_model_max_length(self):
        info = {"config": {"model_max_length": 65536}}
        assert _extract_context_length_from_hf_model(info) == 65536

    def test_out_of_range_returns_none(self):
        info = {"config": {"max_position_embeddings": 500}}
        assert _extract_context_length_from_hf_model(info) is None

    def test_too_large_returns_none(self):
        info = {"config": {"max_position_embeddings": 5000000}}
        assert _extract_context_length_from_hf_model(info) is None

    def test_non_int_value_skipped(self):
        info = {"config": {"max_position_embeddings": "8192"}}
        assert _extract_context_length_from_hf_model(info) is None

    def test_card_data_fallback(self):
        info = {"cardData": {"max_position_embeddings": 16384}}
        assert _extract_context_length_from_hf_model(info) == 16384

    def test_tags_with_context_length(self):
        info = {"tags": ["context-length-131072", "pytorch"]}
        assert _extract_context_length_from_hf_model(info) == 131072

    def test_tags_with_underscore(self):
        info = {"tags": ["context_length_8192"]}
        assert _extract_context_length_from_hf_model(info) == 8192

    def test_tags_no_match_falls_to_family(self):
        info = {"tags": ["pytorch", "transformers"], "id": "meta-llama/Llama-3.1-8B"}
        assert _extract_context_length_from_hf_model(info) == 131072

    def test_model_family_fallback(self):
        info = {"id": "gpt-4-some-model"}
        assert _extract_context_length_from_hf_model(info) == 128000

    def test_empty_info_returns_none(self):
        assert _extract_context_length_from_hf_model({}) is None

    def test_config_key_in_range_boundary_low(self):
        info = {"config": {"max_position_embeddings": 1000}}
        assert _extract_context_length_from_hf_model(info) == 1000

    def test_config_key_in_range_boundary_high(self):
        info = {"config": {"max_position_embeddings": 2000000}}
        assert _extract_context_length_from_hf_model(info) == 2000000

    def test_config_key_below_range(self):
        info = {"config": {"max_position_embeddings": 999}}
        assert _extract_context_length_from_hf_model(info) is None

    def test_config_key_above_range(self):
        info = {"config": {"max_position_embeddings": 2000001}}
        assert _extract_context_length_from_hf_model(info) is None


# ---------------------------------------------------------------------------
# _get_online_token_limit
# ---------------------------------------------------------------------------
class TestGetOnlineTokenLimit:
    @patch("hellmholtz.providers.blablador_config._fetch_huggingface_model_info")
    def test_cache_hit(self, mock_fetch):
        _ONLINE_TOKEN_CACHE["huggingface:model"] = 12345
        result = _get_online_token_limit("model")
        assert result == 12345
        mock_fetch.assert_not_called()

    @patch("hellmholtz.providers.blablador_config._extract_context_length_from_hf_model")
    @patch("hellmholtz.providers.blablador_config._fetch_huggingface_model_info")
    def test_cache_miss_fetches(self, mock_fetch, mock_extract):
        mock_fetch.return_value = {"id": "model"}
        mock_extract.return_value = 8192
        result = _get_online_token_limit("model")
        assert result == 8192
        assert _ONLINE_TOKEN_CACHE["huggingface:model"] == 8192

    @patch("hellmholtz.providers.blablador_config._fetch_huggingface_model_info")
    def test_not_found_cached_as_none(self, mock_fetch):
        mock_fetch.return_value = None
        result = _get_online_token_limit("unknown")
        assert result is None
        assert _ONLINE_TOKEN_CACHE.get("huggingface:unknown") is None

    @patch("hellmholtz.providers.blablador_config._fetch_huggingface_model_info")
    def test_exception_cached_as_none(self, mock_fetch):
        mock_fetch.side_effect = Exception("error")
        result = _get_online_token_limit("badmodel")
        assert result is None
        assert _ONLINE_TOKEN_CACHE.get("huggingface:badmodel") is None

    @patch("hellmholtz.providers.blablador_config._fetch_huggingface_model_info")
    def test_non_hf_provider_skips(self, mock_fetch):
        result = _get_online_token_limit("model", provider="openai")
        assert result is None
        mock_fetch.assert_not_called()

    @patch("hellmholtz.providers.blablador_config._extract_context_length_from_hf_model")
    @patch("hellmholtz.providers.blablador_config._fetch_huggingface_model_info")
    def test_extract_returns_none(self, mock_fetch, mock_extract):
        mock_fetch.return_value = {"id": "model"}
        mock_extract.return_value = None
        result = _get_online_token_limit("model")
        assert result is None
        assert _ONLINE_TOKEN_CACHE.get("huggingface:model") is None


# ---------------------------------------------------------------------------
# _get_provider_token_limit
# ---------------------------------------------------------------------------
class TestGetProviderTokenLimit:
    def test_openai(self):
        with patch("hellmholtz.providers.blablador_config._get_openai_token_limit", return_value=128000) as mock:
            result = _get_provider_token_limit("openai", "gpt-4o")
            mock.assert_called_once_with("gpt-4o")
            assert result == 128000

    def test_anthropic(self):
        with patch("hellmholtz.providers.blablador_config._get_anthropic_token_limit", return_value=200000) as mock:
            result = _get_provider_token_limit("anthropic", "claude-3")
            mock.assert_called_once_with("claude-3")
            assert result == 200000

    def test_google(self):
        with patch("hellmholtz.providers.blablador_config._get_google_token_limit", return_value=1000000) as mock:
            result = _get_provider_token_limit("google", "gemini-pro")
            mock.assert_called_once_with("gemini-pro")
            assert result == 1000000

    def test_ollama(self):
        with patch("hellmholtz.providers.blablador_config._get_ollama_token_limit", return_value=4096) as mock:
            result = _get_provider_token_limit("ollama", "llama3")
            mock.assert_called_once_with("llama3")
            assert result == 4096

    def test_blablador_known(self):
        result = _get_provider_token_limit("blablador", "GPT-OSS-120b")
        assert result == 131072

    def test_blablador_unknown_tries_online(self):
        with patch("hellmholtz.providers.blablador_config._get_online_token_limit", return_value=16384) as mock_online:
            result = _get_provider_token_limit("blablador", "unknown-model")
            mock_online.assert_called_once_with("unknown-model", "huggingface")
            assert result == 16384

    def test_blablador_unknown_online_returns_none(self):
        with patch("hellmholtz.providers.blablador_config._get_online_token_limit", return_value=None):
            result = _get_provider_token_limit("blablador", "unknown-model")
            assert result == DEFAULT_TOKEN_LIMIT

    def test_unknown_provider_uses_blablador(self):
        result = _get_provider_token_limit("unknown-provider", "GPT-OSS-120b")
        assert result == 131072

    def test_unknown_provider_tries_online(self):
        with patch("hellmholtz.providers.blablador_config._get_online_token_limit", return_value=16384) as mock_online:
            result = _get_provider_token_limit("custom", "unknown-model")
            mock_online.assert_called_once_with("unknown-model", "huggingface")
            assert result == 16384

    def test_unknown_provider_online_returns_none(self):
        with patch("hellmholtz.providers.blablador_config._get_online_token_limit", return_value=None):
            result = _get_provider_token_limit("custom", "unknown-model")
            assert result == DEFAULT_TOKEN_LIMIT


# ---------------------------------------------------------------------------
# get_token_limit
# ---------------------------------------------------------------------------
class TestGetTokenLimit:
    def test_with_provider_prefix(self):
        with patch("hellmholtz.providers.blablador_config._get_provider_token_limit", return_value=8192) as mock:
            result = get_token_limit("openai:gpt-4")
            mock.assert_called_once_with("openai", "gpt-4")
            assert result == 8192

    def test_without_provider_defaults_to_blablador(self):
        with patch("hellmholtz.providers.blablador_config._get_provider_token_limit", return_value=131072) as mock:
            result = get_token_limit("GPT-OSS-120b")
            mock.assert_called_once_with("blablador", "GPT-OSS-120b")
            assert result == 131072

    def test_colon_in_model_name(self):
        with patch("hellmholtz.providers.blablador_config._get_provider_token_limit", return_value=4096) as mock:
            result = get_token_limit("ollama:llama3:8b")
            mock.assert_called_once_with("ollama", "llama3:8b")
            assert result == 4096

    def test_provider_is_lowercased(self):
        with patch("hellmholtz.providers.blablador_config._get_provider_token_limit", return_value=1000) as mock:
            result = get_token_limit("OpenAI:GPT-4o")
            mock.assert_called_once_with("openai", "GPT-4o")


# ---------------------------------------------------------------------------
# _get_openai_token_limit
# ---------------------------------------------------------------------------
class TestGetOpenaiTokenLimit:
    def test_gpt4o(self):
        assert _get_openai_token_limit("gpt-4o") == 128000

    def test_gpt4o_mini(self):
        assert _get_openai_token_limit("gpt-4o-mini") == 128000

    def test_gpt4_turbo(self):
        assert _get_openai_token_limit("gpt-4-turbo") == 128000

    def test_gpt4(self):
        assert _get_openai_token_limit("gpt-4") == 8192

    def test_gpt35_turbo(self):
        assert _get_openai_token_limit("gpt-3.5-turbo") == 16384

    def test_text_davinci_003(self):
        assert _get_openai_token_limit("text-davinci-003") == 4096

    def test_text_embedding_ada_002(self):
        assert _get_openai_token_limit("text-embedding-ada-002") == 8192

    def test_unknown(self):
        assert _get_openai_token_limit("o1-preview") == 4096

    def test_case_insensitive(self):
        assert _get_openai_token_limit("GPT-4o") == 128000


# ---------------------------------------------------------------------------
# _get_anthropic_token_limit
# ---------------------------------------------------------------------------
class TestGetAnthropicTokenLimit:
    def test_claude3_opus(self):
        assert _get_anthropic_token_limit("claude-3-opus-20240229") == 200000

    def test_claude3_sonnet(self):
        assert _get_anthropic_token_limit("claude-3-sonnet-20240229") == 200000

    def test_claude3_haiku(self):
        assert _get_anthropic_token_limit("claude-3-haiku-20240307") == 200000

    def test_claude3_generic(self):
        assert _get_anthropic_token_limit("claude-3") == 200000

    def test_claude2(self):
        assert _get_anthropic_token_limit("claude-2.1") == 100000

    def test_claude2_base(self):
        assert _get_anthropic_token_limit("claude-2") == 100000

    def test_unknown(self):
        assert _get_anthropic_token_limit("claude-next") == 100000

    def test_case_insensitive(self):
        assert _get_anthropic_token_limit("Claude-3-Opus") == 200000


# ---------------------------------------------------------------------------
# _get_google_token_limit
# ---------------------------------------------------------------------------
class TestGetGoogleTokenLimit:
    def test_gemini_pro(self):
        assert _get_google_token_limit("gemini-pro") == 1000000

    def test_gemini_flash(self):
        assert _get_google_token_limit("gemini-1.5-flash") == 1000000

    def test_gemini_generic(self):
        assert _get_google_token_limit("gemini") == 1000000

    def test_unknown(self):
        assert _get_google_token_limit("bard") == 32768

    def test_case_insensitive(self):
        assert _get_google_token_limit("Gemini-Pro") == 1000000


# ---------------------------------------------------------------------------
# _get_ollama_token_limit
# ---------------------------------------------------------------------------
class TestGetOllamaTokenLimit:
    def test_llama32(self):
        assert _get_ollama_token_limit("llama3.2") == 131072

    def test_llama32_with_tag(self):
        assert _get_ollama_token_limit("llama3.2:3b") == 131072

    def test_llama31(self):
        assert _get_ollama_token_limit("llama3.1") == 131072

    def test_llama31_with_tag(self):
        assert _get_ollama_token_limit("llama3.1:70b") == 131072

    def test_llama3(self):
        assert _get_ollama_token_limit("llama3") == 8192

    def test_mistral(self):
        assert _get_ollama_token_limit("mistral") == 32768

    def test_codellama(self):
        assert _get_ollama_token_limit("codellama") == 16384

    def test_phi(self):
        assert _get_ollama_token_limit("phi") == 4096

    def test_unknown(self):
        assert _get_ollama_token_limit("some-model") == 4096

    def test_case_insensitive(self):
        assert _get_ollama_token_limit("Llama3.2") == 131072


# ---------------------------------------------------------------------------
# _get_blablador_token_limit
# ---------------------------------------------------------------------------
class TestGetBlabladorTokenLimit:
    def test_known_model(self):
        result = _get_blablador_token_limit("GPT-OSS-120b")
        assert result == 131072

    def test_known_model_by_alias(self):
        result = _get_blablador_token_limit("fast")
        assert result == 32768

    def test_unknown_model(self):
        result = _get_blablador_token_limit("totally-fake-model")
        assert result == DEFAULT_TOKEN_LIMIT

    def test_known_model_by_id(self):
        result = _get_blablador_token_limit("15")
        assert result == 32768


# ---------------------------------------------------------------------------
# get_all_provider_token_limits
# ---------------------------------------------------------------------------
class TestGetAllProviderTokenLimits:
    def test_all_providers_present(self):
        limits = get_all_provider_token_limits()
        assert "openai" in limits
        assert "anthropic" in limits
        assert "google" in limits
        assert "ollama" in limits
        assert "blablador" in limits

    def test_blablador_models_populated(self):
        limits = get_all_provider_token_limits()
        assert "GPT-OSS-120b" in limits["blablador"]
        assert limits["blablador"]["GPT-OSS-120b"] == 131072

    def test_blablador_aliases_populated(self):
        limits = get_all_provider_token_limits()
        assert "fast" in limits["blablador"]

    def test_online_not_included_by_default(self):
        _ONLINE_TOKEN_CACHE["huggingface:test-model"] = 16384
        limits = get_all_provider_token_limits()
        assert "online" not in limits

    def test_online_included_when_flagged(self):
        _ONLINE_TOKEN_CACHE["huggingface:test-model"] = 16384
        limits = get_all_provider_token_limits(include_online=True)
        assert "online" in limits
        assert limits["online"]["huggingface:test-model"] == 16384

    def test_online_none_values_excluded(self):
        _ONLINE_TOKEN_CACHE["huggingface:bad"] = None
        _ONLINE_TOKEN_CACHE["huggingface:good"] = 8192
        limits = get_all_provider_token_limits(include_online=True)
        assert "huggingface:bad" not in limits["online"]
        assert limits["online"]["huggingface:good"] == 8192

    def test_openai_models_count(self):
        limits = get_all_provider_token_limits()
        assert len(limits["openai"]) == 6

    def test_anthropic_models_count(self):
        limits = get_all_provider_token_limits()
        assert len(limits["anthropic"]) == 6

    def test_google_models_count(self):
        limits = get_all_provider_token_limits()
        assert len(limits["google"]) == 4

    def test_ollama_models_count(self):
        limits = get_all_provider_token_limits()
        assert len(limits["ollama"]) == 12


# ---------------------------------------------------------------------------
# clear_online_token_cache
# ---------------------------------------------------------------------------
class TestClearOnlineTokenCache:
    def test_clears_cache(self):
        _ONLINE_TOKEN_CACHE["test"] = 123
        _ONLINE_TOKEN_CACHE["other"] = 456
        clear_online_token_cache()
        assert len(_ONLINE_TOKEN_CACHE) == 0

    def test_empty_cache_stays_empty(self):
        clear_online_token_cache()
        assert len(_ONLINE_TOKEN_CACHE) == 0

    def test_cache_empty_after_clear(self):
        _ONLINE_TOKEN_CACHE["key"] = 100
        clear_online_token_cache()
        assert "key" not in _ONLINE_TOKEN_CACHE
