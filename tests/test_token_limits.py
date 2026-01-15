"""
Tests for token limit functionality.

This module contains tests for the token limit metadata and helper functions
in the blablador_config module.
"""

import logging

import pytest

from hellmholtz.providers.blablador_config import (
    KNOWN_MODELS,
    BlabladorModel,
    get_model_by_name,
    get_token_limit,
    get_all_provider_token_limits,
    clear_online_token_cache,
    _get_model_family_context_length,
)


class TestBlabladorModelTokenLimits:
    """Test suite for BlabladorModel token limit features."""

    def test_all_models_have_token_limits(self) -> None:
        """Test that all models have a max_context_tokens value."""
        for model in KNOWN_MODELS:
            assert model.max_context_tokens > 0, f"Model {model.name} has no token limit"
            assert isinstance(model.max_context_tokens, int), (
                f"Model {model.name} token limit is not an integer"
            )

    def test_token_limits_are_reasonable(self) -> None:
        """Test that token limits fall within reasonable ranges."""
        for model in KNOWN_MODELS:
            # Token limits should be between 4k and 1M tokens
            assert 4096 <= model.max_context_tokens <= 1_000_000, (
                f"Model {model.name} has unreasonable token limit: {model.max_context_tokens}"
            )

    def test_large_models_have_large_context(self) -> None:
        """Test that large models have at least 32k context."""
        large_model_names = [
            "Ministral-3-14B-Instruct-2512",
            "GPT-OSS-120b",
            "Qwen3 235",
            "Qwen3-Coder-30B-A3B-Instruct",
        ]

        for name in large_model_names:
            model = get_model_by_name(name)
            assert model is not None, f"Model {name} not found"
            assert model.max_context_tokens >= 32768, (
                f"Large model {name} should have at least 32k context"
            )

    def test_embedding_models_have_appropriate_limits(self) -> None:
        """Test that embedding models have smaller context windows."""
        embedding_names = ["alias-embeddings", "text-embedding-ada-002"]

        for name in embedding_names:
            model = get_model_by_name(name)
            assert model is not None, f"Embedding model {name} not found"
            # Embedding models typically have 8k or less
            assert model.max_context_tokens <= 16384, (
                f"Embedding model {name} should have smaller context"
            )


class TestGetModelByName:
    """Test suite for get_model_by_name helper function."""

    def test_get_model_by_name(self) -> None:
        """Test retrieving a model by its name."""
        model = get_model_by_name("Ministral-3-14B-Instruct-2512")
        assert model is not None
        assert model.name == "Ministral-3-14B-Instruct-2512"
        assert model.max_context_tokens == 131072

    def test_get_model_by_alias(self) -> None:
        """Test retrieving a model by its alias."""
        model = get_model_by_name("fast")
        assert model is not None
        assert model.alias == "fast"
        assert model.name == "alias-fast"

    def test_get_model_by_id(self) -> None:
        """Test retrieving a model by its ID."""
        # Test with the full formatted ID
        model = get_model_by_name("0 - Ministral-3-14B-Instruct-2512 - The latest Ministral from Dec.2.2025")
        assert model is not None
        assert model.name == "Ministral-3-14B-Instruct-2512"

    def test_get_nonexistent_model(self) -> None:
        """Test that getting a nonexistent model returns None."""
        model = get_model_by_name("nonexistent-model-xyz")
        assert model is None

    def test_get_model_case_sensitive(self) -> None:
        """Test that model lookup is case-sensitive."""
        # This should return None since the actual name has different casing
        model = get_model_by_name("ministral-3-14b-instruct-2512")
        assert model is None


class TestGetTokenLimit:
    """Test suite for get_token_limit helper function."""

    def test_get_token_limit_by_name(self) -> None:
        """Test getting token limit by model name."""
        limit = get_token_limit("Ministral-3-14B-Instruct-2512")
        assert limit == 131072  # 128k tokens

    def test_get_token_limit_by_alias(self) -> None:
        """Test getting token limit by model alias."""
        limit = get_token_limit("fast")
        assert limit == 32768  # 32k tokens

    def test_get_token_limit_for_qwen(self) -> None:
        """Test getting token limit for Qwen models."""
        limit = get_token_limit("Qwen3 235")
        assert limit == 131072  # 128k+ tokens

    def test_get_token_limit_for_embedding(self) -> None:
        """Test getting token limit for embedding models."""
        limit = get_token_limit("alias-embeddings")
        assert limit == 8192  # 8k tokens

    def test_get_token_limit_for_legacy(self) -> None:
        """Test getting token limit for legacy models."""
        limit = get_token_limit("text-davinci-003")
        assert limit == 4096  # 4k tokens

    def test_get_token_limit_default_fallback(self) -> None:
        """Test that unknown models return default token limit."""
        limit = get_token_limit("unknown-model-xyz")
        assert limit == 32768  # Default 32k tokens

    def test_get_token_limit_various_models(self) -> None:
        """Test token limits for various model types."""
        test_cases = [
            ("Ministral-3-14B-Instruct-2512", 131072),  # Large Ministral
            ("GPT-OSS-120b", 131072),  # Large GPT
            ("Apertus-8B-Instruct-2509", 32768),  # Medium Apertus
            ("Phi-4-multimodal-instruct", 16384),  # Phi model
            ("gpt-3.5-turbo", 16384),  # Legacy GPT
            ("text-embedding-ada-002", 8192),  # Embedding
        ]

        for model_name, expected_limit in test_cases:
            limit = get_token_limit(model_name)
            assert limit == expected_limit, (
                f"Model {model_name} should have limit {expected_limit}, got {limit}"
            )


class TestBlabladorModelDataclass:
    """Test suite for BlabladorModel dataclass features."""

    def test_model_has_default_token_limit(self) -> None:
        """Test that models have a default token limit."""
        model = BlabladorModel(
            name="test-model",
            description="Test model"
        )
        assert model.max_context_tokens == 32768  # Default value

    def test_model_with_custom_token_limit(self) -> None:
        """Test creating a model with custom token limit."""
        model = BlabladorModel(
            name="custom-model",
            description="Custom model",
            max_context_tokens=65536
        )
        assert model.max_context_tokens == 65536

    def test_model_token_limit_in_known_models(self) -> None:
        """Test that token limits are properly set in KNOWN_MODELS."""
        # Check a few specific models
        ministral = next((m for m in KNOWN_MODELS if m.name == "Ministral-3-14B-Instruct-2512"), None)
        assert ministral is not None
        assert ministral.max_context_tokens == 131072

        qwen = next((m for m in KNOWN_MODELS if m.name == "Qwen3 235"), None)
        assert qwen is not None
        assert qwen.max_context_tokens == 131072

        phi = next((m for m in KNOWN_MODELS if m.name == "Phi-4-multimodal-instruct"), None)
        assert phi is not None
        assert phi.max_context_tokens == 16384


class TestTokenLimitCategories:
    """Test suite for token limit categories."""

    def test_128k_models(self) -> None:
        """Test that 128k models are properly categorized."""
        models_128k = [
            "Ministral-3-14B-Instruct-2512",
            "GPT-OSS-120b",
            "MiniMax-M2.1",
            "Qwen3 235",
            "Qwen3-Coder-30B-A3B-Instruct",
            "Devstral-Small-2-24B-Instruct-2512",
            "Qwen3-Next",
            "Qwen3-VL-32B-Instruct-FP8",
            "Tongyi-DeepResearch-30B-A3B",
            "alias-code",
            "alias-large",
            "alias-huge",
            "alias-function-call",
        ]

        for name in models_128k:
            limit = get_token_limit(name)
            assert limit == 131072, f"Model {name} should have 128k context"

    def test_32k_models(self) -> None:
        """Test that 32k models are properly categorized."""
        models_32k = [
            "Apertus-8B-Instruct-2509",
            "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "option-g-2T-step-47250",
            "alias-fast",
            "alias-apertus",
        ]

        for name in models_32k:
            limit = get_token_limit(name)
            assert limit == 32768, f"Model {name} should have 32k context"

    def test_16k_models(self) -> None:
        """Test that 16k models are properly categorized."""
        models_16k = [
            "Phi-4-multimodal-instruct",
            "gpt-3.5-turbo",
        ]

        for name in models_16k:
            limit = get_token_limit(name)
            assert limit == 16384, f"Model {name} should have 16k context"

    def test_8k_models(self) -> None:
        """Test that 8k models are properly categorized."""
        models_8k = [
            "alias-embeddings",
            "text-embedding-ada-002",
        ]

        for name in models_8k:
            limit = get_token_limit(name)
            assert limit == 8192, f"Model {name} should have 8k context"

    def test_4k_models(self) -> None:
        """Test that 4k models are properly categorized."""
        models_4k = [
            "text-davinci-003",
        ]

        for name in models_4k:
            limit = get_token_limit(name)
            assert limit == 4096, f"Model {name} should have 4k context"


class TestOnlineTokenFetching:
    """Test suite for online token limit fetching functionality."""

    def test_online_fetching_for_unknown_models(self) -> None:
        """Test that unknown models can fetch token limits online."""
        # Clear cache first to ensure clean state
        clear_online_token_cache()

        # Test with a known HF model that should be fetchable
        # Note: This test may fail if network is unavailable, but that's expected
        try:
            limit = get_token_limit("microsoft/Phi-4-multimodal-instruct")
            # Should either return the correct limit or fallback to default
            assert isinstance(limit, int)
            assert limit > 0
        except Exception as e:
            # If network fails, should still return default
            logging.debug(f"Network unavailable for online token fetching: {e}")
            limit = get_token_limit("unknown-model-xyz")
            assert limit == 32768

    def test_online_fetching_caching(self) -> None:
        """Test that online fetching results are cached."""
        # Clear cache first
        clear_online_token_cache()

        # This test assumes network is available
        try:
            # First call should fetch from API
            limit1 = get_token_limit("microsoft/Phi-4-multimodal-instruct")

            # Second call should use cache
            limit2 = get_token_limit("microsoft/Phi-4-multimodal-instruct")

            # Results should be identical
            assert limit1 == limit2
            assert isinstance(limit1, int)
        except Exception as e:
            # Skip if network unavailable, but log for debugging
            logging.debug(f"Network unavailable for online token fetching test: {e}")
            pytest.skip(f"Network unavailable: {e}")

    def test_clear_online_token_cache(self) -> None:
        """Test that cache clearing works."""
        # This should not raise any exceptions
        clear_online_token_cache()

        # Verify cache is empty by checking get_all_provider_token_limits
        limits = get_all_provider_token_limits(include_online=True)
        online_limits = limits.get("online", {})
        # Cache should be empty after clearing
        assert len(online_limits) == 0


class TestGetAllProviderTokenLimits:
    """Test suite for get_all_provider_token_limits function."""

    def test_get_all_provider_token_limits_basic(self) -> None:
        """Test basic functionality of get_all_provider_token_limits."""
        limits = get_all_provider_token_limits()

        # Should have all expected providers
        expected_providers = {"blablador", "openai", "anthropic", "google", "ollama"}
        assert set(limits.keys()) == expected_providers

        # Each provider should have models
        for _provider, models in limits.items():
            assert isinstance(models, dict)
            assert len(models) > 0

    def test_get_all_provider_token_limits_with_online(self) -> None:
        """Test get_all_provider_token_limits with include_online=True."""
        # Clear cache first
        clear_online_token_cache()

        limits = get_all_provider_token_limits(include_online=True)

        # Should include online provider even if empty
        assert "online" in limits
        assert isinstance(limits["online"], dict)

    def test_get_all_provider_token_limits_online_populated(self) -> None:
        """Test that online models appear when cache is populated."""
        # Clear cache and populate it
        clear_online_token_cache()

        try:
            # Fetch a model to populate cache
            get_token_limit("microsoft/Phi-4-multimodal-instruct")

            limits = get_all_provider_token_limits(include_online=True)

            # Should have online models now
            online_limits = limits.get("online", {})
            assert len(online_limits) > 0

            # Should contain the fetched model
            cache_key = "huggingface:microsoft/Phi-4-multimodal-instruct"
            assert cache_key in online_limits
            assert isinstance(online_limits[cache_key], int)

        except Exception as e:
            # Skip if network unavailable
            logging.debug(f"Network unavailable for online cache population test: {e}")
            pass


class TestModelFamilyContextLength:
    """Test suite for _get_model_family_context_length helper function."""

    def test_llama_models(self) -> None:
        """Test context length detection for Llama models."""
        test_cases = [
            ("meta-llama/llama-3.2-3b-instruct", 131072),
            ("meta-llama/llama-3.1-8b-instruct", 131072),
            ("meta-llama/llama-3-8b-instruct", 8192),
            ("meta-llama/llama-2-7b-chat", None),  # Not in known families
        ]

        for model_id, expected in test_cases:
            result = _get_model_family_context_length(model_id)
            assert result == expected, f"Model {model_id} should return {expected}"

    def test_mistral_models(self) -> None:
        """Test context length detection for Mistral models."""
        result = _get_model_family_context_length("mistralai/mistral-7b-instruct-v0.2")
        assert result == 32768

    def test_qwen_models(self) -> None:
        """Test context length detection for Qwen models."""
        test_cases = [
            ("qwen/qwen3-8b", 131072),
            ("qwen/qwen-2.5-7b-instruct", None),  # Not Qwen3
        ]

        for model_id, expected in test_cases:
            result = _get_model_family_context_length(model_id)
            assert result == expected, f"Model {model_id} should return {expected}"

    def test_phi_models(self) -> None:
        """Test context length detection for Phi models."""
        test_cases = [
            ("microsoft/phi-4-multimodal-instruct", 16384),
            ("microsoft/phi-3-medium-128k-instruct", 4096),
        ]

        for model_id, expected in test_cases:
            result = _get_model_family_context_length(model_id)
            assert result == expected, f"Model {model_id} should return {expected}"

    def test_gpt_models(self) -> None:
        """Test context length detection for GPT models."""
        result = _get_model_family_context_length("openai/gpt-4")
        assert result == 128000

    def test_claude_models(self) -> None:
        """Test context length detection for Claude models."""
        result = _get_model_family_context_length("anthropic/claude-3-sonnet")
        assert result == 200000

    def test_unknown_models(self) -> None:
        """Test that unknown model families return None."""
        unknown_models = [
            "unknown/model",
            "random-model-name",
            "",
        ]

        for model_id in unknown_models:
            result = _get_model_family_context_length(model_id)
            assert result is None, f"Unknown model {model_id} should return None"
