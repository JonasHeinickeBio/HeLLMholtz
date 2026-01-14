"""
Tests for token limit functionality.

This module contains tests for the token limit metadata and helper functions
in the blablador_config module.
"""

import pytest

from hellmholtz.providers.blablador_config import (
    KNOWN_MODELS,
    BlabladorModel,
    get_model_by_name,
    get_token_limit,
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
