"""
Tests for configuration functionality.

This module contains comprehensive tests for the configuration system,
including environment variable parsing, default values, and settings validation.
"""

import os
import pytest
from unittest.mock import patch
from typing import List, Optional

from hellmholtz.core.config import get_settings, Settings


class TestSettings:
    """Test suite for Settings dataclass and configuration."""

    def test_settings_default_values(self) -> None:
        """Test default values when no environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()

            assert settings.default_models == []
            assert settings.timeout_seconds == 30.0
            assert settings.blablador_api_key is None
            assert settings.blablador_base_url is None

    def test_settings_env_override_default_models(self) -> None:
        """Test overriding default_models via environment variable."""
        env_vars = {
            "AISUITE_DEFAULT_MODELS": "openai:gpt-4o,anthropic:claude-3-sonnet-20240229,ollama:llama3"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()
            expected_models = [
                "openai:gpt-4o",
                "anthropic:claude-3-sonnet-20240229",
                "ollama:llama3"
            ]
            assert settings.default_models == expected_models

    def test_settings_env_override_blablador_config(self) -> None:
        """Test overriding Blablador configuration via environment variables."""
        env_vars = {
            "BLABLADOR_API_KEY": "test-api-key-123",
            "BLABLADOR_API_BASE": "https://api.blablador.example.com/v1"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()
            assert settings.blablador_api_key == "test-api-key-123"
            assert settings.blablador_base_url == "https://api.blablador.example.com/v1"

    def test_settings_env_override_timeout(self) -> None:
        """Test overriding timeout via environment variable."""
        env_vars = {
            "HELMHOLTZ_TIMEOUT_SECONDS": "120.5"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()
            assert settings.timeout_seconds == 120.5

    def test_settings_env_override_all_variables(self) -> None:
        """Test overriding all settings via environment variables."""
        env_vars = {
            "AISUITE_DEFAULT_MODELS": "openai:gpt-4o,blablador:test-model",
            "HELMHOLTZ_TIMEOUT_SECONDS": "60.0",
            "BLABLADOR_API_KEY": "secret-key",
            "BLABLADOR_API_BASE": "https://custom.api.com"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()

            assert settings.default_models == ["openai:gpt-4o", "blablador:test-model"]
            assert settings.timeout_seconds == 60.0
            assert settings.blablador_api_key == "secret-key"
            assert settings.blablador_base_url == "https://custom.api.com"

    def test_settings_empty_env_values(self) -> None:
        """Test behavior with empty environment variable values."""
        env_vars = {
            "AISUITE_DEFAULT_MODELS": "",
            "BLABLADOR_API_KEY": "",
            "BLABLADOR_API_BASE": ""
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()

            assert settings.default_models == []
            assert settings.blablador_api_key == ""
            assert settings.blablador_base_url == ""

    def test_settings_whitespace_handling(self) -> None:
        """Test that whitespace in environment variables is handled properly."""
        env_vars = {
            "AISUITE_DEFAULT_MODELS": " openai:gpt-4o , anthropic:claude-3-sonnet-20240229 "
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()
            # Implementation strips whitespace from individual models
            expected = ["openai:gpt-4o", "anthropic:claude-3-sonnet-20240229"]
            assert settings.default_models == expected

    def test_settings_numeric_timeout_conversion(self) -> None:
        """Test that timeout values are properly converted to float."""
        test_cases = [
            ("30", 30.0),
            ("30.5", 30.5),
            ("0", 0.0),
            ("300.99", 300.99)
        ]

        for env_value, expected in test_cases:
            env_vars = {"HELMHOLTZ_TIMEOUT_SECONDS": env_value}
            with patch.dict(os.environ, env_vars, clear=True):
                settings = get_settings()
                assert settings.timeout_seconds == expected

    def test_settings_singleton_behavior(self) -> None:
        """Test that get_settings returns different instances (no caching)."""
        with patch.dict(os.environ, {}, clear=True):
            settings1 = get_settings()
            settings2 = get_settings()

            # Should be different instances since no caching is used
            assert settings1 is not settings2
            # But they should have the same values
            assert settings1.default_models == settings2.default_models

    def test_settings_mutability(self) -> None:
        """Test that Settings instances are mutable."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()

            # Should be able to modify attributes
            original_timeout = settings.timeout_seconds
            settings.timeout_seconds = 60.0
            assert settings.timeout_seconds == 60.0
            assert settings.timeout_seconds != original_timeout

    @pytest.mark.parametrize("env_var,value,expected", [
        ("HELMHOLTZ_TIMEOUT_SECONDS", "30.0", 30.0),
        ("HELMHOLTZ_TIMEOUT_SECONDS", "60", 60.0),
    ])
    def test_settings_timeout_values(self, env_var: str, value: str, expected: float) -> None:
        """Test various timeout value configurations."""
        env_vars = {env_var: value}
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()

            assert settings.timeout_seconds == expected

    def test_settings_complex_model_list(self) -> None:
        """Test parsing complex model lists with various formats."""
        env_vars = {
            "AISUITE_DEFAULT_MODELS": "openai:gpt-4o,blablador:Ministral-3-14B-Instruct-2512,anthropic:claude-3-5-sonnet-20241022,ollama:llama3.2:3b"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()

            expected_models = [
                "openai:gpt-4o",
                "blablador:Ministral-3-14B-Instruct-2512",
                "anthropic:claude-3-5-sonnet-20241022",
                "ollama:llama3.2:3b"
            ]
            assert settings.default_models == expected_models
