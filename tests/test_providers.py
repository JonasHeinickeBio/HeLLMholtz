"""
Tests for provider functionality.

This module contains comprehensive tests for the provider modules,
including unit tests for BlabladorProvider and related utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.providers.blablador_provider import BlabladorProvider


class TestBlabladorProvider:
    """Test suite for BlabladorProvider."""

    @pytest.fixture
    def mock_openai_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_init_with_config(self, mock_openai_client: MagicMock) -> None:
        """Test initialization with config parameters."""
        with patch("hellmholtz.providers.blablador_provider.openai.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = mock_openai_client

            config = {"api_key": "test_key", "base_url": "https://api.example.com/v1"}

            provider = BlabladorProvider(**config)

            mock_openai_cls.assert_called_once_with(**config)
            assert provider.client == mock_openai_client

    def test_init_with_env_vars(self, mock_openai_client: MagicMock) -> None:
        """Test initialization using environment variables."""
        with patch("hellmholtz.providers.blablador_provider.openai.OpenAI") as mock_openai_cls:
            mock_openai_cls.return_value = mock_openai_client

            with patch.dict(
                "os.environ",
                {
                    "BLABLADOR_API_KEY": "env_key",
                    "BLABLADOR_API_BASE": "https://env.example.com/v1",
                },
            ):
                config = {}
                BlabladorProvider(**config)

                expected_config = {"api_key": "env_key", "base_url": "https://env.example.com/v1"}
                mock_openai_cls.assert_called_once_with(**expected_config)

    def test_init_missing_api_key(self) -> None:
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="Blablador API key is missing"):
            BlabladorProvider()

    def test_init_missing_base_url(self) -> None:
        """Test initialization fails without base URL."""
        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}, clear=True), pytest.raises(ValueError, match="Blablador Base URL is missing"):
            BlabladorProvider()

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_chat_completions_create_basic(self, mock_openai_cls: MagicMock) -> None:
        """Test basic chat completions create."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()

            messages = [{"role": "user", "content": "Hello"}]
            kwargs = {"temperature": 0.7}

            result = provider.chat_completions_create("test-model", messages, **kwargs)

            assert result == mock_response
            mock_client.chat.completions.create.assert_called_once_with(
                model="test-model", messages=messages, **kwargs
            )

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_chat_completions_create_model_resolution(self, mock_openai_cls: MagicMock) -> None:
        """Test model name resolution in chat completions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            # Mock the KNOWN_MODELS to include a test model
            mock_model = MagicMock()
            mock_model.id = "test"
            mock_model.name = "TestModel"
            mock_model.alias = "test-alias"
            mock_model.api_id = "resolved-model-id"

            with patch("hellmholtz.providers.blablador_provider.KNOWN_MODELS", [mock_model]):
                provider = BlabladorProvider()

                messages = [{"role": "user", "content": "Hello"}]

                # Test with different model identifiers
                test_cases = [
                    ("test", "resolved-model-id"),
                    ("TestModel", "resolved-model-id"),
                    ("test-alias", "resolved-model-id"),
                    ("unknown-model", "unknown-model"),  # Should pass through unchanged
                ]

                for input_model, expected_model in test_cases:
                    mock_client.reset_mock()
                    provider.chat_completions_create(input_model, messages)

                    mock_client.chat.completions.create.assert_called_with(
                        model=expected_model, messages=messages
                    )

    @patch("hellmholtz.providers.blablador_provider.openai.OpenAI")
    def test_chat_completions_create_error_handling(self, mock_openai_cls: MagicMock) -> None:
        """Test error handling in chat completions."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_cls.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()

            messages = [{"role": "user", "content": "Hello"}]

            # Should raise the exception (no special handling in this provider)
            with pytest.raises(Exception, match="API Error"):
                provider.chat_completions_create("test-model", messages)
