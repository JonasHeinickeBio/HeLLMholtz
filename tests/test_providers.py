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
            # Mock available models to include test-model
            provider._get_available_models = MagicMock(return_value=["test-model"])

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
                # Mock available models to include the resolved models
                provider._get_available_models = MagicMock(return_value=["resolved-model-id", "unknown-model"])

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
        from aisuite.provider import LLMError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_cls.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()
            provider._get_available_models = MagicMock(return_value=["test-model"])

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(LLMError, match="An error occurred: API Error"):
                provider.chat_completions_create("test-model", messages)

    @patch("hellmholtz.providers.blablador_provider.openai")
    def test_chat_completions_create_api_connection_error(self, mock_openai: MagicMock) -> None:
        """Test APIConnectionError handling in chat completions."""
        from aisuite.provider import LLMError

        # Create a mock exception class
        class MockAPIConnectionError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.message = message

        mock_openai.APIConnectionError = MockAPIConnectionError
        mock_openai.APIStatusError = Exception  # Dummy, won't be used in this test
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = MockAPIConnectionError("Connection failed")

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()
            provider._get_available_models = MagicMock(return_value=["test-model"])

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(LLMError, match="Connection error: Connection failed"):
                provider.chat_completions_create("test-model", messages)

    @patch("hellmholtz.providers.blablador_provider.openai")
    def test_chat_completions_create_api_connection_error_localhost(self, mock_openai: MagicMock) -> None:
        """Test APIConnectionError with localhost redirect handling."""
        from aisuite.provider import LLMError

        # Create a mock exception class
        class MockAPIConnectionError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.message = message

        mock_openai.APIConnectionError = MockAPIConnectionError
        mock_openai.APIStatusError = Exception  # Dummy, won't be used in this test
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = MockAPIConnectionError("Connection to localhost:8000 failed")

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()
            provider._get_available_models = MagicMock(return_value=["test-model"])

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(LLMError, match="Server configuration error: The API is redirecting to localhost"):
                provider.chat_completions_create("test-model", messages)

    @patch("hellmholtz.providers.blablador_provider.openai")
    def test_chat_completions_create_api_status_error_400(self, mock_openai: MagicMock) -> None:
        """Test APIStatusError 400 handling in chat completions."""
        from aisuite.provider import LLMError

        # Create a proper exception class
        class MockAPIStatusError(Exception):
            def __init__(self, message="", **kwargs):
                super().__init__(message)
                self.status_code = 400
                self.message = "Invalid model"

        # Set the exception classes on the mock
        class DummyAPIConnectionError(Exception):
            pass
        mock_openai.APIConnectionError = DummyAPIConnectionError
        mock_openai.APIStatusError = MockAPIStatusError

        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = MockAPIStatusError("Bad request")
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()
            provider._get_available_models = MagicMock(return_value=["test-model"])

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(LLMError, match="Bad Request: Invalid model"):
                provider.chat_completions_create("test-model", messages)

    @patch("hellmholtz.providers.blablador_provider.openai")
    def test_chat_completions_create_api_status_error_500(self, mock_openai: MagicMock) -> None:
        """Test APIStatusError 500 handling in chat completions."""
        from aisuite.provider import LLMError

        # Create a proper exception class
        class MockAPIStatusError(Exception):
            def __init__(self, message="", **kwargs):
                super().__init__(message)
                self.status_code = 500
                self.message = "Internal server error"

        class DummyAPIConnectionError(Exception):
            pass

        mock_openai.APIConnectionError = DummyAPIConnectionError  # Dummy, won't be used in this test
        mock_openai.APIStatusError = MockAPIStatusError
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = MockAPIStatusError("Server error")

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()
            provider._get_available_models = MagicMock(return_value=["test-model"])

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(LLMError, match="API Error \\(500\\): Internal server error"):
                provider.chat_completions_create("test-model", messages)

    @patch("hellmholtz.providers.blablador_provider.openai")
    def test_chat_completions_create_api_status_error_localhost_500(self, mock_openai: MagicMock) -> None:
        """Test APIStatusError 500 with localhost error handling."""
        from aisuite.provider import LLMError

        # Create a proper exception class
        class MockAPIStatusError(Exception):
            def __init__(self, message="", **kwargs):
                super().__init__(message)
                self.status_code = 500
                self.message = "Server error"

            def __str__(self):
                return "Failed to connect to localhost:8000"

        class DummyAPIConnectionError(Exception):
            pass

        mock_openai.APIConnectionError = DummyAPIConnectionError  # Dummy, won't be used in this test
        mock_openai.APIStatusError = MockAPIStatusError
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = MockAPIStatusError("Server error")

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            provider = BlabladorProvider()
            provider._get_available_models = MagicMock(return_value=["test-model"])

            messages = [{"role": "user", "content": "Hello"}]

            with pytest.raises(LLMError, match="Server configuration error: The API returned an error indicating it failed to reach an internal service at localhost"):
                provider.chat_completions_create("test-model", messages)


class TestBlabladorAPI:
    """Test suite for Blablador API functions."""

    @patch("httpx.get")
    def test_list_models_success(self, mock_get: MagicMock) -> None:
        """Test successful model listing from API."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "1 - GPT-OSS-120b - GPT-OSS-120b model"},
                {"id": "2 - Ministral-3-14B - Ministral-3-14B model"},
            ]
        }
        mock_get.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            from hellmholtz.providers.blablador import list_models

            models = list_models()

            assert len(models) == 2
            assert models[0].id == "1"
            assert models[0].name == "GPT-OSS-120b"  # From KNOWN_MODELS
            assert models[0].description == "GPT-OSS-120b model"  # Keeps API description
            assert models[0].source == "Blablador"

            assert models[1].id == "2"
            assert models[1].name == "Qwen3 235"  # From KNOWN_MODELS
            assert models[1].description == "Ministral-3-14B model"  # Keeps API description

    @patch("httpx.get")
    def test_list_models_missing_config(self, mock_get: MagicMock) -> None:
        """Test model listing with missing configuration."""
        from hellmholtz.providers.blablador import list_models

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Blablador API key and Base URL must be set"):
                list_models()

    @patch("httpx.get")
    def test_list_models_api_error(self, mock_get: MagicMock) -> None:
        """Test model listing with API error."""
        mock_get.side_effect = Exception("Connection failed")

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            from hellmholtz.providers.blablador import list_models

            with pytest.raises(RuntimeError, match="Failed to fetch models from Blablador"):
                list_models()

    @patch("httpx.get")
    def test_list_models_fallback_parsing(self, mock_get: MagicMock) -> None:
        """Test model listing with fallback parsing for different ID formats."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "1 - Simple Model"},  # No description
                {"id": "UnknownFormat"},  # No pattern match
            ]
        }
        mock_get.return_value = mock_response

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            from hellmholtz.providers.blablador import list_models

            models = list_models()

            assert len(models) == 2
            assert models[0].id == "1"
            assert models[0].name == "Simple Model"
            assert models[0].description == ""

            assert models[1].id == "UnknownFormat"
            assert models[1].name == "UnknownFormat"
            assert models[1].description == ""

    @patch("httpx.get")
    def test_list_models_known_model_enrichment(self, mock_get: MagicMock) -> None:
        """Test model listing with known model enrichment."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "1 - TestModel - Basic description"},
            ]
        }
        mock_get.return_value = mock_response

        # Mock known model
        mock_known_model = MagicMock()
        mock_known_model.id = "1"
        mock_known_model.name = "TestModel"
        mock_known_model.alias = "test"
        mock_known_model.description = "Enhanced description"
        mock_known_model.source = "Blablador"

        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            with patch("hellmholtz.providers.blablador.KNOWN_MODELS", [mock_known_model]):
                from hellmholtz.providers.blablador import list_models

                models = list_models()

                assert len(models) == 1
                assert models[0].id == "1"
                assert models[0].name == "TestModel"  # Should use known name
                assert models[0].alias == "test"  # Should get alias from known model
                assert models[0].description == "Basic description"  # Keeps API description since it's not empty
