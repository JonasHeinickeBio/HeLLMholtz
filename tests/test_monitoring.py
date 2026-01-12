"""
Tests for monitoring functionality.

This module contains comprehensive tests for the monitoring module,
including unit tests for ModelAvailabilityMonitor and integration tests.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.monitoring import ModelAvailabilityMonitor


class TestModelAvailabilityMonitor:
    """Test suite for ModelAvailabilityMonitor."""

    @pytest.fixture
    def mock_api_response(self) -> dict[str, Any]:
        """Create a mock API response for models endpoint."""
        return {
            "data": [
                {"id": "1 - GPT-OSS-120b - GPT-OSS-120b model"},
                {"id": "2 - Ministral-3-14B - Ministral-3-14B model"},
            ]
        }

    @pytest.fixture
    def mock_chat_response(self) -> str:
        """Create a mock chat response."""
        return "Test response"

    def test_init_with_env_vars(self) -> None:
        """Test initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {"BLABLADOR_API_KEY": "test_key", "BLABLADOR_API_BASE": "https://api.example.com/v1"},
        ):
            monitor = ModelAvailabilityMonitor()
            assert monitor.api_key == "test_key"
            assert monitor.api_base == "https://api.example.com/v1"

    def test_init_with_params(self) -> None:
        """Test initialization with explicit parameters."""
        monitor = ModelAvailabilityMonitor(
            api_key="explicit_key", api_base="https://explicit.example.com/v1"
        )
        assert monitor.api_key == "explicit_key"
        assert monitor.api_base == "https://explicit.example.com/v1"

    def test_init_missing_api_key(self) -> None:
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError, match="BLABLADOR_API_KEY not found"):
            ModelAvailabilityMonitor()

    def test_init_default_api_base(self) -> None:
        """Test initialization uses default API base when not provided."""
        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}, clear=True):
            monitor = ModelAvailabilityMonitor()
            assert monitor.api_base == "https://api.blablador.example.com/v1"

    @patch("requests.get")
    def test_get_api_models_success(
        self, mock_get: MagicMock, mock_api_response: dict[str, Any]
    ) -> None:
        """Test successful API models fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}, clear=True):
            monitor = ModelAvailabilityMonitor()
            models = monitor.get_api_models()

            assert len(models) == 2
            assert models[0]["id"] == "1 - GPT-OSS-120b - GPT-OSS-120b model"
            mock_get.assert_called_once_with(
                "https://api.blablador.example.com/v1/models",
                headers={"Authorization": "Bearer test_key"},
                timeout=30,
            )

    @patch("requests.get")
    def test_get_api_models_request_failure(self, mock_get: MagicMock) -> None:
        """Test API models fetch with request failure."""
        mock_get.side_effect = Exception("Connection failed")

        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}):
            monitor = ModelAvailabilityMonitor()

            with pytest.raises(RuntimeError, match="Failed to fetch models from API"):
                monitor.get_api_models()

    def test_get_configured_models(self) -> None:
        """Test getting configured models."""
        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}):
            monitor = ModelAvailabilityMonitor()
            models = monitor.get_configured_models()

            # Should return a dict mapping API IDs to BlabladorModel instances
            assert isinstance(models, dict)
            # The exact content depends on blablador_config.KNOWN_MODELS

    @patch("hellmholtz.monitoring.chat")
    def test_test_model_accessibility_success(
        self, mock_chat: MagicMock, mock_chat_response: str
    ) -> None:
        """Test successful model accessibility test."""
        mock_chat.return_value = mock_chat_response

        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}):
            monitor = ModelAvailabilityMonitor()
            accessible, latency = monitor.test_model_accessibility("test-model")

            assert accessible is True
            assert isinstance(latency, float)
            assert latency >= 0
            mock_chat.assert_called_once_with(
                "blablador:test-model",
                [{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=10.0,
            )

    @patch("hellmholtz.monitoring.chat")
    def test_test_model_accessibility_failure(self, mock_chat: MagicMock) -> None:
        """Test model accessibility test with failure."""
        mock_chat.side_effect = Exception("Model not available")

        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}):
            monitor = ModelAvailabilityMonitor()
            accessible, latency = monitor.test_model_accessibility("test-model")

            assert accessible is False
            assert latency == 0.0

    @patch("hellmholtz.monitoring.chat")
    def test_test_model_accessibility_empty_response(self, mock_chat: MagicMock) -> None:
        """Test model accessibility test with empty response."""
        mock_chat.return_value = ""

        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}):
            monitor = ModelAvailabilityMonitor()
            accessible, latency = monitor.test_model_accessibility("test-model")

            assert accessible is False
            assert isinstance(latency, float)

    @patch("hellmholtz.monitoring.chat")
    def test_test_model_accessibility_none_response(self, mock_chat: MagicMock) -> None:
        """Test model accessibility test with None response."""
        mock_chat.return_value = None

        with patch.dict("os.environ", {"BLABLADOR_API_KEY": "test_key"}):
            monitor = ModelAvailabilityMonitor()
            accessible, latency = monitor.test_model_accessibility("test-model")

            assert accessible is False
            assert isinstance(latency, float)
