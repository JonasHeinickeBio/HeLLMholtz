"""
Tests for client functionality.

This module contains comprehensive tests for the client module,
including unit tests for chat functions, ClientManager singleton,
and integration tests for different providers.
"""

import pytest
from unittest.mock import MagicMock, patch, MagicMock
from typing import List, Dict, Any

from hellmholtz.client import chat, chat_raw, ClientManager


class TestClientFunctions:
    """Test suite for client functions."""

    @pytest.fixture
    def mock_aisuite_client(self) -> MagicMock:
        """Create a mock aisuite client for testing."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @patch("hellmholtz.client.ai.Client")
    def test_chat_basic_functionality(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test basic chat functionality with string response."""
        mock_client_cls.return_value = mock_aisuite_client

        # Reset singleton to ensure clean state
        ClientManager._default_instance = None

        messages = [{"role": "user", "content": "Hello"}]
        response = chat("openai:gpt-4o", messages)

        assert response == "Test response"
        mock_aisuite_client.chat.completions.create.assert_called_once_with(
            model="openai:gpt-4o",
            messages=messages
        )

    @patch("hellmholtz.client.ai.Client")
    def test_chat_with_different_models(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test chat with different model providers."""
        mock_client_cls.return_value = mock_aisuite_client

        ClientManager._default_instance = None

        test_cases = [
            "openai:gpt-4o",
            "anthropic:claude-3-sonnet-20240229",
            "google:gemini-pro",
            "blablador:test-model"
        ]

        messages = [{"role": "user", "content": "Test"}]

        for model in test_cases:
            response = chat(model, messages)
            assert response == "Test response"

        # Verify correct number of calls
        assert mock_aisuite_client.chat.completions.create.call_count == len(test_cases)

    @patch("hellmholtz.client.ai.Client")
    def test_chat_with_complex_messages(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test chat with complex message structures."""
        mock_client_cls.return_value = mock_aisuite_client

        ClientManager._default_instance = None

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]

        response = chat("openai:gpt-4o", messages)

        assert response == "Test response"
        mock_aisuite_client.chat.completions.create.assert_called_once_with(
            model="openai:gpt-4o",
            messages=messages
        )

    @patch("hellmholtz.client.ai.Client")
    def test_chat_raw_returns_full_response(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test chat_raw returns the full response object."""
        mock_client_cls.return_value = mock_aisuite_client

        ClientManager._default_instance = None

        messages = [{"role": "user", "content": "Hello"}]
        response = chat_raw("openai:gpt-4o", messages)

        assert response == mock_aisuite_client.chat.completions.create.return_value
        mock_aisuite_client.chat.completions.create.assert_called_once_with(
            model="openai:gpt-4o",
            messages=messages
        )

    @patch("hellmholtz.client.ai.Client")
    def test_chat_with_empty_messages(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test chat behavior with empty messages."""
        mock_client_cls.return_value = mock_aisuite_client

        ClientManager._default_instance = None

        messages = []
        response = chat("openai:gpt-4o", messages)

        assert response == "Test response"
        mock_aisuite_client.chat.completions.create.assert_called_once_with(
            model="openai:gpt-4o",
            messages=messages
        )

    @patch("hellmholtz.client.ai.Client")
    def test_client_manager_singleton_behavior(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test that ClientManager maintains singleton behavior for the underlying client."""
        mock_client_cls.return_value = mock_aisuite_client

        # Reset singleton
        ClientManager._default_instance = None

        # Get clients - should reuse the same instance
        client1, _ = ClientManager.get_client("openai:gpt-4o")
        client2, _ = ClientManager.get_client("openai:gpt-4o")

        # Should be the same client instance
        assert client1 is client2
        assert ClientManager._default_instance is client1

    @patch("hellmholtz.client.ai.Client")
    def test_client_manager_get_client(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test ClientManager.get_client method."""
        mock_client_cls.return_value = mock_aisuite_client

        ClientManager._default_instance = None

        manager = ClientManager()
        client, model = manager.get_client("openai:gpt-4o")

        assert client is mock_aisuite_client
        assert model == "openai:gpt-4o"
        mock_client_cls.assert_called_once()

    @patch("hellmholtz.client.ai.Client")
    def test_multiple_chat_calls_reuse_client(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test that multiple chat calls reuse the same client instance."""
        mock_client_cls.return_value = mock_aisuite_client

        ClientManager._default_instance = None

        # Make multiple calls
        messages = [{"role": "user", "content": "Test"}]
        chat("openai:gpt-4o", messages)
        chat("openai:gpt-4o", messages)
        chat("openai:gpt-4o", messages)

        # Client should only be created once
        mock_client_cls.assert_called_once()
        # But chat.completions.create should be called three times
        assert mock_aisuite_client.chat.completions.create.call_count == 3

    @patch("hellmholtz.client.ai.Client")
    def test_chat_with_exception_handling(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test that exceptions from the underlying client are propagated."""
        mock_client_cls.return_value = mock_aisuite_client
        mock_aisuite_client.chat.completions.create.side_effect = Exception("API Error")

        ClientManager._default_instance = None

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception, match="API Error"):
            chat("openai:gpt-4o", messages)

    @patch("hellmholtz.client.ai.Client")
    def test_chat_raw_with_exception_handling(self, mock_client_cls: MagicMock, mock_aisuite_client: MagicMock) -> None:
        """Test that chat_raw propagates exceptions from the underlying client."""
        mock_client_cls.return_value = mock_aisuite_client
        mock_aisuite_client.chat.completions.create.side_effect = Exception("API Error")

        ClientManager._default_instance = None

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception, match="API Error"):
            chat_raw("openai:gpt-4o", messages)
