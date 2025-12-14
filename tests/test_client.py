from unittest.mock import MagicMock, patch
from hellmholtz.client import chat, chat_raw, ClientManager

@patch("hellmholtz.client.ai.Client")
def test_chat(mock_client_cls: MagicMock) -> None:
    # Setup mock
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello world"))]
    mock_instance.chat.completions.create.return_value = mock_response

    # Reset singleton
    ClientManager._default_instance = None

    response = chat("openai:gpt-4o", [{"role": "user", "content": "Hi"}])

    assert response == "Hello world"
    mock_instance.chat.completions.create.assert_called_once()

@patch("hellmholtz.client.ai.Client")
def test_chat_raw(mock_client_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_client_cls.return_value = mock_instance

    # Reset singleton
    ClientManager._default_instance = None

    chat_raw("openai:gpt-4o", [{"role": "user", "content": "Hi"}])
    mock_instance.chat.completions.create.assert_called_once()
