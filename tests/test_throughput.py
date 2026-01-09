from unittest.mock import MagicMock, patch
from hellmholtz.benchmark import run_throughput_benchmark

@patch("hellmholtz.benchmark.runner.chat_raw")
def test_throughput_run(mock_chat: object) -> None:
    # Mock response with proper structure
    mock_response = MagicMock()
    mock_response.usage.completion_tokens = 100
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response with some content."
    mock_chat.return_value = mock_response

    result = run_throughput_benchmark("openai:gpt-4o")

    assert result["success"] is True
    assert result["output_tokens"] == 100
    assert result["tokens_per_sec"] > 0
