"""
Tests for throughput benchmark functionality.

This module contains comprehensive tests for the throughput benchmarking,
including unit tests for token throughput calculations and edge cases.
"""

from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.benchmark.runner import run_throughput_benchmark


class TestThroughputBenchmark:
    """Test suite for throughput benchmark functionality."""

    @pytest.fixture
    def mock_chat_response_with_usage(self) -> MagicMock:
        """Create a mock chat response with usage information."""
        mock_response = MagicMock()
        mock_response.usage.completion_tokens = 100
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "This is a test response with some content for throughput testing."
        )
        return mock_response

    @pytest.fixture
    def mock_chat_response_no_usage(self) -> MagicMock:
        """Create a mock chat response without usage information."""
        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "This is a test response with some content for throughput testing."
        )
        return mock_response

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_success_with_usage(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        mock_chat_response_with_usage: MagicMock,
    ) -> None:
        """Test successful throughput benchmark with usage information."""
        # Mock timing
        mock_perf_counter.side_effect = [0.0, 1.0]  # Start at 0, end at 1 second

        mock_chat_raw.return_value = mock_chat_response_with_usage

        result = run_throughput_benchmark("openai:gpt-4o")

        assert result["success"] is True
        assert result["model"] == "openai:gpt-4o"
        assert result["latency"] == 1.0
        assert result["output_tokens"] == 100
        assert result["tokens_per_sec"] == 100.0  # 100 tokens / 1 second

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_success_no_usage(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        mock_chat_response_no_usage: MagicMock,
    ) -> None:
        """Test successful throughput benchmark without usage information."""
        # Mock timing
        mock_perf_counter.side_effect = [0.0, 2.0]  # Start at 0, end at 2 seconds

        mock_chat_raw.return_value = mock_chat_response_no_usage

        result = run_throughput_benchmark("ollama:llama3")

        assert result["success"] is True
        assert result["model"] == "ollama:llama3"
        assert result["latency"] == 2.0

        # Should estimate tokens: len(content) // 4 (integer division)
        content = mock_chat_response_no_usage.choices[0].message.content
        expected_tokens = len(content) // 4
        assert result["output_tokens"] == expected_tokens
        assert result["tokens_per_sec"] == expected_tokens / 2.0

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_custom_parameters(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        mock_chat_response_with_usage: MagicMock,
    ) -> None:
        """Test throughput benchmark with custom prompt and max_tokens."""
        mock_perf_counter.side_effect = [0.0, 0.5]
        mock_chat_raw.return_value = mock_chat_response_with_usage

        custom_prompt = "Write a short poem about coding."
        max_tokens = 50

        result = run_throughput_benchmark(
            model="anthropic:claude-3-sonnet-20240229", prompt=custom_prompt, max_tokens=max_tokens
        )

        assert result["success"] is True
        assert result["model"] == "anthropic:claude-3-sonnet-20240229"

        # Verify the call was made with correct parameters
        mock_chat_raw.assert_called_once_with(
            model="anthropic:claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": custom_prompt}],
            max_tokens=max_tokens,
        )

    @patch("hellmholtz.benchmark.runner.chat_raw")
    def test_throughput_benchmark_failure(self, mock_chat_raw: MagicMock) -> None:
        """Test throughput benchmark when chat call fails."""
        mock_chat_raw.side_effect = Exception("API Error")

        result = run_throughput_benchmark("openai:gpt-4o")

        assert result["success"] is False
        assert result["model"] == "openai:gpt-4o"
        assert "error" in result
        assert result["error"] == "API Error"

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_zero_latency(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        mock_chat_response_with_usage: MagicMock,
    ) -> None:
        """Test throughput benchmark with zero latency (edge case)."""
        # Mock timing with zero latency
        mock_perf_counter.side_effect = [0.0, 0.0]

        mock_chat_raw.return_value = mock_chat_response_with_usage

        result = run_throughput_benchmark("openai:gpt-4o")

        assert result["success"] is True
        assert result["latency"] == 0.0
        assert result["tokens_per_sec"] == 0  # Division by zero protection

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_different_models(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        mock_chat_response_with_usage: MagicMock,
    ) -> None:
        """Test throughput benchmark with different model providers."""
        # Each benchmark call needs 2 perf_counter calls (start and end)
        mock_perf_counter.side_effect = [0.0, 1.0] * 5  # 5 models × 2 calls each
        mock_chat_raw.return_value = mock_chat_response_with_usage

        test_models = [
            "openai:gpt-4o",
            "anthropic:claude-3-sonnet-20240229",
            "google:gemini-pro",
            "ollama:llama3.2:3b",
            "blablador:test-model",
        ]

        for model in test_models:
            result = run_throughput_benchmark(model)
            assert result["success"] is True
            assert result["model"] == model
            assert result["tokens_per_sec"] == 100.0

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_empty_response(
        self, mock_perf_counter: MagicMock, mock_chat_raw: MagicMock
    ) -> None:
        """Test throughput benchmark with empty response content."""
        mock_perf_counter.side_effect = [0.0, 1.0]

        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""  # Empty content

        mock_chat_raw.return_value = mock_response

        result = run_throughput_benchmark("openai:gpt-4o")

        assert result["success"] is True
        assert result["output_tokens"] == 0  # len("") / 4 = 0
        assert result["tokens_per_sec"] == 0

    @pytest.mark.parametrize("content,expected_tokens", [
        ("Hello", 1),  # 5 chars // 4 = 1
        ("Hello world!", 3),  # 12 chars // 4 = 3
        ("This is a longer response with more content.", 11),  # 44 chars // 4 = 11
    ])
    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_token_estimation(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        content: str,
        expected_tokens: int
    ) -> None:
        """Test token estimation when usage info is not available."""
        mock_perf_counter.side_effect = [0.0, 1.0]

        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content

        mock_chat_raw.return_value = mock_response

        result = run_throughput_benchmark("ollama:llama3")

        assert result["success"] is True
        assert result["output_tokens"] == expected_tokens
        assert result["tokens_per_sec"] == expected_tokens

    @patch("hellmholtz.benchmark.runner.chat_raw")
    @patch("time.perf_counter")
    def test_throughput_benchmark_default_parameters(
        self,
        mock_perf_counter: MagicMock,
        mock_chat_raw: MagicMock,
        mock_chat_response_with_usage: MagicMock,
    ) -> None:
        """Test throughput benchmark with default parameters."""
        mock_perf_counter.side_effect = [0.0, 1.0]
        mock_chat_raw.return_value = mock_chat_response_with_usage

        result = run_throughput_benchmark("openai:gpt-4o")

        # Verify default prompt and max_tokens were used
        mock_chat_raw.assert_called_once_with(
            model="openai:gpt-4o",
            messages=[{"role": "user", "content": "Write a long story about a space adventure."}],
            max_tokens=100,
        )

        assert result["success"] is True
