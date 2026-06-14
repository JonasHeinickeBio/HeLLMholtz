"""Tests for hellmholtz.cli.integrations implementation functions."""

from unittest.mock import MagicMock, patch

import pytest
import typer

from hellmholtz.cli.integrations import (
    _bench_throughput_impl,
    _lm_eval_impl,
    _proxy_impl,
)


class TestLmEvalImpl:
    """Tests for _lm_eval_impl."""

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.lm_eval.run_lm_eval")
    def test_calls_with_single_task(self, mock_run: MagicMock, mock_err: MagicMock) -> None:
        _lm_eval_impl("openai:gpt-4o", "mmlu", num_fewshot=5, limit=10.0)
        mock_run.assert_called_once_with(
            "openai:gpt-4o", ["mmlu"], num_fewshot=5, limit=10.0
        )

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.lm_eval.run_lm_eval")
    def test_calls_with_multiple_tasks(self, mock_run: MagicMock, mock_err: MagicMock) -> None:
        _lm_eval_impl("model", "mmlu,hellaswag,arc", num_fewshot=None, limit=None)
        mock_run.assert_called_once_with(
            "model", ["mmlu", "hellaswag", "arc"], num_fewshot=None, limit=None
        )

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.lm_eval.run_lm_eval")
    def test_strips_whitespace_from_tasks(self, mock_run: MagicMock, mock_err: MagicMock) -> None:
        _lm_eval_impl("model", " task1 , task2 ", num_fewshot=None, limit=None)
        mock_run.assert_called_once_with(
            "model", ["task1", "task2"], num_fewshot=None, limit=None
        )

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.lm_eval.run_lm_eval", side_effect=RuntimeError("eval failed"))
    def test_error_calls_handle_error(self, mock_run: MagicMock, mock_err: MagicMock) -> None:
        _lm_eval_impl("model", "mmlu", num_fewshot=None, limit=None)
        mock_err.assert_called_once()


class TestProxyImpl:
    """Tests for _proxy_impl."""

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.litellm.start_proxy")
    def test_starts_proxy_with_defaults(self, mock_start: MagicMock, mock_err: MagicMock) -> None:
        _proxy_impl("openai:gpt-4o", port=4000, debug=False)
        mock_start.assert_called_once_with("openai:gpt-4o", port=4000, debug=False)

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.litellm.start_proxy")
    def test_starts_proxy_with_debug(self, mock_start: MagicMock, mock_err: MagicMock) -> None:
        _proxy_impl("model", port=8080, debug=True)
        mock_start.assert_called_once_with("model", port=8080, debug=True)

    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.integrations.litellm.start_proxy", side_effect=OSError("port in use"))
    def test_proxy_error_calls_handle_error(self, mock_start: MagicMock, mock_err: MagicMock) -> None:
        _proxy_impl("model", port=4000, debug=False)
        mock_err.assert_called_once()


class TestBenchThroughputImpl:
    """Tests for _bench_throughput_impl."""

    @patch("hellmholtz.cli.integrations.typer")
    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.benchmark.run_throughput_benchmark")
    def test_successful_benchmark(self, mock_bench: MagicMock, mock_err: MagicMock, mock_typer: MagicMock) -> None:
        mock_bench.return_value = {
            "success": True,
            "model": "openai:gpt-4o",
            "tokens_per_sec": 42.5,
            "latency": 1.23,
            "output_tokens": 50,
        }

        _bench_throughput_impl("openai:gpt-4o", "Write a story", max_tokens=100)

        mock_bench.assert_called_once_with("openai:gpt-4o", "Write a story", 100)
        echo_msgs = [str(c) for c in mock_typer.echo.call_args_list]
        assert any("42.50" in m for m in echo_msgs)

    @patch("hellmholtz.cli.integrations.typer")
    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.benchmark.run_throughput_benchmark")
    def test_failed_benchmark(self, mock_bench: MagicMock, mock_err: MagicMock, mock_typer: MagicMock) -> None:
        mock_bench.return_value = {
            "success": False,
            "error": "connection refused",
        }

        _bench_throughput_impl("model", "prompt", 100)
        mock_err.assert_not_called()

    @patch("hellmholtz.cli.integrations.typer")
    @patch("hellmholtz.cli.integrations.handle_error")
    @patch("hellmholtz.benchmark.run_throughput_benchmark", side_effect=Exception("crash"))
    def test_benchmark_exception(self, mock_bench: MagicMock, mock_err: MagicMock, mock_typer: MagicMock) -> None:
        _bench_throughput_impl("model", "prompt", 100)
        mock_err.assert_called_once()
