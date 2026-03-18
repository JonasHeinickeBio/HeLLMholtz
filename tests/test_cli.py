"""
Tests for CLI functionality.

This module contains comprehensive tests for the CLI module,
including unit tests for commands and integration tests.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hellmholtz.cli import app


class TestCLI:
    """Test suite for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_results_file(self, tmp_path: Path) -> Path:
        """Create a temporary results file for testing."""
        results_file = tmp_path / "test_results.json"
        sample_results = [
            {
                "model": "openai:gpt-4o",
                "prompt_id": "test_prompt_1",
                "response_text": "Test response",
                "latency_seconds": 1.5,
                "success": True,
                "timestamp": "2024-01-01T00:00:00",
                "input_tokens": 10,
                "output_tokens": 20,
            }
        ]
        results_file.write_text(json.dumps(sample_results))
        return results_file

    def test_app_exists(self) -> None:
        """Test that the CLI app exists."""
        assert app is not None

    @patch("hellmholtz.cli.chat.chat")
    def test_chat_command_basic(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test basic chat command."""
        mock_chat.return_value = "Test response"

        result = runner.invoke(app, ["chat", "--model", "openai:gpt-4o", "Hello world"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        mock_chat.assert_called_once()

    @patch("hellmholtz.cli.chat.chat")
    def test_chat_command_with_options(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test chat command with temperature and max tokens."""
        mock_chat.return_value = "Test response"

        result = runner.invoke(
            app,
            [
                "chat",
                "--model",
                "openai:gpt-4o",
                "--temperature",
                "0.7",
                "--max-tokens",
                "100",
                "Hello world",
            ],
        )

        assert result.exit_code == 0
        mock_chat.assert_called_once()

    def test_chat_command_missing_model(self, runner: CliRunner) -> None:
        """Test chat command fails without model."""
        result = runner.invoke(app, ["chat", "Hello world"])

        assert result.exit_code != 0

    def test_chat_command_missing_message(self, runner: CliRunner) -> None:
        """Test chat command fails without message."""
        result = runner.invoke(app, ["chat", "--model", "openai:gpt-4o"])

        assert result.exit_code != 0

    @patch("hellmholtz.reporting.generate_markdown_report")
    def test_report_markdown_command(
        self, mock_generate: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test markdown report generation."""
        mock_generate.return_value = "# Test Report\n\nContent here."

        result = runner.invoke(
            app,
            [
                "report",
                str(temp_results_file),
                "--format",
                "markdown",
                "--output",
                "test_report.md",
            ],
        )

        assert result.exit_code == 0
        # Should be called with the loaded results (list of BenchmarkResult)
        assert mock_generate.called
        args, _kwargs = mock_generate.call_args
        assert len(args) == 1
        assert isinstance(args[0], list)  # Should be a list of BenchmarkResult objects

    @patch("hellmholtz.reporting.generate_html_report")
    def test_report_html_command(
        self, mock_generate: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test HTML report generation."""
        mock_generate.return_value = "<html><body>Test Report</body></html>"

        result = runner.invoke(
            app,
            ["report", str(temp_results_file), "--format", "html", "--output", "test_report.html"],
        )

        assert result.exit_code == 0
        assert mock_generate.called

    @patch("hellmholtz.reporting.generate_html_report_simple")
    def test_report_html_simple_command(
        self, mock_generate: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test simple HTML report generation."""
        mock_generate.return_value = "<html><body>Simple Report</body></html>"

        result = runner.invoke(app, ["report", str(temp_results_file), "--format", "html-simple"])

        assert result.exit_code == 0
        assert mock_generate.called

    @patch("hellmholtz.reporting.generate_html_report_detailed")
    def test_report_html_detailed_command(
        self, mock_generate: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test detailed HTML report generation."""
        mock_generate.return_value = "<html><body>Detailed Report</body></html>"

        result = runner.invoke(
            app, ["report", str(temp_results_file), "--format", "html-detailed"]
        )

        assert result.exit_code == 0
        assert mock_generate.called

    @patch("hellmholtz.reporting.generate_html_report_full")
    def test_report_html_full_command(
        self, mock_generate: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test full HTML report generation."""
        mock_generate.return_value = "<html><body>Full Report</body></html>"

        result = runner.invoke(app, ["report", str(temp_results_file), "--format", "html-full"])

        assert result.exit_code == 0
        assert mock_generate.called

    def test_report_command_invalid_format(
        self, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test report command with invalid format defaults to markdown."""
        with patch("hellmholtz.reporting.generate_markdown_report") as mock_generate:
            mock_generate.return_value = "# Test Report"

            result = runner.invoke(app, ["report", str(temp_results_file), "--format", "invalid"])

            assert result.exit_code == 0
            assert mock_generate.called

    def test_report_command_missing_file(self, runner: CliRunner) -> None:
        """Test report command fails without results file."""
        result = runner.invoke(app, ["report"])

        assert result.exit_code != 0

    def test_configure_logging(self) -> None:
        """Test logging configuration."""
        import logging

        from hellmholtz.cli.common import configure_logging

        # Should not raise any exceptions
        configure_logging()

        # Check that logging is configured
        logger = logging.getLogger("hellmholtz.cli")
        assert logger.level <= logging.INFO

    @patch("hellmholtz.benchmark.run_benchmarks")
    def test_bench_command_basic(self, mock_run_benchmarks: MagicMock, runner: CliRunner) -> None:
        """Test basic bench command."""
        result = runner.invoke(app, ["bench", "--models", "openai:gpt-4o"])

        assert result.exit_code == 0
        mock_run_benchmarks.assert_called_once()

    @patch("hellmholtz.integrations.lm_eval.run_lm_eval")
    def test_lm_eval_command(self, mock_run_lm_eval: MagicMock, runner: CliRunner) -> None:
        """Test LM eval command."""
        result = runner.invoke(app, ["lm-eval", "openai:gpt-4o", "arc_easy,arc_challenge"])

        assert result.exit_code == 0
        mock_run_lm_eval.assert_called_once_with("openai:gpt-4o", ["arc_easy", "arc_challenge"], num_fewshot=None, limit=None)

    @patch("hellmholtz.integrations.litellm.start_proxy")
    def test_proxy_command(self, mock_start_proxy: MagicMock, runner: CliRunner) -> None:
        """Test proxy command."""
        result = runner.invoke(app, ["proxy", "openai:gpt-4o"])

        assert result.exit_code == 0
        mock_start_proxy.assert_called_once_with("openai:gpt-4o", port=4000, debug=False)

    @patch("hellmholtz.benchmark.run_throughput_benchmark")
    def test_bench_throughput_command(self, mock_run_throughput: MagicMock, runner: CliRunner) -> None:
        """Test throughput benchmark command."""
        mock_run_throughput.return_value = {
            "success": True,
            "model": "openai:gpt-4o",
            "tokens_per_sec": 50.5,
            "latency": 2.1,
            "output_tokens": 100
        }

        result = runner.invoke(app, ["bench-throughput", "openai:gpt-4o"])

        assert result.exit_code == 0
        mock_run_throughput.assert_called_once()

    @patch("hellmholtz.providers.blablador_config.get_token_limit")
    @patch("hellmholtz.providers.blablador.list_models")
    def test_models_command(self, mock_list_models: MagicMock, mock_get_token_limit: MagicMock, runner: CliRunner) -> None:
        """Test models command."""
        from hellmholtz.core.prompts import Message

        # Mock model data
        mock_model = MagicMock()
        mock_model.id = "1"
        mock_model.name = "Test Model"
        mock_model.alias = "test"
        mock_model.source = "Blablador"
        mock_model.description = "A test model"
        mock_list_models.return_value = [mock_model]
        mock_get_token_limit.return_value = 128000

        result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "Test Model" in result.output
        assert "125k" in result.output  # 128000 / 1024 = 125
        mock_list_models.assert_called_once()
        mock_get_token_limit.assert_called_once_with("Test Model")

    @patch("hellmholtz.evaluation_analysis.analyze_evaluations_cli")
    def test_analyze_command(self, mock_analyze: MagicMock, runner: CliRunner, temp_results_file: Path) -> None:
        """Test analyze command."""
        result = runner.invoke(app, ["analyze", str(temp_results_file)])

        assert result.exit_code == 0
        mock_analyze.assert_called_once_with(str(temp_results_file), None)
