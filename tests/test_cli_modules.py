"""
Tests for modularized CLI functionality.

This module contains comprehensive unit tests for the refactored CLI modules:
- common.py: DRY helpers and utilities
- chat.py: Chat command group
- benchmark.py: Benchmark command group
- models.py: Models command group
- integrations.py: Integration command group
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.exceptions import Exit
from typer.testing import CliRunner

from hellmholtz.cli import app
from hellmholtz.cli.common import (
    configure_logging,
    format_token_limit,
    generate_output_path,
    handle_error,
    parse_models,
    parse_temperatures,
    save_report_to_file,
)


class TestCommonHelpers:
    """Test suite for common.py helpers."""

    def test_format_token_limit_small(self) -> None:
        """Test token limit formatting for small values."""
        assert format_token_limit(100) == "100"
        assert format_token_limit(512) == "512"

    def test_format_token_limit_kilobytes(self) -> None:
        """Test token limit formatting for kilobyte range."""
        assert format_token_limit(1024) == "1k"
        assert format_token_limit(4096) == "4k"
        assert format_token_limit(128000) == "125k"

    def test_format_token_limit_megabytes(self) -> None:
        """Test token limit formatting for megabyte range."""
        assert format_token_limit(1024 * 1024) == "1M"
        assert format_token_limit(1024 * 1024 * 4) == "4M"

    def test_configure_logging(self) -> None:
        """Test logging configuration setup."""
        # Should not raise any exceptions
        configure_logging()

        # Verify logging is configured
        logger = logging.getLogger("test_logger")
        assert logger is not None

    def test_parse_temperatures_valid(self) -> None:
        """Test temperature parsing with valid input."""
        temps = parse_temperatures("0.1,0.5,0.9")
        assert temps == [0.1, 0.5, 0.9]

    def test_parse_temperatures_whitespace(self) -> None:
        """Test temperature parsing with whitespace."""
        temps = parse_temperatures("0.1 , 0.5 , 0.9")
        assert temps == [0.1, 0.5, 0.9]

    def test_parse_temperatures_default(self) -> None:
        """Test temperature parsing with None returns default."""
        temps = parse_temperatures(None)
        assert temps == [0.1, 0.7, 1.0]

    def test_parse_temperatures_invalid(self) -> None:
        """Test temperature parsing with invalid input exits."""
        with pytest.raises(Exit):
            parse_temperatures("invalid,values")

    @patch("hellmholtz.core.config.get_settings")
    def test_parse_models_with_input(self, mock_settings: MagicMock) -> None:
        """Test model parsing with explicit input."""
        models = parse_models("openai:gpt-4o,anthropic:claude-3")
        assert models == ["openai:gpt-4o", "anthropic:claude-3"]

    @patch("hellmholtz.providers.blablador.list_models")
    @patch("hellmholtz.core.config.get_settings")
    def test_parse_models_default(
        self, mock_settings: MagicMock, mock_list_models: MagicMock
    ) -> None:
        """Test model parsing with defaults."""
        mock_settings.return_value.default_models = ["openai:gpt-4o"]
        mock_list_models.return_value = []

        models = parse_models(None)
        assert "openai:gpt-4o" in models

    @patch("hellmholtz.providers.blablador.list_models")
    @patch("hellmholtz.core.config.get_settings")
    def test_parse_models_no_available(
        self, mock_settings: MagicMock, mock_blablador: MagicMock
    ) -> None:
        """Test model parsing with no available models exits."""
        mock_settings.return_value.default_models = []
        mock_blablador.return_value = []

        with pytest.raises(Exit):
            parse_models(None)

    def test_handle_error_exits(self) -> None:
        """Test that handle_error raises Exit."""
        with pytest.raises(Exit):
            handle_error(ValueError("test error"), "Test context")

    def test_generate_output_path_markdown(self, tmp_path: Path) -> None:
        """Test output path generation for markdown."""
        mock_result = MagicMock()
        mock_result.timestamp = "2024-01-01T00:00:00"

        path = generate_output_path([mock_result], "markdown")
        assert path.suffix == ".md"
        assert "benchmark_report_" in str(path)

    def test_generate_output_path_html(self, tmp_path: Path) -> None:
        """Test output path generation for HTML."""
        mock_result = MagicMock()
        mock_result.timestamp = "2024-01-01T00:00:00"

        path = generate_output_path([mock_result], "html")
        assert path.suffix == ".html"
        assert "html/" in str(path)

    def test_save_report_to_file(self, tmp_path: Path) -> None:
        """Test saving report content to file."""
        output_path = tmp_path / "test_report.md"
        content = "# Test Report\n\nContent here."

        save_report_to_file(content, output_path)

        assert output_path.exists()
        assert output_path.read_text() == content

    def test_save_report_to_file_creates_dirs(self, tmp_path: Path) -> None:
        """Test that save_report_to_file creates necessary directories."""
        output_path = tmp_path / "reports" / "nested" / "test_report.md"
        content = "# Test Report"

        save_report_to_file(content, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()


class TestChatCommands:
    """Test suite for chat.py command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner for testing."""
        return CliRunner()

    @patch("hellmholtz.cli.chat.chat")
    def test_chat_command_basic(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test basic chat command."""
        mock_chat.return_value = "Test response"

        result = runner.invoke(app, ["chat", "--model", "openai:gpt-4o", "Hello"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        mock_chat.assert_called_once()

    @patch("hellmholtz.cli.chat.chat")
    def test_chat_command_with_temperature(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test chat command with custom temperature."""
        mock_chat.return_value = "Test response"

        result = runner.invoke(
            app, ["chat", "--model", "openai:gpt-4o", "--temperature", "0.5", "Hello"]
        )

        assert result.exit_code == 0
        # Verify temperature was passed
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    @patch("hellmholtz.cli.chat.chat")
    def test_chat_command_default_temperature(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test chat command uses default temperature."""
        mock_chat.return_value = "response"

        result = runner.invoke(app, ["chat", "--model", "openai:gpt-4o", "test"])

        assert result.exit_code == 0
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["temperature"] == 0.7  # Default


class TestBenchmarkCommands:
    """Test suite for benchmark.py command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_results_file(self, tmp_path: Path) -> Path:
        """Create a temporary results file."""
        results_file = tmp_path / "results.json"
        sample_results = [
            {
                "model": "openai:gpt-4o",
                "prompt_id": "test",
                "response_text": "response",
                "latency_seconds": 1.0,
                "success": True,
                "timestamp": "2024-01-01T00:00:00",
                "input_tokens": 10,
                "output_tokens": 20,
            }
        ]
        results_file.write_text(json.dumps(sample_results))
        return results_file

    @patch("hellmholtz.benchmark.run_benchmarks")
    def test_bench_command_basic(self, mock_bench: MagicMock, runner: CliRunner) -> None:
        """Test bench command."""
        mock_bench.return_value = []

        result = runner.invoke(app, ["bench", "--models", "openai:gpt-4o"])

        assert result.exit_code == 0

    @patch("hellmholtz.reporting.generate_markdown_report")
    def test_report_command_markdown(
        self, mock_gen: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test report command with markdown format."""
        mock_gen.return_value = "# Report"

        result = runner.invoke(app, ["report", str(temp_results_file), "--format", "markdown"])

        assert result.exit_code == 0

    @patch("hellmholtz.reporting.generate_html_report")
    def test_report_command_html(
        self, mock_gen: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test report command with HTML format."""
        mock_gen.return_value = "<html></html>"

        result = runner.invoke(app, ["report", str(temp_results_file), "--format", "html"])

        assert result.exit_code == 0

    @patch("hellmholtz.reporting.chart.generate_performance_chart")
    def test_chart_command(
        self, mock_chart: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test chart command."""
        result = runner.invoke(app, ["chart", str(temp_results_file)])

        assert result.exit_code == 0

    @patch("hellmholtz.evaluation_analysis.analyze_evaluations_cli")
    def test_analyze_command(
        self, mock_analyze: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test analyze command."""
        result = runner.invoke(app, ["analyze", str(temp_results_file)])

        assert result.exit_code == 0
        mock_analyze.assert_called_once()


class TestModelsCommands:
    """Test suite for models.py command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner for testing."""
        return CliRunner()

    @patch("hellmholtz.providers.blablador_config.get_token_limit")
    @patch("hellmholtz.providers.blablador.list_models")
    def test_models_command(self, mock_list: MagicMock, mock_tokens: MagicMock, runner: CliRunner) -> None:
        """Test models command."""
        mock_model = MagicMock()
        mock_model.id = "1"
        mock_model.name = "Test Model"
        mock_model.alias = "test"
        mock_model.source = "Blablador"
        mock_model.description = "Test"
        mock_list.return_value = [mock_model]
        mock_tokens.return_value = 128000

        result = runner.invoke(app, ["models"])

        assert result.exit_code == 0
        assert "Test Model" in result.output

    @patch("hellmholtz.client.check_model_availability")
    def test_check_command_available(self, mock_check: MagicMock, runner: CliRunner) -> None:
        """Test check command with available model."""
        mock_check.return_value = True

        result = runner.invoke(app, ["check", "openai:gpt-4o"])

        assert result.exit_code == 0
        assert "available" in result.output.lower()

    @patch("hellmholtz.client.check_model_availability")
    def test_check_command_unavailable(self, mock_check: MagicMock, runner: CliRunner) -> None:
        """Test check command with unavailable model."""
        mock_check.return_value = False

        result = runner.invoke(app, ["check", "openai:gpt-4o"])

        assert result.exit_code == 1


class TestIntegrationCommands:
    """Test suite for integrations.py command group."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner for testing."""
        return CliRunner()

    @patch("hellmholtz.integrations.lm_eval.run_lm_eval")
    def test_lm_eval_command(self, mock_lm_eval: MagicMock, runner: CliRunner) -> None:
        """Test lm_eval command."""
        result = runner.invoke(app, ["lm-eval", "openai:gpt-4o", "arc_easy"])

        assert result.exit_code == 0
        mock_lm_eval.assert_called_once()

    @patch("hellmholtz.integrations.litellm.start_proxy")
    def test_proxy_command(self, mock_proxy: MagicMock, runner: CliRunner) -> None:
        """Test proxy command."""
        result = runner.invoke(app, ["proxy", "openai:gpt-4o"])

        assert result.exit_code == 0
        mock_proxy.assert_called_once()

    @patch("hellmholtz.benchmark.run_throughput_benchmark")
    def test_bench_throughput_command(
        self, mock_throughput: MagicMock, runner: CliRunner
    ) -> None:
        """Test bench_throughput command."""
        mock_throughput.return_value = {
            "success": True,
            "model": "openai:gpt-4o",
            "tokens_per_sec": 50.0,
            "latency": 2.0,
            "output_tokens": 100,
        }

        result = runner.invoke(app, ["bench-throughput", "openai:gpt-4o"])

        assert result.exit_code == 0


class TestCLIIntegration:
    """Integration tests for the modularized CLI."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_app_help(self, runner: CliRunner) -> None:
        """Test that app help works and shows all commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        # Should list all command groups
        assert "chat" in result.output
        assert "bench" in result.output
        assert "report" in result.output
        assert "models" in result.output
        assert "check" in result.output
        assert "monitor" in result.output
        assert "lm-eval" in result.output
        assert "proxy" in result.output

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test that invalid command exits with error."""
        result = runner.invoke(app, ["invalid-command"])

        assert result.exit_code != 0

    @patch("hellmholtz.cli.chat.chat")
    def test_multiple_commands(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test that multiple commands can be executed."""
        mock_chat.return_value = "response"

        # Run first command
        result1 = runner.invoke(app, ["chat", "--model", "openai:gpt-4o", "test1"])
        assert result1.exit_code == 0

        # Run second command (with fresh runner)
        runner2 = CliRunner()
        result2 = runner2.invoke(app, ["chat", "--model", "openai:gpt-4o", "test2"])
        assert result2.exit_code == 0
