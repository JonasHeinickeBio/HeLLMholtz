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

    @patch("hellmholtz.cli.chat")
    def test_chat_command_basic(self, mock_chat: MagicMock, runner: CliRunner) -> None:
        """Test basic chat command."""
        mock_chat.return_value = "Test response"

        result = runner.invoke(app, ["chat", "--model", "openai:gpt-4o", "Hello world"])

        assert result.exit_code == 0
        assert "Test response" in result.output
        mock_chat.assert_called_once()

    @patch("hellmholtz.cli.chat")
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

    @patch("hellmholtz.cli.generate_markdown_report")
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
        args, kwargs = mock_generate.call_args
        assert len(args) == 1
        assert isinstance(args[0], list)  # Should be a list of BenchmarkResult objects

    @patch("hellmholtz.cli.generate_html_report")
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

    @patch("hellmholtz.cli.generate_html_report_simple")
    def test_report_html_simple_command(
        self, mock_generate: MagicMock, runner: CliRunner, temp_results_file: Path
    ) -> None:
        """Test simple HTML report generation."""
        mock_generate.return_value = "<html><body>Simple Report</body></html>"

        result = runner.invoke(app, ["report", str(temp_results_file), "--format", "html-simple"])

        assert result.exit_code == 0
        assert mock_generate.called

    @patch("hellmholtz.cli.generate_html_report_detailed")
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

    @patch("hellmholtz.cli.generate_html_report_full")
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
        with patch("hellmholtz.cli.generate_markdown_report") as mock_generate:
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

        from hellmholtz.cli import configure_logging

        # Should not raise any exceptions
        configure_logging()

        # Check that logging is configured
        logger = logging.getLogger("hellmholtz.cli")
        assert logger.level <= logging.INFO
