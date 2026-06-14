"""Tests for hellmholtz.cli.models implementation functions."""

from unittest.mock import MagicMock, patch

import pytest
import typer

from hellmholtz.cli.models import (
    _auto_agent_impl,
    _check_impl,
    _models_impl,
    _monitor_impl,
)


class TestModelsImpl:
    """Tests for _models_impl."""

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=8192)
    @patch("hellmholtz.providers.blablador.list_models")
    def test_displays_models(
        self, mock_list: MagicMock, mock_token_limit: MagicMock,
        mock_typer: MagicMock, mock_err: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model.id = "model-1"
        mock_model.name = "model-1"
        mock_model.alias = "alias1"
        mock_model.source = "openai"
        mock_model.description = "A test model"
        mock_list.return_value = [mock_model]

        _models_impl()

        assert mock_typer.echo.call_count >= 2  # header + separator + model row

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.providers.blablador.list_models")
    def test_handles_list_error(self, mock_list: MagicMock, mock_err: MagicMock) -> None:
        mock_list.side_effect = RuntimeError("API down")
        _models_impl()
        mock_err.assert_called_once()

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=4096)
    @patch("hellmholtz.providers.blablador.list_models")
    def test_id_same_as_name_shows_empty_id(
        self, mock_list: MagicMock, mock_token_limit: MagicMock,
        mock_typer: MagicMock, mock_err: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model.id = "same-name"
        mock_model.name = "same-name"
        mock_model.alias = ""
        mock_model.source = "blablador"
        mock_model.description = "Fallback model"
        mock_list.return_value = [mock_model]

        _models_impl()

        echo_calls = [str(c) for c in mock_typer.echo.call_args_list]
        assert any("same-name" in c for c in echo_calls)


class TestCheckImpl:
    """Tests for _check_impl."""

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.client.check_model_availability", return_value=True)
    def test_available_model(self, mock_check: MagicMock, mock_typer: MagicMock, mock_err: MagicMock) -> None:
        _check_impl("openai:gpt-4o")
        mock_check.assert_called_once_with("openai:gpt-4o")
        echo_msgs = [str(c) for c in mock_typer.echo.call_args_list]
        assert any("available" in m.lower() for m in echo_msgs)

    @patch("hellmholtz.client.check_model_availability", return_value=False)
    def test_unavailable_model_exits(self, mock_check: MagicMock) -> None:
        with pytest.raises(typer.Exit) as exc_info:
            _check_impl("bad:model")
        assert exc_info.value.exit_code == 1

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.client.check_model_availability", side_effect=ConnectionError("timeout"))
    def test_check_exception_calls_handle_error(
        self, mock_check: MagicMock, mock_typer: MagicMock, mock_err: MagicMock,
    ) -> None:
        _check_impl("model:test")
        mock_err.assert_called_once()


class TestMonitorImpl:
    """Tests for _monitor_impl."""

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.monitoring.ModelAvailabilityMonitor")
    def test_monitor_without_save(self, MockMonitor: MagicMock, mock_typer: MagicMock, mock_err: MagicMock) -> None:
        instance = MockMonitor.return_value
        instance.analyze_availability.return_value = {"status": "ok"}
        instance.generate_report.return_value = "Report text"

        _monitor_impl(test_accessibility=False, save_report=False)

        instance.analyze_availability.assert_called_once_with(test_accessibility=False)
        instance.generate_report.assert_called_once()
        instance.save_report.assert_not_called()

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.monitoring.ModelAvailabilityMonitor")
    def test_monitor_with_save(self, MockMonitor: MagicMock, mock_typer: MagicMock, mock_err: MagicMock) -> None:
        instance = MockMonitor.return_value
        instance.analyze_availability.return_value = {"status": "ok"}
        instance.generate_report.return_value = "Report text"
        instance.save_report.return_value = "/tmp/report.md"

        _monitor_impl(test_accessibility=True, save_report=True)

        instance.save_report.assert_called_once_with("Report text")

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.monitoring.ModelAvailabilityMonitor", side_effect=RuntimeError("fail"))
    def test_monitor_error(self, MockMonitor: MagicMock, mock_err: MagicMock) -> None:
        _monitor_impl(False, True)
        mock_err.assert_called_once()


class TestAutoAgentImpl:
    """Tests for _auto_agent_impl."""

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.cli.models.typer")
    @patch("hellmholtz.monitoring.ModelAvailabilityMonitor")
    def test_auto_agent_success(self, MockMonitor: MagicMock, mock_typer: MagicMock, mock_err: MagicMock) -> None:
        instance = MockMonitor.return_value
        instance.run_auto_config_agent.return_value = {
            "report": "Agent report",
            "report_path": "/tmp/agent.md",
        }

        _auto_agent_impl(test_accessibility=True)

        instance.run_auto_config_agent.assert_called_once_with(test_accessibility=True)
        echo_msgs = [str(c) for c in mock_typer.echo.call_args_list]
        assert any("Agent report" in m for m in echo_msgs)

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.monitoring.ModelAvailabilityMonitor", side_effect=Exception("boom"))
    def test_auto_agent_error(self, MockMonitor: MagicMock, mock_err: MagicMock) -> None:
        _auto_agent_impl(False)
        mock_err.assert_called_once()
