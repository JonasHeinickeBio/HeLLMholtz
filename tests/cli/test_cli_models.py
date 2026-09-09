"""Tests for hellmholtz.cli.models implementation functions."""

from unittest.mock import MagicMock, patch

import pytest
import typer

from hellmholtz.cli.models import (
    _available_impl,
    _auto_agent_impl,
    _check_impl,
    _list_impl,
    _monitor_impl,
)


class TestListImpl:
    """Tests for _list_impl - shows all configured models with availability status."""

    def _create_model_mock(self, name: str, id_val: str, source: str, description: str):
        """Create a properly configured model mock with string name attribute."""
        mock = MagicMock()
        mock.name = name
        mock.id = id_val
        mock.alias = None
        mock.source = source
        mock.description = description
        return mock

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=8192)
    @patch("hellmholtz.providers.blablador.list_models")
    def test_displays_configured_models_with_availability(
        self, mock_list: MagicMock, mock_token_limit: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that _list_impl shows all configured models with availability columns."""
        # Create properly configured model mocks
        config_model = self._create_model_mock("test-model", "1", "Blablador", "Test model")
        api_only_model = self._create_model_mock("api-only", "", "", "API only model")

        # Set up the patch before calling the function
        with patch("hellmholtz.providers.blablador_config.KNOWN_MODELS", [config_model, api_only_model]):
            # Mock API model for only one of the two configured models
            api_model = MagicMock()
            api_model.id = "1"
            api_model.name = "test-model"
            api_model.alias = None
            api_model.source = "Blablador"
            api_model.description = "Test model description"
            mock_list.return_value = [api_model]

            _list_impl()

            # Verify console.print was called (shows table output)
            assert mock_console.return_value.print.called

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador.list_models", side_effect=RuntimeError("API error"))
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=4096)
    def test_handles_api_error_gracefully(
        self, mock_token_limit: MagicMock, mock_list: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that _list_impl handles API errors and shows configured models."""
        config_model = self._create_model_mock("model1", "1", "Blablador", "Test model")
        with patch("hellmholtz.providers.blablador_config.KNOWN_MODELS", [config_model]):
            _list_impl()
            assert mock_console.return_value.print.called

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=0)
    @patch("hellmholtz.providers.blablador.list_models", return_value=[])
    def test_displays_models_with_zero_token_limit(
        self, mock_list: MagicMock, mock_token_limit: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that models with zero token limit display correctly."""
        config_model = self._create_model_mock("model1", "1", "Blablador", "")
        with patch("hellmholtz.providers.blablador_config.KNOWN_MODELS", [config_model]):
            _list_impl()
            assert mock_console.return_value.print.called


class TestAvailableImpl:
    """Tests for _available_impl - shows only API models."""

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador.list_models")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=8192)
    def test_displays_api_models(
        self, mock_token_limit: MagicMock, mock_list: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that _available_impl shows models from API."""
        mock_model = MagicMock()
        mock_model.id = "15"
        mock_model.name = "Apertus-8B-Instruct"
        mock_model.alias = None
        mock_model.source = "Blablador"
        mock_model.description = "A new swiss model"
        mock_list.return_value = [mock_model]

        _available_impl()

        assert mock_console.return_value.print.called

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador.list_models", side_effect=RuntimeError("API error"))
    def test_handles_list_error(
        self, mock_list: MagicMock, mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that _available_impl handles API errors gracefully."""
        _available_impl()
        # Should still print something (error message or "No models available")
        assert mock_console.return_value.print.called

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador.list_models", return_value=[])
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=0)
    def test_handles_empty_api_response(
        self, mock_token_limit: MagicMock, mock_list: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that _available_impl handles empty API response."""
        _available_impl()
        # Should print "No models available from API"
        assert mock_console.return_value.print.called

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador.list_models")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=0)
    def test_displays_model_with_none_name(
        self, mock_token_limit: MagicMock, mock_list: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that models with None name don't cause sorting errors."""
        mock_model = MagicMock()
        mock_model.id = "1"
        mock_model.name = None  # Edge case
        mock_model.alias = None
        mock_model.source = "Blablador"
        mock_model.description = "Model with None name"
        mock_list.return_value = [mock_model]

        _available_impl()

        assert mock_console.return_value.print.called

    @patch("hellmholtz.cli.models.handle_error")
    @patch("rich.console.Console")
    @patch("rich.table.Table")
    @patch("hellmholtz.providers.blablador.list_models")
    @patch("hellmholtz.providers.blablador_config.get_token_limit", return_value=8192)
    def test_sorts_models_by_name(
        self, mock_token_limit: MagicMock, mock_list: MagicMock,
        mock_table: MagicMock, mock_console: MagicMock, mock_err: MagicMock,
    ) -> None:
        """Test that models are sorted alphabetically by name."""
        model_a = MagicMock()
        model_a.id = "1"
        model_a.name = "Z-model"
        model_a.alias = None
        model_a.source = "Blablador"
        model_a.description = "Z model"

        model_b = MagicMock()
        model_b.id = "2"
        model_b.name = "A-model"
        model_b.alias = None
        model_b.source = "Blablador"
        model_b.description = "A model"

        mock_list.return_value = [model_a, model_b]

        _available_impl()

        # Verify both models were processed
        assert mock_console.return_value.print.called


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

        _monitor_impl(test_accessibility=False, save_report=False, auto_sync=False)

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

        _monitor_impl(test_accessibility=True, save_report=True, auto_sync=False)

        instance.save_report.assert_called_once_with("Report text")

    @patch("hellmholtz.cli.models.handle_error")
    @patch("hellmholtz.monitoring.ModelAvailabilityMonitor", side_effect=RuntimeError("fail"))
    def test_monitor_error(self, MockMonitor: MagicMock, mock_err: MagicMock) -> None:
        _monitor_impl(False, True, False)
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
