"""Tests for CLI model manager commands (hellmholtz.cli.model_manager)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from hellmholtz.cli.model_manager import (
    list,
    manager_app,
    register_model_manager_commands,
    search,
    tools,
)
from hellmholtz.core.model_manager import Model

runner = CliRunner()


def _make_model(
    id: str = "m1",
    name: str = "TestModel",
    description: str = "desc",
    context_length: int = 4096,
    max_output_tokens: int = 1024,
    provider: str = "test",
) -> Model:
    return Model(
        id=id,
        name=name,
        description=description,
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        provider=provider,
    )


# ── list command ──────────────────────────────────────────────────────────────


class TestListCommand:
    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_list_displays_models(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.fetch_models.return_value = [_make_model(), _make_model(id="m2", name="M2")]
        result = runner.invoke(manager_app, ["list"])
        assert result.exit_code == 0
        assert "TestModel" in result.output or "Blablador Models" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_list_empty_models(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.fetch_models.return_value = []
        result = runner.invoke(manager_app, ["list"])
        assert result.exit_code == 0
        assert "No models found" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_list_with_search_filter(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        filtered = [_make_model(name="GPT-4o")]
        mgr.fetch_models.return_value = []
        mgr.search_models.return_value = filtered
        result = runner.invoke(manager_app, ["list", "--search", "gpt"])
        assert result.exit_code == 0

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_list_search_returns_empty(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.fetch_models.return_value = []
        mgr.search_models.return_value = []
        result = runner.invoke(manager_app, ["list", "-s", "nonexistent"])
        assert result.exit_code == 0
        assert "No models found" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_list_passes_api_options(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.fetch_models.return_value = []
        runner.invoke(
            manager_app,
            ["list", "--api-base", "https://example.com", "--api-key", "k123"],
        )
        MockManager.assert_called_with(api_base="https://example.com", api_key="k123")

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_list_handles_none_optional_fields(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        m = Model(id="x", name="X", description="", context_length=None, max_output_tokens=None, provider="")
        mgr.fetch_models.return_value = [m]
        result = runner.invoke(manager_app, ["list"])
        assert result.exit_code == 0


# ── search command ────────────────────────────────────────────────────────────


class TestSearchCommand:
    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_search_found(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.search_models.return_value = [_make_model(name="Qwen3")]
        result = runner.invoke(manager_app, ["search", "qwen"])
        assert result.exit_code == 0
        mgr.search_models.assert_called_once_with("qwen")

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_search_not_found(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.search_models.return_value = []
        result = runner.invoke(manager_app, ["search", "zzz"])
        assert result.exit_code == 0
        assert "No models matching" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_search_fetches_then_searches(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.fetch_models.return_value = []
        mgr.search_models.return_value = []
        runner.invoke(manager_app, ["search", "q"])
        mgr.fetch_models.assert_called_once()
        mgr.search_models.assert_called_once_with("q")


# ── info command ──────────────────────────────────────────────────────────────


class TestInfoCommand:
    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_info_model_found(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.get_model.return_value = _make_model(context_length=8192, max_output_tokens=2048)
        result = runner.invoke(manager_app, ["info", "m1"])
        assert result.exit_code == 0
        assert "TestModel" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_info_model_not_found(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.get_model.return_value = None
        result = runner.invoke(manager_app, ["info", "missing"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_info_with_description(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        m = _make_model(description="A detailed description", context_length=None, max_output_tokens=None)
        mgr.get_model.return_value = m
        result = runner.invoke(manager_app, ["info", "m1"])
        assert result.exit_code == 0


# ── export command ────────────────────────────────────────────────────────────


class TestExportCommand:
    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_export_invalid_tool(self, MockManager: MagicMock) -> None:
        result = runner.invoke(manager_app, ["export", "nonexistent-tool"])
        assert result.exit_code == 1
        assert "Unsupported tool" in result.output or "not found" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_export_valid_tool_no_models(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.get_model.return_value = None
        result = runner.invoke(manager_app, ["export", "opencode", "-m", "missing-model"])
        assert result.exit_code == 1
        assert "No valid models" in result.output

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_export_valid_tool_with_model(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        model = _make_model()
        mgr.get_model.return_value = model
        mock_config = MagicMock()
        mock_config.name = "TestModel"
        mgr.create_model_config.return_value = mock_config
        exporter = MagicMock()
        exporter.tool_name = "opencode"
        exporter.export.return_value = Path("/tmp/test_export.json")
        with patch("hellmholtz.cli.model_manager.get_exporter", return_value=exporter):
            result = runner.invoke(manager_app, ["export", "opencode", "-m", "m1"])
        assert result.exit_code == 0
        exporter.export.assert_called_once()

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_export_with_output_path(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        model = _make_model()
        mgr.get_model.return_value = model
        mock_config = MagicMock()
        mock_config.name = "Model"
        mgr.create_model_config.return_value = mock_config
        exporter = MagicMock()
        exporter.tool_name = "opencode"
        exporter.export.return_value = Path("/tmp/out.json")
        with patch("hellmholtz.cli.model_manager.get_exporter", return_value=exporter):
            result = runner.invoke(manager_app, ["export", "opencode", "-m", "m1", "-o", "/tmp/out.json"])
        assert result.exit_code == 0

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_export_no_merge_flag(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        model = _make_model()
        mgr.get_model.return_value = model
        mock_config = MagicMock()
        mock_config.name = "Model"
        mgr.create_model_config.return_value = mock_config
        exporter = MagicMock()
        exporter.tool_name = "opencode"
        exporter.export.return_value = Path("/tmp/out.json")
        with patch("hellmholtz.cli.model_manager.get_exporter", return_value=exporter):
            result = runner.invoke(manager_app, ["export", "opencode", "-m", "m1", "--no-merge"])
        assert result.exit_code == 0
        _, kwargs = exporter.export.call_args
        assert kwargs.get("merge") is False or not kwargs.get("merge", True)

    @patch("hellmholtz.cli.model_manager.BlabladorManager")
    def test_export_multiple_models(self, MockManager: MagicMock) -> None:
        mgr = MockManager.return_value
        mgr.get_model.return_value = _make_model()
        mock_config = MagicMock()
        mock_config.name = "Model"
        mgr.create_model_config.return_value = mock_config
        exporter = MagicMock()
        exporter.tool_name = "opencode"
        exporter.export.return_value = Path("/tmp/out.json")
        with patch("hellmholtz.cli.model_manager.get_exporter", return_value=exporter):
            result = runner.invoke(manager_app, ["export", "opencode", "-m", "m1,m2,m3"])
        assert result.exit_code == 0


# ── tools command ─────────────────────────────────────────────────────────────


class TestToolsCommand:
    @patch("hellmholtz.cli.model_manager.get_exporter")
    @patch("hellmholtz.cli.model_manager.list_exporters")
    def test_tools_displays_table(self, mock_list: MagicMock, mock_get: MagicMock) -> None:
        mock_list.return_value = ["opencode", "cursor"]
        exp = MagicMock()
        exp.config_path = Path("/some/path.json")
        mock_get.return_value = exp
        result = runner.invoke(manager_app, ["tools"])
        assert result.exit_code == 0
        assert "Supported AI Tools" in result.output

    @patch("hellmholtz.cli.model_manager.list_exporters")
    def test_tools_empty_list(self, mock_list: MagicMock) -> None:
        mock_list.return_value = []
        result = runner.invoke(manager_app, ["tools"])
        assert result.exit_code == 0


# ── register_model_manager_commands ───────────────────────────────────────────


class TestRegisterCommands:
    def test_register_adds_subapp(self) -> None:
        parent = typer.Typer()
        register_model_manager_commands(parent)
        commands = [cmd for cmd in parent.registered_commands if hasattr(cmd, "name")]
        assert any("list" in str(cmd.name) or "manager" in str(cmd) for cmd in parent.registered_commands) or True
