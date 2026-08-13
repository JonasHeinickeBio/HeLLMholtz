"""Tests for litellm integration (hellmholtz.integrations.litellm)."""

from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.integrations.litellm import start_proxy


class TestStartProxy:
    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_success_default_args(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        start_proxy("gpt-4")
        mock_subprocess.run.assert_called_once_with(
            ["litellm", "--model", "gpt-4", "--port", "4000"],
            check=True,
        )

    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_success_custom_port(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        start_proxy("gpt-4", port=8080)
        cmd = mock_subprocess.run.call_args[0][0]
        assert "--port" in cmd
        assert "8080" in cmd

    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_success_with_config_path(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        start_proxy("gpt-4", config_path="/tmp/config.yaml")
        cmd = mock_subprocess.run.call_args[0][0]
        assert "--config" in cmd
        assert "/tmp/config.yaml" in cmd

    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_success_with_debug(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        start_proxy("gpt-4", debug=True)
        cmd = mock_subprocess.run.call_args[0][0]
        assert "--debug" in cmd

    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_success_no_debug_no_config(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        start_proxy("gpt-4", debug=False, config_path=None)
        cmd = mock_subprocess.run.call_args[0][0]
        assert "--debug" not in cmd
        assert "--config" not in cmd

    @patch("hellmholtz.integrations.litellm.sys")
    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_file_not_found_error(self, mock_subprocess: MagicMock, mock_sys: MagicMock) -> None:
        mock_subprocess.run.side_effect = FileNotFoundError("litellm not found")
        start_proxy("gpt-4")
        mock_sys.exit.assert_called_once_with(1)

    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_keyboard_interrupt(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.side_effect = KeyboardInterrupt()
        start_proxy("gpt-4")
        mock_subprocess.run.assert_called_once()

    @patch("hellmholtz.integrations.litellm.subprocess")
    def test_model_and_port_in_cmd(self, mock_subprocess: MagicMock) -> None:
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        start_proxy("claude-3-opus", port=9090)
        cmd = mock_subprocess.run.call_args[0][0]
        assert cmd[0] == "litellm"
        assert cmd[1] == "--model"
        assert cmd[2] == "claude-3-opus"
        assert cmd[3] == "--port"
        assert cmd[4] == "9090"
