"""Tests for `hellm ollama` and `hellm chat --fallback`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hellmholtz.cli import app
from hellmholtz.client import FallbackError, FallbackResult
from hellmholtz.providers.ollama import OllamaUnavailableError


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestOllamaStatus:
    def test_running(self, runner: CliRunner) -> None:
        with patch("hellmholtz.cli.ollama.ollama.list_models", return_value=["a", "b"]):
            result = runner.invoke(app, ["ollama", "status"])
        assert result.exit_code == 0
        assert "running" in result.output
        assert "2 model(s)" in result.output

    def test_unavailable(self, runner: CliRunner) -> None:
        with patch(
            "hellmholtz.cli.ollama.ollama.list_models",
            side_effect=OllamaUnavailableError("not reachable"),
        ):
            result = runner.invoke(app, ["ollama", "status"])
        assert result.exit_code == 1
        assert "unavailable" in result.output


class TestOllamaModels:
    def test_lists_qualified(self, runner: CliRunner) -> None:
        with patch("hellmholtz.cli.ollama.ollama.list_models", return_value=["llama3.2:3b"]):
            result = runner.invoke(app, ["ollama", "models"])
        assert result.exit_code == 0
        assert "ollama:llama3.2:3b" in result.output

    def test_error(self, runner: CliRunner) -> None:
        with patch(
            "hellmholtz.cli.ollama.ollama.list_models",
            side_effect=OllamaUnavailableError("down"),
        ):
            result = runner.invoke(app, ["ollama", "models"])
        assert result.exit_code == 1
        assert "down" in result.output


class TestOllamaChat:
    def test_chat(self, runner: CliRunner) -> None:
        with patch("hellmholtz.cli.ollama.ollama_chat", return_value="pong") as m:
            result = runner.invoke(app, ["ollama", "chat", "ping", "--model", "llama3.2"])
        assert result.exit_code == 0
        assert "pong" in result.output
        m.assert_called_once_with("ping", model="llama3.2", temperature=0.7)

    def test_chat_error(self, runner: CliRunner) -> None:
        with patch(
            "hellmholtz.cli.ollama.ollama_chat", side_effect=OllamaUnavailableError("down")
        ):
            result = runner.invoke(app, ["ollama", "chat", "ping"])
        assert result.exit_code == 1


class TestChatFallbackFlag:
    def test_without_flag_uses_plain_chat(self, runner: CliRunner) -> None:
        with (
            patch("hellmholtz.cli.chat.chat", return_value="plain") as plain,
            patch("hellmholtz.cli.chat.chat_with_fallback_detailed") as fb,
        ):
            result = runner.invoke(app, ["chat", "--model", "a:b", "hi"])
        assert result.exit_code == 0
        assert "plain" in result.output
        plain.assert_called_once()
        fb.assert_not_called()

    def test_flag_uses_fallback_and_notes_switch(self, runner: CliRunner) -> None:
        res = FallbackResult(text="local", model="ollama:m", attempts=[("a:b", "503")])
        with patch("hellmholtz.cli.chat.chat_with_fallback_detailed", return_value=res) as fb:
            result = runner.invoke(app, ["chat", "--model", "a:b", "-f", "ollama", "-f", "m", "hi"])
        assert result.exit_code == 0
        assert "local" in result.output
        assert "answered by ollama:m" in result.output
        assert fb.call_args.kwargs["fallbacks"] == ["ollama", "m"]

    def test_flag_no_note_when_primary_answers(self, runner: CliRunner) -> None:
        res = FallbackResult(text="primary", model="a:b")
        with patch("hellmholtz.cli.chat.chat_with_fallback_detailed", return_value=res):
            result = runner.invoke(app, ["chat", "--model", "a:b", "--fallback", "ollama", "hi"])
        assert result.exit_code == 0
        assert "answered by" not in result.output

    def test_all_fail_exits_nonzero(self, runner: CliRunner) -> None:
        err = FallbackError([("a:b", "down"), ("ollama", "not reachable")])
        with patch("hellmholtz.cli.chat.chat_with_fallback_detailed", side_effect=err):
            result = runner.invoke(app, ["chat", "--model", "a:b", "-f", "ollama", "hi"])
        assert result.exit_code == 1
        assert "All models failed" in result.output
