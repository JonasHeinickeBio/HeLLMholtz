"""
Tests for the HeLLMholtz MCP server (work item 2 of issue #40).

Covers model resolution, the Claude Desktop config snippet, tool behavior
with a mocked chat backend, and the CLI command wiring. The end-to-end
test runs real MCP tool calls through the in-memory session.
"""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hellmholtz.cli import app
from hellmholtz.mcp.tools import (
    DEFAULT_MODEL_ENV_VAR,
    FALLBACK_MODEL,
    HellmTools,
    claude_desktop_config,
    resolve_default_model,
)

try:  # optional extra: guard against missing *or* broken installs
    import mcp.shared.memory  # noqa: F401

    HAS_MCP = True
except ImportError:
    HAS_MCP = False


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host env from leaking into resolution tests."""
    monkeypatch.delenv(DEFAULT_MODEL_ENV_VAR, raising=False)
    monkeypatch.delenv("AISUITE_DEFAULT_MODELS", raising=False)


class TestResolveDefaultModel:
    """Test suite for resolve_default_model."""

    def test_explicit_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DEFAULT_MODEL_ENV_VAR, "blablador:Env")
        assert resolve_default_model("blablador:Explicit") == "blablador:Explicit"

    def test_env_beats_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DEFAULT_MODEL_ENV_VAR, "blablador:Env")
        assert resolve_default_model(None) == "blablador:Env"

    def test_settings_default_models_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AISUITE_DEFAULT_MODELS", "openai:gpt-4o, other:model")
        assert resolve_default_model(None) == "openai:gpt-4o"

    def test_fallback(self) -> None:
        assert resolve_default_model(None) == FALLBACK_MODEL


class TestClaudeDesktopConfig:
    """Test suite for claude_desktop_config."""

    def test_minimal_config(self) -> None:
        cfg = json.loads(claude_desktop_config())
        server = cfg["mcpServers"]["hellmholtz"]
        assert server["command"] == "hellm"
        assert server["args"] == ["mcp"]

    def test_config_with_model_and_args(self) -> None:
        cfg = json.loads(
            claude_desktop_config(model="blablador:M", args=["--transport", "stdio"])
        )
        server = cfg["mcpServers"]["hellmholtz"]
        assert server["args"] == ["mcp", "--model", "blablador:M", "--transport", "stdio"]


class TestHellmTools:
    """Tool behavior with mocked chat/catalog backends."""

    def test_ask_external_sends_messages(self) -> None:
        tools = HellmTools(default_model="blablador:M")
        with patch("hellmholtz.client.chat", return_value="answer") as chat:
            out = tools.ask_external("hi", system="be nice", temperature=0.2, max_tokens=5)
        assert out == "answer"
        model, messages = chat.call_args.args
        kwargs = chat.call_args.kwargs
        assert model == "blablador:M"
        assert messages == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        assert kwargs == {"temperature": 0.2, "max_tokens": 5}

    def test_ask_external_empty_prompt(self) -> None:
        tools = HellmTools()
        assert tools.ask_external("   ").startswith("Error:")

    def test_ask_external_error_becomes_text(self) -> None:
        tools = HellmTools(default_model="blablador:M")
        with patch("hellmholtz.client.chat", side_effect=RuntimeError("boom")):
            out = tools.ask_external("hi")
        assert "Error calling model 'blablador:M'" in out
        assert "boom" in out

    def test_chat_external_valid_json(self) -> None:
        tools = HellmTools(default_model="blablador:M")
        messages = json.dumps(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        )
        with patch("hellmholtz.client.chat", return_value="ok") as chat:
            out = tools.chat_external(messages)
        assert out == "ok"
        assert chat.call_args.args[1] == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    @pytest.mark.parametrize(
        "bad",
        [
            "not json",
            "[]",
            '"just a string"',
            json.dumps([{"role": "hacker", "content": "x"}]),
            json.dumps([{"role": "user", "content": 42}]),
        ],
    )
    def test_chat_external_rejects_bad_input(self, bad: str) -> None:
        tools = HellmTools()
        assert tools.chat_external(bad).startswith("Error:")

    def test_list_models_formats_catalog(self) -> None:
        class M:
            name = "Model A"
            id = "7"

        tools = HellmTools()
        with patch("hellmholtz.providers.blablador.list_models", return_value=[M()]):
            out = tools.list_models()
        assert out == "blablador:Model A"

    def test_list_models_error_becomes_text(self) -> None:
        tools = HellmTools()
        with patch(
            "hellmholtz.providers.blablador.list_models",
            side_effect=ValueError("no key"),
        ):
            out = tools.list_models()
        assert out.startswith("Error listing models:")

    def test_get_info_reports_default_model(self) -> None:
        tools = HellmTools(default_model="blablador:M")
        assert "blablador:M" in tools.get_info()


class TestCliMcpCommand:
    """CLI wiring for `hellm mcp`."""

    def test_print_config(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "--print-config", "--model", "blablador:M"])
        assert result.exit_code == 0
        cfg = json.loads(result.stdout)
        assert cfg["mcpServers"]["hellmholtz"]["args"] == ["mcp", "--model", "blablador:M"]

    def test_registered_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert "mcp" in result.stdout


@pytest.mark.skipif(
    not HAS_MCP,
    reason="requires the optional 'mcp' extra (pip install 'hellmholtz[mcp]')",
)
class TestServerEndToEnd:
    """Real MCP protocol round-trip through the in-memory session."""

    def test_list_and_call_tools(self) -> None:
        import asyncio

        from hellmholtz.mcp.server import create_server

        server = create_server(default_model="blablador:M")

        async def run() -> None:
            from mcp.shared.memory import create_connected_server_and_client_session

            async with create_connected_server_and_client_session(server) as client:
                names = {t.name for t in (await client.list_tools()).tools}
                assert names == {"ask_external", "chat_external", "list_models", "get_info"}

                with patch("hellmholtz.client.chat", return_value="pong") as chat:
                    result = await client.call_tool("ask_external", {"prompt": "ping"})
                    text = result.content[0].text
                    assert text == "pong"
                    assert chat.call_args.args[0] == "blablador:M"

                bad = await client.call_tool("ask_external", {"prompt": "  "})
                assert bad.content[0].text.startswith("Error:")

        asyncio.run(run())
