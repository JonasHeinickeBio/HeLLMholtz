"""
Tests for the HeLLMholtz MCP server (work item 2 of issue #40).

Covers model resolution, the Claude Desktop config snippet, tool behavior
with a mocked chat backend, and the CLI command wiring. The end-to-end
test runs real MCP tool calls through the in-memory session.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from hellmholtz.cli import app
from hellmholtz.mcp.tools import (
    DEFAULT_MODEL_ENV_VAR,
    FALLBACK_MODEL,
    HellmTools,
    claude_desktop_config,
    prompt_explain,
    prompt_summarize,
    prompt_translate,
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

    def test_check_model_available(self) -> None:
        tools = HellmTools(default_model="blablador:M")
        with patch("hellmholtz.client.check_model_availability", return_value=True) as probe:
            out = tools.check_model()
        assert out == "Model 'blablador:M' is available."
        probe.assert_called_once_with("blablador:M")

    def test_check_model_unavailable(self) -> None:
        tools = HellmTools()
        with patch("hellmholtz.client.check_model_availability", return_value=False):
            out = tools.check_model("blablador:Ghost")
        assert "NOT available" in out
        assert "blablador:Ghost" in out

    def test_check_model_error_becomes_text(self) -> None:
        tools = HellmTools()
        with patch(
            "hellmholtz.client.check_model_availability",
            side_effect=RuntimeError("offline"),
        ):
            out = tools.check_model("blablador:M")
        assert out.startswith("Error checking model 'blablador:M'")
        assert "offline" in out

    def test_models_json_catalog(self) -> None:
        class M:
            name = "Model A"
            id = "7"

        tools = HellmTools()
        with patch("hellmholtz.providers.blablador.list_models", return_value=[M()]):
            payload = json.loads(tools.models_json())
        assert payload == {
            "object": "list",
            "models": [{"id": "blablador:Model A", "name": "Model A", "upstream_id": "7"}],
        }

    def test_models_json_error_becomes_object(self) -> None:
        tools = HellmTools()
        with patch(
            "hellmholtz.providers.blablador.list_models",
            side_effect=ValueError("no key"),
        ):
            payload = json.loads(tools.models_json())
        assert payload["models"] == []
        assert "no key" in payload["error"]


class TestRunDoctor:
    """run_doctor reuses the hellm doctor checks and formats a report."""

    @staticmethod
    def _settings() -> SimpleNamespace:
        return SimpleNamespace(
            blablador_api_key="sk-test-123456",
            blablador_base_url="https://api.example.com/v1",
        )

    def test_report_with_all_ok(self) -> None:
        tools = HellmTools(default_model="blablador:alias-large")
        models_body = json.dumps({"data": [{"id": "alias-large"}]})
        with (
            patch("hellmholtz.mcp.tools.get_settings", return_value=self._settings()),
            patch("hellmholtz.diagnostics._http_get_with_headers", return_value=models_body),
            patch("hellmholtz.diagnostics._http_post_json", return_value="{}"),
        ):
            out = tools.run_doctor()
        assert out.startswith("hellm doctor report (model: blablador:alias-large)")
        assert "[OK] credentials:" in out
        assert "[OK] endpoint:" in out
        assert "[OK] model:" in out
        assert "[OK] chat round-trip:" in out
        assert out.endswith("All checks passed.")

    def test_skip_chat_and_failed_endpoint(self) -> None:
        tools = HellmTools(default_model="blablador:M")
        with (
            patch("hellmholtz.mcp.tools.get_settings", return_value=self._settings()),
            patch(
                "hellmholtz.diagnostics._http_get_with_headers",
                side_effect=OSError("refused"),
            ),
        ):
            out = tools.run_doctor(skip_chat=True)
        assert "[FAIL] endpoint:" in out
        assert "refused" in out
        assert "[OK] model:" not in out  # model listing skipped when endpoint failed
        assert "[OK] chat round-trip: skipped" in out
        assert "check(s) failed." in out


class TestPromptBuilders:
    """MCP prompt builders produce ready-to-send chat messages."""

    def test_summarize_without_style(self) -> None:
        (msg,) = prompt_summarize("long text here")
        assert msg["role"] == "user"
        assert "Summarize" in msg["content"]
        assert msg["content"].endswith("long text here")
        assert "style/audience" not in msg["content"]

    def test_summarize_with_style(self) -> None:
        (msg,) = prompt_summarize("paper abstract", style="a 12-year-old")
        assert "a 12-year-old" in msg["content"]

    def test_translate(self) -> None:
        (msg,) = prompt_translate("good morning", "German")
        assert "into German" in msg["content"]
        assert msg["content"].endswith("good morning")

    def test_explain_audience_default_and_override(self) -> None:
        (default_msg,) = prompt_explain("quantum tunneling is...")
        assert "technically literate reader" in default_msg["content"]
        (custom_msg,) = prompt_explain("quantum tunneling is...", audience="clinicians")
        assert "for clinicians" in custom_msg["content"]


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
                assert names == {
                    "ask_external",
                    "chat_external",
                    "check_model",
                    "run_doctor",
                    "list_models",
                    "get_info",
                }

                with patch("hellmholtz.client.chat", return_value="pong") as chat:
                    result = await client.call_tool("ask_external", {"prompt": "ping"})
                    text = result.content[0].text
                    assert text == "pong"
                    assert chat.call_args.args[0] == "blablador:M"

                bad = await client.call_tool("ask_external", {"prompt": "  "})
                assert bad.content[0].text.startswith("Error:")

        asyncio.run(run())

    def test_resources_and_prompts(self) -> None:
        import asyncio

        from hellmholtz.mcp.server import create_server

        server = create_server(default_model="blablador:M")

        async def run() -> None:
            from mcp.shared.memory import create_connected_server_and_client_session

            async with create_connected_server_and_client_session(server) as client:
                uris = {str(r.uri) for r in (await client.list_resources()).resources}
                assert uris == {"hellm://info", "hellm://models"}

                info = await client.read_resource("hellm://info")
                assert "blablador:M" in info.contents[0].text

                with patch(
                    "hellmholtz.providers.blablador.list_models",
                    return_value=[],
                ):
                    models = await client.read_resource("hellm://models")
                assert json.loads(models.contents[0].text)["object"] == "list"

                names = {p.name for p in (await client.list_prompts()).prompts}
                assert names == {"summarize", "translate", "explain"}

                rendered = await client.get_prompt(
                    "translate", {"text": "hello", "target_language": "German"}
                )
                content = rendered.messages[0].content
                assert "into German" in content.text
                assert content.text.endswith("hello")

        asyncio.run(run())
