"""Tests for `hellm doctor` and `hellm proxy --print-claude-settings`."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hellmholtz.cli import app
from hellmholtz.core.config import Settings
from hellmholtz.diagnostics import CheckResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _settings() -> Settings:
    return Settings(
        default_models=["blablador:alias-fast"],
        blablador_api_key="glpa-test-key-value",
        blablador_base_url="https://api.example.de/v1",
    )


def _models_body(*ids: str) -> str:
    return json.dumps({"object": "list", "data": [{"id": i} for i in ids]})


def _isolate(monkeypatch: pytest.MonkeyPatch, body: str = "") -> None:
    """Neutralize outbound HTTP and the settings/env side channels doctor touches."""
    fake = lambda url, timeout, headers=None: body  # noqa: E731
    monkeypatch.setattr("hellmholtz.cli.doctor._http_get_with_headers", fake)
    monkeypatch.setattr("hellmholtz.diagnostics._http_get_with_headers", fake)
    monkeypatch.setattr("hellmholtz.mcp.tools.get_settings", _settings)
    monkeypatch.delenv("HELLM_MCP_MODEL", raising=False)


class TestDoctorCLI:
    def test_healthy_report(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hellmholtz.core.config.get_settings", _settings)
        _isolate(monkeypatch, _models_body("alias-fast"))
        monkeypatch.setattr(
            "hellmholtz.cli.doctor.check_chat",
            lambda *a, **k: CheckResult("chat round-trip", True, "model responded"),
        )
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "All checks passed" in result.stdout
        assert "model responded" in result.stdout

    def test_skip_chat(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hellmholtz.core.config.get_settings", _settings)
        _isolate(monkeypatch, _models_body("alias-fast"))
        result = runner.invoke(app, ["doctor", "--skip-chat"])
        assert result.exit_code == 0
        assert "skipped (--skip-chat)" in result.stdout

    def test_unknown_model_warns_but_exits_zero(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("hellmholtz.core.config.get_settings", _settings)
        _isolate(monkeypatch, _models_body("alias-fast"))
        monkeypatch.setattr(
            "hellmholtz.cli.doctor.check_chat",
            lambda *a, **k: CheckResult(
                "chat round-trip", False, "HTTP 404 (model id not found upstream)", warn=True
            ),
        )
        result = runner.invoke(app, ["doctor", "--model", "blablador:ghost"])
        assert result.exit_code == 0
        assert "WARN" in result.stdout
        assert "Healthy with warnings" in result.stdout

    def test_missing_key_fails(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "hellmholtz.core.config.get_settings",
            lambda: Settings(blablador_api_key=None, blablador_base_url=None),
        )
        result = runner.invoke(app, ["doctor", "--skip-chat"])
        assert result.exit_code == 1
        assert "FAIL" in result.stdout
        assert "check(s) failed" in result.stdout

    def test_key_value_never_printed(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("hellmholtz.core.config.get_settings", _settings)
        _isolate(monkeypatch, _models_body("alias-fast"))
        result = runner.invoke(app, ["doctor", "--skip-chat"])
        assert "glpa-test-key-value" not in result.stdout


class TestPrintClaudeSettings:
    def _settings_json(self, runner: CliRunner, args: list[str]) -> dict[str, object]:
        result = runner.invoke(
            app, ["proxy", "blablador:alias-fast", "--print-claude-settings", *args]
        )
        assert result.exit_code == 0
        return json.loads(result.stdout)

    def test_prints_valid_settings_json_with_static_key(self, runner: CliRunner) -> None:
        payload = self._settings_json(
            runner, ["--name", "claude", "--master-key", "sk-test1234567890"]
        )
        env = payload["env"]
        assert isinstance(env, dict)
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4000"
        assert env["ANTHROPIC_MODEL"] == "claude"
        assert payload["apiKeyHelper"] == "echo sk-test1234567890"

    def test_key_from_env_uses_helper_reference(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-from-env-1234567890")
        payload = self._settings_json(runner, [])
        assert payload["apiKeyHelper"] == 'echo "$LITELLM_MASTER_KEY"'

    def test_no_proxy_side_effects(self, runner: CliRunner) -> None:
        # Command must exit before start_proxy is ever reached.
        import hellmholtz.integrations.litellm as litellm

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            litellm, "start_proxy", lambda *a, **k: pytest.fail("start_proxy must not run")
        )
        result = runner.invoke(
            app,
            [
                "proxy", "blablador:alias-fast", "--print-claude-settings",
                "--master-key", "sk-x1234567890",
            ],
        )
        monkeypatch.undo()
        assert result.exit_code == 0


class TestClaudeCodeSettingsFunction:
    def test_json_shape_and_indent(self) -> None:
        from hellmholtz.integrations.litellm import claude_code_settings

        raw = claude_code_settings("0.0.0.0", 4000, "claude", "sk-abc123456789", False)
        payload = json.loads(raw)  # must be valid JSON
        assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://0.0.0.0:4000"
        assert payload["env"]["ANTHROPIC_MODEL"] == "claude"
        assert payload["apiKeyHelper"] == "echo sk-abc123456789"
        assert "\n  " in raw  # pretty-printed for direct paste

    def test_env_helper_variant(self) -> None:
        from hellmholtz.integrations.litellm import claude_code_settings

        raw = claude_code_settings("127.0.0.1", 4000, "m", "sk-whatever", True)
        assert json.loads(raw)["apiKeyHelper"] == 'echo "$LITELLM_MASTER_KEY"'
