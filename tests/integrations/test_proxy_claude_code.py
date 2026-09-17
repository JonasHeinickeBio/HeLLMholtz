"""
Tests for the LiteLLM proxy Claude Code (Anthropic-compatible) support.

Covers master key generation, config generation, snippet generation and
start_proxy command construction, including the --claude-code flow.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from hellmholtz.integrations.litellm import (
    MASTER_KEY_ENV_VAR,
    _to_config_model,
    build_proxy_config,
    claude_code_snippet,
    generate_master_key,
    start_proxy,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LITELLM_MASTER_KEY does not leak into tests."""
    monkeypatch.delenv(MASTER_KEY_ENV_VAR, raising=False)


class TestGenerateMasterKey:
    """Test suite for generate_master_key."""

    def test_key_format(self) -> None:
        """Generated keys carry the sk-hellm- prefix and random suffix."""
        key = generate_master_key()
        assert key.startswith("sk-hellm-")
        assert len(key) > len("sk-hellm-")

    def test_key_uniqueness(self) -> None:
        """Consecutive generations produce distinct keys."""
        keys = {generate_master_key() for _ in range(50)}
        assert len(keys) == 50

    def test_key_is_urlsafe(self) -> None:
        """Generated keys only contain YAML/shell-safe characters."""
        key = generate_master_key()
        suffix = key.removeprefix("sk-hellm-")
        assert all(c.isalnum() or c in "-_" for c in suffix)


class TestBuildProxyConfig:
    """Test suite for build_proxy_config."""

    def test_without_master_key(self) -> None:
        """Config without a master key omits general_settings."""
        config = build_proxy_config("openai:gpt-4o", "claude")
        assert config == (
            "model_list:\n"
            "  - model_name: claude\n"
            "    litellm_params:\n"
            "      model: openai/gpt-4o\n"
        )
        assert "master_key" not in config

    def test_with_master_key(self) -> None:
        """Config with a master key appends general_settings."""
        config = build_proxy_config("openai:gpt-4o", "claude", master_key="sk-abc")
        assert config == (
            "model_list:\n"
            "  - model_name: claude\n"
            "    litellm_params:\n"
            "      model: openai/gpt-4o\n"
            "general_settings:\n"
            "  master_key: sk-abc\n"
        )


class TestToConfigModel:
    """Test suite for _to_config_model (colon -> slash provider prefix)."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openai:gpt-4o", "openai/gpt-4o"),
            ("ollama:llama3.2", "ollama/llama3.2"),
            ("gemini:gemini-2.5-pro", "gemini/gemini-2.5-pro"),
            # Already in slash form (no colon) - unchanged.
            ("openai/gpt-4o", "openai/gpt-4o"),
            # Provider-specific model names containing no colon - unchanged.
            ("gpt-4o", "gpt-4o"),
            # Colon with empty halves - unchanged.
            (":gpt-4o", ":gpt-4o"),
            ("openai:", "openai:"),
        ],
    )
    def test_conversion(self, model: str, expected: str) -> None:
        """Colon-prefixed providers are rewritten; everything else is kept."""
        assert _to_config_model(model) == expected

    @pytest.mark.parametrize(
        ("model", "name", "key"),
        [
            ("", "claude", None),
            ("openai:gpt-4o", "", None),
            ("openai:gpt-4o", "claude", ""),
            ("openai:gpt-4o", "claude\ninjection", None),
            ("openai:gpt-4o", "claude", "bad\nkey"),
        ],
    )
    def test_rejects_invalid_values(
        self, model: str, name: str, key: str | None
    ) -> None:
        """Empty or control-character values are rejected."""
        with pytest.raises(ValueError):
            build_proxy_config(model, name, master_key=key)


class TestClaudeCodeSnippet:
    """Test suite for claude_code_snippet."""

    def test_literal_key(self) -> None:
        """Snippet embeds the literal key and unsets ANTHROPIC_API_KEY."""
        snippet = claude_code_snippet("127.0.0.1", 4000, "claude", "sk-hellm-abc")
        assert snippet == (
            "unset ANTHROPIC_API_KEY\n"
            'export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"\n'
            'export ANTHROPIC_AUTH_TOKEN="sk-hellm-abc"\n'
            'export ANTHROPIC_MODEL="claude"\n'
            "claude"
        )

    def test_key_from_env(self) -> None:
        """Snippet references the env var instead of the literal key."""
        snippet = claude_code_snippet(
            "localhost", 4100, "gpt", "unused", key_from_env=True
        )
        assert f'export ANTHROPIC_AUTH_TOKEN="${MASTER_KEY_ENV_VAR}"' in snippet
        assert "unused" not in snippet
        assert 'export ANTHROPIC_BASE_URL="http://localhost:4100"' in snippet
        assert 'export ANTHROPIC_MODEL="gpt"' in snippet


class TestStartProxy:
    """Test suite for start_proxy command construction."""

    def test_model_flag_by_default(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Without name/key the legacy --model flag is used."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", port=4000)

        cmd = mock_run.call_args.args[0]
        assert cmd == [
            "litellm",
            "--model",
            "openai:gpt-4o",
            "--port",
            "4000",
            "--host",
            "127.0.0.1",
        ]
        out = capsys.readouterr().out
        assert "Starting LiteLLM proxy for openai:gpt-4o" in out
        assert "open proxy" in out

    def test_custom_host_and_port(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Host and port are passed through to litellm."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", host="0.0.0.0", port=4100)

        cmd = mock_run.call_args.args[0]
        host_idx = cmd.index("--host")
        port_idx = cmd.index("--port")
        assert cmd[host_idx + 1] == "0.0.0.0"
        assert cmd[port_idx + 1] == "4100"
        assert "http://0.0.0.0:4100" in capsys.readouterr().out

    def test_debug_flag(self) -> None:
        """--debug is forwarded to litellm."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", debug=True)

        assert mock_run.call_args.args[0][-1] == "--debug"

    def test_explicit_config_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A user-provided config file replaces --model."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", config_path="/tmp/hellm-proxy.yaml")

        cmd = mock_run.call_args.args[0]
        assert cmd[:3] == ["litellm", "--config", "/tmp/hellm-proxy.yaml"]
        assert "--model" not in cmd
        assert "ignored" not in capsys.readouterr().out

    def test_explicit_config_ignores_name_and_key(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--name/--master-key with --config produce a warning."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy(
                "openai:gpt-4o", config_path="c.yaml", model_name="x", master_key="sk-x"
            )

        assert "--name and --master-key are ignored" in capsys.readouterr().out

    def test_claude_code_generates_key_and_config(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--claude-code auto-generates a key, writes a temp config and prints
        the snippet; the config is cleaned up afterwards."""
        captured: dict[str, str] = {}

        def fake_run(command: list[str], **kwargs: object) -> None:
            config_file = Path(command[command.index("--config") + 1])
            captured["config"] = config_file.read_text()
            captured["path"] = str(config_file)
            captured["cmd"] = list(command)

        with patch(
            "hellmholtz.integrations.litellm.subprocess.run", side_effect=fake_run
        ):
            start_proxy("openai:gpt-4o", port=4000, claude_code=True)

        # Config was generated with the auto-generated key
        assert "master_key: sk-hellm-" in captured["config"]
        assert "  - model_name: openai:gpt-4o\n" in captured["config"]
        # litellm_params.model uses the config-compatible slash form
        assert "      model: openai/gpt-4o\n" in captured["config"]
        # Temp config is cleaned up when the proxy stops
        assert not Path(captured["path"]).exists()
        # Config path replaces the placeholder, port/host stay intact
        cmd = captured["cmd"]
        assert cmd[cmd.index("--config") + 1] == captured["path"]
        assert "--port" in cmd and "4000" in cmd
        assert "--host" in cmd and "127.0.0.1" in cmd

        # Snippet printed before the (blocking) subprocess call
        out = capsys.readouterr().out
        match = re.search(r'ANTHROPIC_AUTH_TOKEN="(sk-hellm-[^"]+)"', out)
        assert match is not None
        assert match.group(1) in captured["config"]
        assert 'export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"' in out
        assert 'export ANTHROPIC_MODEL="openai:gpt-4o"' in out
        assert "unset ANTHROPIC_API_KEY" in out

    def test_claude_code_with_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--claude-code --name uses the alias in the snippet and config."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", model_name="claude", claude_code=True)

        out = capsys.readouterr().out
        assert 'export ANTHROPIC_MODEL="claude"' in out
        cmd = mock_run.call_args.args[0]
        assert "--config" in cmd

    def test_env_var_key(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """LITELLM_MASTER_KEY is honored and referenced, not printed."""
        monkeypatch.setenv(MASTER_KEY_ENV_VAR, "sk-from-env")
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", claude_code=True)

        out = capsys.readouterr().out
        assert f'export ANTHROPIC_AUTH_TOKEN="${MASTER_KEY_ENV_VAR}"' in out
        assert "sk-from-env" not in out
        cmd = mock_run.call_args.args[0]
        assert "--config" in cmd

    def test_env_var_key_without_claude_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Env key without --claude-code prints the bearer hint instead."""
        monkeypatch.setenv(MASTER_KEY_ENV_VAR, "sk-from-env")
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o")

        out = capsys.readouterr().out
        assert f"Authorization: Bearer ${MASTER_KEY_ENV_VAR}" in out
        cmd = mock_run.call_args.args[0]
        assert "--config" in cmd

    def test_master_key_prints_bearer_hint(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Explicit master key without --claude-code prints the bearer hint."""
        with patch("hellmholtz.integrations.litellm.subprocess.run") as mock_run:
            start_proxy("openai:gpt-4o", master_key="sk-x")

        out = capsys.readouterr().out
        assert "Authorization: Bearer the provided master key" in out
        assert "sk-x" not in out
        cmd = mock_run.call_args.args[0]
        assert "--config" in cmd

    def test_name_only_generates_open_config(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--name without a key generates an open (unauthenticated) config."""
        captured: dict[str, str] = {}

        def fake_run(command: list[str], **kwargs: object) -> None:
            config_file = Path(command[command.index("--config") + 1])
            captured["config"] = config_file.read_text()

        with patch(
            "hellmholtz.integrations.litellm.subprocess.run", side_effect=fake_run
        ):
            start_proxy("openai:gpt-4o", model_name="claude")

        assert "master_key" not in captured["config"]
        assert "open proxy" in capsys.readouterr().out

    def test_keyboard_interrupt(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Ctrl+C during the proxy run stops gracefully."""
        with patch(
            "hellmholtz.integrations.litellm.subprocess.run",
            side_effect=KeyboardInterrupt,
        ):
            start_proxy("openai:gpt-4o")

        assert "Stopping proxy..." in capsys.readouterr().out

    def test_missing_litellm_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing litellm binary exits with code 1 and a hint."""
        with patch(
            "hellmholtz.integrations.litellm.subprocess.run",
            side_effect=FileNotFoundError,
        ), pytest.raises(SystemExit) as exc:
            start_proxy("openai:gpt-4o")

        assert exc.value.code == 1
        assert "pip install .[proxy]" in capsys.readouterr().err

    def test_signal_terminated_is_clean_stop(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A proxy killed by a signal (negative returncode) is a clean stop."""
        import subprocess

        with patch(
            "hellmholtz.integrations.litellm.subprocess.run",
            side_effect=subprocess.CalledProcessError(-15, ["litellm"]),
        ):
            start_proxy("openai:gpt-4o")

        assert "Proxy stopped (signal 15)" in capsys.readouterr().out

    def test_nonzero_exit_still_raises(self) -> None:
        """A non-zero proxy exit code is still treated as an error."""
        import subprocess

        with patch(
            "hellmholtz.integrations.litellm.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["litellm"]),
        ), pytest.raises(subprocess.CalledProcessError):
            start_proxy("openai:gpt-4o")
