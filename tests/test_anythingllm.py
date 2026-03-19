"""Tests for the AnythingLLM integration.

All HTTP calls are mocked – no running AnythingLLM instance required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from hellmholtz.cli import app
from hellmholtz.integrations.anythingllm import AnythingLLMClient, AnythingLLMError

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(json_data: dict, *, status_code: int = 200) -> MagicMock:
    ok = status_code < 400
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.reason = "OK" if ok else "Error"
    resp.url = "http://localhost:3001/api/v1/test"
    resp.content = json.dumps(json_data).encode()
    resp.json.return_value = json_data
    return resp


# ---------------------------------------------------------------------------
# AnythingLLMClient unit tests
# ---------------------------------------------------------------------------


class TestAnythingLLMClientInit:
    def test_default_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANYTHINGLLM_BASE_URL", raising=False)
        monkeypatch.setenv("ANYTHINGLLM_API_KEY", "key123")
        client = AnythingLLMClient()
        assert client._base_url == "http://localhost:3001"

    def test_custom_base_url(self) -> None:
        client = AnythingLLMClient(base_url="http://myserver:3001", api_key="k")
        assert client._base_url == "http://myserver:3001"

    def test_trailing_slash_stripped(self) -> None:
        client = AnythingLLMClient(base_url="http://myserver:3001/", api_key="k")
        assert client._base_url == "http://myserver:3001"

    def test_env_var_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANYTHINGLLM_BASE_URL", "http://envserver:3001")
        monkeypatch.setenv("ANYTHINGLLM_API_KEY", "key123")
        client = AnythingLLMClient()
        assert client._base_url == "http://envserver:3001"

    def test_repr_does_not_contain_api_key(self) -> None:
        client = AnythingLLMClient(base_url="http://x:3001", api_key="supersecret")
        assert "supersecret" not in repr(client)

    def test_missing_api_key_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
        import logging

        with caplog.at_level(logging.WARNING, logger="hellmholtz.integrations.anythingllm"):
            AnythingLLMClient(api_key="")
        assert "ANYTHINGLLM_API_KEY" in caplog.text


class TestAnythingLLMClientPing:
    def test_ping_success(self) -> None:
        with patch("requests.get", return_value=_mock_response({"online": True})):
            client = AnythingLLMClient(api_key="key")
            assert client.ping() is True

    def test_ping_failure_returns_false(self) -> None:
        import requests as req

        with patch("requests.get", side_effect=req.RequestException("conn error")):
            client = AnythingLLMClient(api_key="key")
            assert client.ping() is False


class TestAnythingLLMClientListWorkspaces:
    def test_returns_list(self) -> None:
        payload = {"workspaces": [{"slug": "ws1", "name": "WS One"}]}
        with patch("requests.get", return_value=_mock_response(payload)):
            client = AnythingLLMClient(api_key="key")
            result = client.list_workspaces()
        assert result == [{"slug": "ws1", "name": "WS One"}]

    def test_empty_when_no_workspaces(self) -> None:
        with patch("requests.get", return_value=_mock_response({})):
            client = AnythingLLMClient(api_key="key")
            result = client.list_workspaces()
        assert result == []

    def test_raises_on_api_error(self) -> None:
        with patch("requests.get", return_value=_mock_response({}, status_code=401)):
            client = AnythingLLMClient(api_key="bad-key")
            with pytest.raises(AnythingLLMError, match="401"):
                client.list_workspaces()


class TestAnythingLLMClientChat:
    def test_returns_text_response(self) -> None:
        payload = {"textResponse": "The answer is 42.", "sources": []}
        with patch("requests.post", return_value=_mock_response(payload)):
            client = AnythingLLMClient(api_key="key")
            reply = client.chat("my-ws", "What is the answer?")
        assert reply == "The answer is 42."

    def test_empty_text_response_returns_empty_string(self) -> None:
        with patch("requests.post", return_value=_mock_response({})):
            client = AnythingLLMClient(api_key="key")
            reply = client.chat("my-ws", "hello")
        assert reply == ""

    def test_passes_mode_and_session_id(self) -> None:
        payload = {"textResponse": "ok"}
        with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
            client = AnythingLLMClient(api_key="key")
            client.chat("ws", "hi", mode="query", session_id="sess-1")
        called_json = mock_post.call_args.kwargs.get("json", {})
        assert called_json["mode"] == "query"
        assert called_json["sessionId"] == "sess-1"

    def test_raises_on_api_error(self) -> None:
        with patch("requests.post", return_value=_mock_response({}, status_code=500)):
            client = AnythingLLMClient(api_key="key")
            with pytest.raises(AnythingLLMError, match="500"):
                client.chat("ws", "hello")


class TestAnythingLLMClientUpload:
    def test_upload_success(self, tmp_path: Path) -> None:
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")
        payload = {"document": {"location": "custom/report.pdf"}}

        def _fake_post(url: str, **kwargs: object) -> MagicMock:
            if "upload" in url:
                return _mock_response(payload)
            return _mock_response({})  # embed call

        with patch("requests.post", side_effect=_fake_post):
            client = AnythingLLMClient(api_key="key")
            result = client.upload_document("ws", doc)
        assert "document" in result

    def test_upload_missing_file_raises(self, tmp_path: Path) -> None:
        client = AnythingLLMClient(api_key="key")
        with pytest.raises(FileNotFoundError, match="not found"):
            client.upload_document("ws", tmp_path / "nonexistent.pdf")


class TestCreateWorkspace:
    def test_create_workspace(self) -> None:
        payload = {"workspace": {"slug": "new-ws", "name": "New WS"}}
        with patch("requests.post", return_value=_mock_response(payload)):
            client = AnythingLLMClient(api_key="key")
            result = client.create_workspace("New WS")
        assert result["slug"] == "new-ws"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_reads_anythingllm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANYTHINGLLM_BASE_URL", "http://myserver:3001")
    monkeypatch.setenv("ANYTHINGLLM_API_KEY", "secret-key")
    # Re-instantiate so field defaults are re-evaluated
    from hellmholtz.core.config import Settings

    s = Settings()
    assert s.anythingllm_base_url == "http://myserver:3001"
    assert s.anythingllm_api_key == "secret-key"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestAnythingLLMCLIPing:
    def test_ping_success(self) -> None:
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.ping", return_value=True
        ):
            result = runner.invoke(
                app, ["anythingllm", "ping", "--api-key", "k", "--base-url", "http://x:3001"]
            )
        assert result.exit_code == 0
        assert "reachable" in result.output

    def test_ping_failure(self) -> None:
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.ping", return_value=False
        ):
            result = runner.invoke(
                app, ["anythingllm", "ping", "--api-key", "k", "--base-url", "http://x:3001"]
            )
        assert result.exit_code != 0


class TestAnythingLLMCLIWorkspaces:
    def test_lists_workspaces(self) -> None:
        workspaces = [{"slug": "ws1", "name": "My Workspace"}]
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.list_workspaces",
            return_value=workspaces,
        ):
            result = runner.invoke(app, ["anythingllm", "workspaces", "--api-key", "k"])
        assert result.exit_code == 0
        assert "ws1" in result.output

    def test_empty_workspaces(self) -> None:
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.list_workspaces",
            return_value=[],
        ):
            result = runner.invoke(app, ["anythingllm", "workspaces", "--api-key", "k"])
        assert result.exit_code == 0
        assert "No workspaces" in result.output


class TestAnythingLLMCLIChat:
    def test_chat_prints_reply(self) -> None:
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.chat",
            return_value="The answer is 42.",
        ):
            result = runner.invoke(
                app,
                ["anythingllm", "chat", "What is the answer?", "--workspace", "ws", "--api-key", "k"],
            )
        assert result.exit_code == 0
        assert "The answer is 42." in result.output


class TestAnythingLLMCLIUpload:
    def test_upload_success(self, tmp_path: Path) -> None:
        doc = tmp_path / "notes.txt"
        doc.write_text("some content")
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.upload_document",
            return_value={"document": {"location": "custom/notes.txt"}},
        ):
            result = runner.invoke(
                app,
                ["anythingllm", "upload", str(doc), "--workspace", "ws", "--api-key", "k"],
            )
        assert result.exit_code == 0
        assert "Uploaded" in result.output

    def test_upload_missing_file(self, tmp_path: Path) -> None:
        with patch(
            "hellmholtz.integrations.anythingllm.AnythingLLMClient.upload_document",
            side_effect=FileNotFoundError("Document not found: /tmp/missing.pdf"),
        ):
            result = runner.invoke(
                app,
                [
                    "anythingllm",
                    "upload",
                    str(tmp_path / "missing.pdf"),
                    "--workspace",
                    "ws",
                    "--api-key",
                    "k",
                ],
            )
        assert result.exit_code != 0
        assert "Error" in result.output
