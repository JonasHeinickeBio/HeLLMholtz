"""Tests for failure-logging helpers and their effect on fallback logging."""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz import client
from hellmholtz.client import FallbackError, chat, chat_with_fallback
from hellmholtz.core.logging_utils import failures_handled_downstream, log_failure, short

MSGS = [{"role": "user", "content": "hi"}]
LOG = logging.getLogger("test.logging_utils")


def _levels(caplog: pytest.LogCaptureFixture, *, logger: str = "") -> list[str]:
    return [r.levelname for r in caplog.records if r.name.startswith(logger)]


class TestLogFailure:
    def test_error_by_default(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            log_failure(LOG, "boom")
        assert _levels(caplog, logger="test") == ["ERROR"]

    def test_debug_when_handled_downstream(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG), failures_handled_downstream():
            log_failure(LOG, "boom")
        assert _levels(caplog, logger="test") == ["DEBUG"]

    def test_restored_after_block_even_on_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with pytest.raises(ValueError), failures_handled_downstream():
            raise ValueError
        with caplog.at_level(logging.DEBUG):
            log_failure(LOG, "boom")
        assert _levels(caplog, logger="test") == ["ERROR"]

    def test_nested_blocks_restore_outer_state(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG), failures_handled_downstream():
            with failures_handled_downstream():
                pass
            log_failure(LOG, "still quiet")
        assert _levels(caplog, logger="test") == ["DEBUG"]

    def test_other_threads_unaffected(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG), failures_handled_downstream():
            t = threading.Thread(target=log_failure, args=(LOG, "other thread"))
            t.start()
            t.join()
        assert _levels(caplog, logger="test") == ["ERROR"]


class TestShort:
    def test_first_line_only(self) -> None:
        assert short("first\nsecond\nthird") == "first"

    def test_truncates(self) -> None:
        out = short("x" * 500, limit=20)
        assert len(out) == 20
        assert out.endswith("…")

    def test_short_passthrough_and_empty(self) -> None:
        assert short("ok") == "ok"
        assert short("") == ""
        assert short(ValueError("bad")) == "bad"


def _fake_aisuite(fail_for: set[str]) -> MagicMock:
    """aisuite client whose completions raise for the given model ids."""

    def create(model: str, messages: Any, **kwargs: Any) -> MagicMock:
        if model in fail_for:
            client.log_failure(client.logger, f"Chat completion failed for {model}: noisy\nmore")
            raise RuntimeError(f"{model} down\nlong provider detail")
        resp = MagicMock()
        resp.usage = None
        resp.choices[0].message.content = f"answer from {model}"
        return resp

    fake = MagicMock()
    fake.chat.completions.create.side_effect = create
    return fake


@pytest.fixture(autouse=True)
def _ollama_up() -> Any:
    with patch.object(client.ollama, "is_available", return_value=True):
        yield


class TestFallbackLogging:
    def test_successful_fallback_logs_no_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        fake = _fake_aisuite({"a:b"})
        with (
            patch.object(client.ClientManager, "get_client", side_effect=lambda m: (fake, m)),
            caplog.at_level(logging.DEBUG),
        ):
            assert chat_with_fallback("a:b", MSGS, fallbacks="m") == "answer from ollama:m"
        levels = _levels(caplog, logger="hellmholtz")
        assert "ERROR" not in levels
        assert levels.count("WARNING") == 2  # one per failed model + the "fell back" line

    def test_warning_is_one_concise_line(self, caplog: pytest.LogCaptureFixture) -> None:
        fake = _fake_aisuite({"a:b"})
        with (
            patch.object(client.ClientManager, "get_client", side_effect=lambda m: (fake, m)),
            caplog.at_level(logging.WARNING),
        ):
            chat_with_fallback("a:b", MSGS, fallbacks="m")
        failed = next(r.getMessage() for r in caplog.records if "failed" in r.getMessage())
        assert "a:b down" in failed
        assert "long provider detail" not in failed
        assert "\n" not in failed

    def test_all_failed_exception_keeps_full_detail(self, caplog: pytest.LogCaptureFixture) -> None:
        fake = _fake_aisuite({"a:b", "ollama:m"})
        with (
            patch.object(client.ClientManager, "get_client", side_effect=lambda m: (fake, m)),
            caplog.at_level(logging.DEBUG),
            pytest.raises(FallbackError) as exc,
        ):
            chat_with_fallback("a:b", MSGS, fallbacks="m")
        assert "long provider detail" in exc.value.attempts[0][1]
        assert "ERROR" not in _levels(caplog, logger="hellmholtz")

    def test_plain_chat_failure_still_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        fake = _fake_aisuite({"a:b"})
        with (
            patch.object(client.ClientManager, "get_client", side_effect=lambda m: (fake, m)),
            caplog.at_level(logging.DEBUG),
            pytest.raises(RuntimeError),
        ):
            chat("a:b", MSGS)
        assert "ERROR" in _levels(caplog, logger="hellmholtz")
