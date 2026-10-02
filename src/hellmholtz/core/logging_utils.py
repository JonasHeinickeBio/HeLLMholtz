"""Logging helpers for failures that a caller handles itself.

Chat failures are normally logged at ERROR where they happen. When the caller
retries on another model (see :func:`hellmholtz.client.chat_with_fallback`), the
failure is not an error yet, and the same message logged by every layer makes a
successful fallback look like a crash. Inside :func:`failures_handled_downstream`
:func:`log_failure` demotes those messages to DEBUG.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import logging

__all__ = ["failures_handled_downstream", "log_failure", "short"]

_handled_downstream: ContextVar[bool] = ContextVar(
    "hellm_failures_handled_downstream", default=False
)


@contextmanager
def failures_handled_downstream() -> Iterator[None]:
    """Demote :func:`log_failure` messages to DEBUG within this block."""
    token = _handled_downstream.set(True)
    try:
        yield
    finally:
        _handled_downstream.reset(token)


def log_failure(logger: logging.Logger, message: str) -> None:
    """Log a failure at ERROR, or DEBUG if a caller is handling it."""
    logger.log(logging.DEBUG if _handled_downstream.get() else logging.ERROR, message)


def short(error: object, limit: int = 160) -> str:
    """First line of *error*, truncated, for one-line log messages."""
    first = (str(error).splitlines() or [""])[0].strip()
    return first if len(first) <= limit else f"{first[: limit - 1]}…"
