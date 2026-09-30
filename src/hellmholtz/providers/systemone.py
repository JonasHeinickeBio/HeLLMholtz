"""System-One (Jev-compatible) typed decision support client.

The Blablador platform exposes a dedicated ``/v1/systemone`` endpoint that
serves typed decision requests (a natural-language state plus questions with
criteria) and returns per-question answers with probabilities, confidences and
routing metadata. The reference implementation is the Jev/Laya decision model,
reachable through the ``alias-laya`` alias. Laya is served on its own dedicated
host (``https://laya.blablador.fz-juelich.de``), independent of the general
Blablador API base URL.

Unlike regular Blablador models, System-One models have **no chat surface**:
availability must be checked through :func:`check_availability` instead of a
chat-completion probe.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from hellmholtz.core.config import get_settings

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_SYSTEMONE_MODEL",
    "QUESTION_TYPES",
    "SYSTEMONE_PATH",
    "SystemOneAction",
    "SystemOneAnswer",
    "SystemOneDetection",
    "SystemOneError",
    "SystemOneQuestion",
    "SystemOneResponse",
    "SystemOneRouting",
    "SystemOneUsage",
    "check_availability",
    "get_api_key",
    "get_endpoint",
    "parse_criteria",
    "parse_kv",
    "parse_question",
    "route",
    "validate_question_type",
]

#: Default System-One model (Jev-compatible decision model).
DEFAULT_SYSTEMONE_MODEL = "alias-laya"

#: Base URL of the dedicated Laya host, used when no explicit endpoint is
#: configured. Laya has moved to its own subdomain, separate from the general
#: Blablador API base URL (``BLABLADOR_API_BASE``).
DEFAULT_BASE_URL = "https://laya.blablador.fz-juelich.de/v1"

#: Path of the System-One endpoint relative to the base URL.
SYSTEMONE_PATH = "/systemone"

#: Question types accepted by the System-One API.
QUESTION_TYPES: tuple[str, ...] = ("choice", "score", "noul")


class SystemOneError(RuntimeError):
    """Raised when a System-One request fails or the endpoint is unavailable."""


class SystemOneQuestion(BaseModel):
    """A single typed question in a System-One request."""

    model_config = ConfigDict(extra="ignore")

    type: str = "choice"
    instructions: str = ""
    criteria: dict[str, str] = {}


class SystemOneAction(BaseModel):
    """Action suggestion attached to an answer."""

    model_config = ConfigDict(extra="ignore")

    act_probability: float | None = None


class SystemOneAnswer(BaseModel):
    """The answer to a single question in a System-One response."""

    model_config = ConfigDict(extra="ignore")

    type: str = "choice"
    choice: str | None = None
    probabilities: dict[str, float] = {}
    confidence: float | None = None
    action: SystemOneAction | None = None


class SystemOneUsage(BaseModel):
    """Token usage reported by the System-One endpoint."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneDetection(BaseModel):
    """Routing detection details (detected script/language)."""

    model_config = ConfigDict(extra="ignore")

    script: str = ""
    language: str = ""


class SystemOneRouting(BaseModel):
    """Routing metadata explaining how the request was handled."""

    model_config = ConfigDict(extra="ignore")

    model: str = ""
    repo: str = ""
    reason: str = ""
    detection: SystemOneDetection = SystemOneDetection()


class SystemOneResponse(BaseModel):
    """Full response of the System-One endpoint."""

    model_config = ConfigDict(extra="ignore")

    model: str = ""
    answers: dict[str, SystemOneAnswer] = {}
    usage: SystemOneUsage = SystemOneUsage()
    routing: SystemOneRouting = SystemOneRouting()


def get_endpoint() -> str:
    """Return the URL of the System-One endpoint.

    Resolution order:

    1. ``SYSTEMONE_ENDPOINT`` environment variable (explicit URL).
    2. :data:`DEFAULT_BASE_URL` (the dedicated Laya host) + ``/systemone``.

    Laya is served on its own dedicated subdomain and is intentionally *not*
    derived from ``BLABLADOR_API_BASE``. Private or custom deployments that
    host System-One elsewhere should set ``SYSTEMONE_ENDPOINT`` explicitly.
    """
    settings = get_settings()
    if settings.systemone_endpoint:
        return settings.systemone_endpoint
    return f"{DEFAULT_BASE_URL.rstrip('/')}{SYSTEMONE_PATH}"


def get_api_key() -> str | None:
    """Return the API key used for System-One requests.

    ``SYSTEMONE_API_KEY`` takes precedence over ``BLABLADOR_API_KEY``.
    """
    return os.getenv("SYSTEMONE_API_KEY") or os.getenv("BLABLADOR_API_KEY")


def validate_question_type(question_type: str) -> str:
    """Validate a System-One question type.

    Args:
        question_type: The type to validate.

    Returns:
        The validated type.

    Raises:
        ValueError: If the type is not one of :data:`QUESTION_TYPES`.
    """
    if question_type not in QUESTION_TYPES:
        allowed = ", ".join(QUESTION_TYPES)
        raise ValueError(f"Invalid question type '{question_type}'. Expected one of: {allowed}.")
    return question_type


def route(
    state: str,
    questions: dict[str, SystemOneQuestion],
    model: str = DEFAULT_SYSTEMONE_MODEL,
    timeout: float | None = None,
) -> SystemOneResponse:
    """Send a typed decision request to the System-One endpoint.

    Args:
        state: The decision context described in natural language.
        questions: Mapping of question name to its typed definition.
        model: System-One model to use (default: ``alias-laya``).
        timeout: Request timeout in seconds (default: configured timeout).

    Returns:
        The parsed System-One response.

    Raises:
        SystemOneError: If no API key is configured, the request fails,
            the endpoint is missing (HTTP 404), or the response cannot
            be parsed.
    """
    api_key = get_api_key()
    if not api_key:
        raise SystemOneError(
            "No API key configured for System-One. "
            "Set SYSTEMONE_API_KEY (or BLABLADOR_API_KEY) in the environment."
        )
    if not state.strip():
        raise SystemOneError("The decision state must be a non-empty string.")
    if not questions:
        raise SystemOneError("At least one question is required.")

    payload = {
        "model": model,
        "state": state,
        "questions": {name: question.model_dump() for name, question in questions.items()},
    }

    try:
        response = httpx.post(
            get_endpoint(),
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout if timeout is not None else get_settings().timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise SystemOneError(f"System-One request failed: {exc}") from exc

    if response.status_code == 404:
        raise SystemOneError(
            f"System-One endpoint not found at {response.url}. "
            "The /v1/systemone route is not deployed on this Blablador host yet."
        )
    if response.status_code == 401:
        raise SystemOneError("System-One request was unauthorized (HTTP 401). Check your API key.")
    if response.status_code >= 400:
        raise SystemOneError(
            f"System-One request failed with HTTP {response.status_code}: {response.text}"
        )

    try:
        data: dict[str, Any] = response.json()
    except ValueError as exc:
        raise SystemOneError(f"System-One response is not valid JSON: {exc}") from exc

    try:
        response_obj: SystemOneResponse = SystemOneResponse.model_validate(data)
        return response_obj
    except ValueError as exc:
        raise SystemOneError(f"Could not parse System-One response: {exc}") from exc


def check_availability(model: str = DEFAULT_SYSTEMONE_MODEL) -> bool:
    """Check whether the System-One endpoint is available for a model.

    Sends a minimal typed-decision request (one choice question with two
    criteria) instead of a chat probe, since System-One models have no chat
    surface.

    Args:
        model: System-One model to check (default: ``alias-laya``).

    Returns:
        True if the endpoint responds successfully, False otherwise.
    """
    try:
        route(
            state="Availability check for the System-One endpoint.",
            questions={
                "probe": SystemOneQuestion(
                    type="choice",
                    instructions="Answer with the first option.",
                    criteria={
                        "ok": "System-One is reachable",
                        "fail": "System-One is unreachable",
                    },
                )
            },
            model=model,
        )
        return True
    except SystemOneError:
        return False


def parse_criteria(raw: str) -> dict[str, str]:
    """Parse choice criteria from a CLI string.

    Format: ``"option1:description 1;option2:description 2"``. Options
    without a description (no ``:``) are allowed.

    Args:
        raw: The raw criteria string.

    Returns:
        Mapping of option name to description.

    Raises:
        ValueError: If no criteria could be parsed.
    """
    criteria: dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            option, description = part.split(":", 1)
        else:
            option, description = part, ""
        option = option.strip()
        if option:
            criteria[option] = description.strip()
    if not criteria:
        raise ValueError(f"No criteria could be parsed from {raw!r}.")
    return criteria


def parse_question(raw: str) -> tuple[str, str]:
    """Parse a question definition from a CLI string.

    Format: ``"name:instructions"``.

    Args:
        raw: The raw question string.

    Returns:
        Tuple of (question name, instructions).

    Raises:
        ValueError: If the string is malformed.
    """
    name, sep, instructions = raw.partition(":")
    name = name.strip()
    if not sep or not name:
        raise ValueError(f"Expected a question in the form 'name:instructions', got {raw!r}.")
    return name, instructions.strip()


def parse_kv(raw: str) -> tuple[str, str]:
    """Parse a name/value pair from a CLI string.

    Format: ``"name=value"``.

    Args:
        raw: The raw pair string.

    Returns:
        Tuple of (name, value).

    Raises:
        ValueError: If the string is malformed.
    """
    name, sep, value = raw.partition("=")
    name = name.strip()
    if not sep or not name:
        raise ValueError(f"Expected a pair in the form 'name=value', got {raw!r}.")
    return name, value
