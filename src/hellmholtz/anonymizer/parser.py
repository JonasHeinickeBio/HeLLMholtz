"""Parsing of ShinrAI PII anonymizer model responses.

The ``Anonymizer: M-size ShinrAI PII 1.3`` model returns the anonymized text
followed by a footer block::

    <anonymized text ...>
    \u2014 4 entities (123 ms) \u2014
    \u2022 PERSON: John Doe \u2192 DR. A. SMITH (replaced, 0.98)
    \u2022 CITY: Berlin \u2192 NORTHBROOK (replaced, 0.95)

This module splits the body from the footer and parses the entity lines.
Parsing is defensive: malformed lines are skipped rather than raising, so a
slightly off-format model response still yields usable output.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from hellmholtz.anonymizer.models import AnonymizedEntity

# Footer marker line, e.g. "\u2014 4 entities (123 ms) \u2014".
# Accepts an em-dash or ASCII hyphen starts for robustness.
_FOOTER_RE = re.compile(
    r"^\s*(?:\u2014|-{1,2})\s*(?P<count>\d+)\s+entit(?:y|ies)\s*(?:\((?P<ms>\d+(?:\.\d+)?)\s*ms\))?",
    re.IGNORECASE,
)

# One entity line, e.g. "\u2022 PERSON: John Doe \u2192 DR. A. SMITH (replaced, 0.98)".
_ENTITY_RE = re.compile(
    r"^\s*(?:\u2022|\u25aa|\u25cf|[-*+])\s*"
    r"(?P<type>[A-Z][A-Z0-9_/-]*)\s*:\s*"
    r"(?P<original>.+?)\s*(?:\u2192|\u21d2|->|=>)\s*"
    r"(?P<replacement>.+?)"
    r"(?:\s*\((?P<meta>[^()]*(?:\([^()]*\)[^()]*)*)\))?\s*$"
)


@dataclass
class ParsedResponse:
    """Result of parsing one model response."""

    body: str
    entities: list[AnonymizedEntity]
    model_reported_ms: float | None
    footer_found: bool


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped the response in them."""
    lines = text.splitlines()
    if len(lines) >= 2:
        first = lines[0].strip()
        last = lines[-1].strip()
        if first.startswith("```") and last.startswith("```"):
            return "\n".join(lines[1:-1])
    return text


def parse_model_response(text: str) -> ParsedResponse:  # noqa: C901
    """Split a model response into anonymized body and entity footer.

    Args:
        text: Raw model response text.

    Returns:
        ParsedResponse with the body text, parsed entities, model-reported
        latency (if present) and a flag indicating whether a footer was found.
        If no footer is found, the entire text is treated as the body and
        ``entities`` is empty.
    """
    text = strip_code_fences(text).strip()
    lines = text.splitlines()

    footer_idx: int | None = None
    for i, line in enumerate(lines):
        if _FOOTER_RE.match(line):
            footer_idx = i
            break

    if footer_idx is None:
        return ParsedResponse(body=text, entities=[], model_reported_ms=None, footer_found=False)

    body = "\n".join(lines[:footer_idx]).strip("\n").rstrip()
    footer_line = lines[footer_idx]

    footer_match = _FOOTER_RE.match(footer_line)
    ms: float | None = None
    if footer_match:
        raw_ms = footer_match.group("ms")
        if raw_ms is not None:
            ms = float(raw_ms)

    entities: list[AnonymizedEntity] = []
    for line in lines[footer_idx + 1 :]:
        if not line.strip():
            continue
        m = _ENTITY_RE.match(line)
        if m is None:
            continue
        meta = m.group("meta")
        source = ""
        confidence: float | None = None
        if meta is not None:
            meta = meta.strip()
            parts = meta.rsplit(",", 1)
            if len(parts) == 2:
                try:
                    confidence = float(parts[1].strip())
                    source = parts[0].strip()
                except ValueError:
                    source = meta
            else:
                source = meta
        entities.append(
            AnonymizedEntity(
                entity_type=m.group("type").strip(),
                original=m.group("original").strip(),
                replacement=m.group("replacement").strip(),
                source=source,
                confidence=confidence,
            )
        )

    return ParsedResponse(
        body=body,
        entities=entities,
        model_reported_ms=ms,
        footer_found=True,
    )
