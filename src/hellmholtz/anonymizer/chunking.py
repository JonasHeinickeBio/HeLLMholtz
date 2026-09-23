"""Word-safe document chunking for long-text anonymization.

The anonymizer model handles a few thousand words comfortably, but
research documents can be much longer. ``split_document`` breaks text into
chunks that never split a word or a sentence mid-token, so that:

* every word appears in exactly one chunk (no overlap, no omission), and
* paragraph/sentence boundaries are preferred as cut points.

Cross-chunk consistency is handled by the :class:`~hellmholtz.anonymizer.anonymizer.Anonymizer`,
which carries the accumulated original->replacement mapping into the prompt
of every subsequent chunk.
"""

from __future__ import annotations

from collections.abc import Callable
import re

#: Default soft limit per chunk in characters.
DEFAULT_MAX_CHUNK_CHARS = 20000

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _expand_units(
    units: list[str], max_chunk_chars: int, split: Callable[[str], list[str]]
) -> list[str]:
    """Replace units longer than ``max_chunk_chars`` with their split parts."""
    out: list[str] = []
    for unit in units:
        if len(unit) <= max_chunk_chars:
            out.append(unit)
        else:
            out.extend(part for part in split(unit) if part.strip())
    return out


def _pack_units(units: list[str], joiner: str, max_chunk_chars: int) -> list[str]:
    """Pack ``units`` into chunks of at most ``max_chunk_chars`` characters.

    Units are joined with ``joiner``; one separator character per join is
    accounted for. A single unit longer than the limit still gets its own
    chunk.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        if current and current_len + 1 + unit_len > max_chunk_chars:
            chunks.append(joiner.join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += unit_len + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(joiner.join(current))

    return [c for c in chunks if c.strip()]


def split_document(text: str, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into word-safe chunks of at most ``max_chunk_chars``.

    Splitting order: paragraphs first, then sentences, then words. Chunks are
    joined with single newlines; no word is ever split across a chunk.

    Args:
        text: The document to split.
        max_chunk_chars: Soft maximum size per chunk. A single word longer
            than this still gets its own chunk.

    Returns:
        List of chunks. ``"".join`` of the chunks (ignoring the joining
        newlines) contains every word of the input exactly once. If the text
        fits within the limit, a single-element list is returned.
    """
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_chars:
        return [text]

    # Level 1: paragraphs (preserving paragraph breaks).
    units = [p for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    # If paragraphs are the units, remember we may merge across blank lines.
    joiner = "\n\n"

    if max(len(p) for p in units) > max_chunk_chars:
        # Level 2: split oversized paragraphs into sentences.
        units = _expand_units(units, max_chunk_chars, _SENTENCE_SPLIT_RE.split)
        joiner = "\n"

    if units and max(len(u) for u in units) > max_chunk_chars:
        # Level 3: split oversized sentences into words.
        units = _expand_units(units, max_chunk_chars, str.split)
        joiner = " "

    return _pack_units(units, joiner, max_chunk_chars)
