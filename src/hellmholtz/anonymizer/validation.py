"""Post-anonymization validation: detect PII leaks.

For every distinct original value reported by the model (or supplied
externally), the anonymized text is checked at three levels:

1. **exact** - case-insensitive substring match,
2. **normalized** - digit-only containment for numeric entity types
   (CARD, IBAN, PHONE, DATE), so that re-formatted numbers are caught,
3. **fuzzy** - token-window comparison via :mod:`difflib` for multi-word
   values (e.g. a person name with transposed or altered tokens).

No external fuzzy-matching dependency is required.
"""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
import re
import unicodedata

from hellmholtz.anonymizer.models import AnonymizedEntity, Leak, ValidationResult

#: Entity types whose values are numeric and should be checked after
#: stripping non-digit characters.
_NUMERIC_TYPES = frozenset({"CARD", "IBAN", "PHONE", "DATE"})

_DEFAULT_FUZZY_THRESHOLD = 0.85
_CONTEXT_RADIUS = 40


def _norm_digits(value: str) -> str:
    """Return only the ASCII digits of ``value``."""
    return "".join(ch for ch in value if ch.isdigit())


def _casefold(value: str) -> str:
    """Casefold with diacritics folded away (so 'Müller' ~ 'muller')."""
    normalized = unicodedata.normalize("NFD", value.casefold())
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return stripped


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    """Return (token, start, end) for every whitespace-separated token."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _fuzzy_find(
    original: str, tokens: list[tuple[str, int, int]], threshold: float
) -> tuple[str, int, int] | None:
    """Find the best token-window match of ``original`` in ``tokens``.

    Returns:
        (matched_text, start, end) of the first window whose similarity to
        ``original`` is at least ``threshold``, or ``None``.
    """
    needle_tokens = [t for t in original.split() if len(t) > 1]
    window = len(needle_tokens)
    if window < 2 or len(tokens) < window:
        return None

    for i in range(len(tokens) - window + 1):
        window_text = " ".join(t[0] for t in tokens[i : i + window])
        ratio = SequenceMatcher(None, original.casefold(), window_text.casefold()).ratio()
        if ratio >= threshold:
            start = tokens[i][1]
            end = tokens[i + window - 1][2]
            return window_text, start, end
    return None


def _context(text: str, start: int, end: int, radius: int = _CONTEXT_RADIUS) -> str:
    """Return a short context window around [start, end)."""
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{text[lo:hi].strip()}{suffix}"


def _find_exact(original: str, haystack_cf: str) -> int | None:
    """Index of ``original`` in the casefolded haystack, or None."""
    idx = haystack_cf.find(_casefold(original))
    return idx if idx >= 0 else None


def _find_normalized(original: str, anonymized: str) -> int | None:
    """Position of the digit sequence of ``original`` inside ``anonymized``."""
    digits = _norm_digits(original)
    if len(digits) < 6:  # short digit runs (e.g. a day) are too generic
        return None
    # Scan the anonymized text for digit runs and test containment.
    for m in re.finditer(r"\d[\d\s./-]*\d|\d+", anonymized):
        run_digits = _norm_digits(m.group(0))
        if digits in run_digits:
            return m.start()
    return None


def validate_anonymity(
    original: str,
    anonymized: str,
    entities: Iterable[AnonymizedEntity],
    *,
    fuzzy: bool = True,
    fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
) -> ValidationResult:
    """Check that no original PII value survives in ``anonymized``.

    Args:
        original: The original document text (needed to confirm the values
            were actually present; unused for matching itself).
        anonymized: The anonymized document text.
        entities: Entities to check; distinct ``original`` values are used.
        fuzzy: Enable fuzzy (difflib) matching for multi-word values.
        fuzzy_threshold: Similarity ratio (0-1) that counts as a leak.

    Returns:
        ValidationResult with the list of leaks (empty when anonymous).
    """
    distinct: list[AnonymizedEntity] = []
    seen: set[str] = set()
    for e in entities:
        key = _casefold(e.original)
        if key and key not in seen:
            seen.add(key)
            distinct.append(e)

    anonymized_cf = _casefold(anonymized)
    tokens = _token_spans(anonymized)

    leaks: list[Leak] = []
    for e in distinct:
        leak: Leak | None = None

        idx = _find_exact(e.original, anonymized_cf)
        if idx is not None:
            length = len(e.original)
            leak = Leak(
                original=e.original,
                matched_text=anonymized[idx : idx + length],
                match_type="exact",
                entity_type=e.entity_type,
                position=idx,
                context=_context(anonymized, idx, idx + length),
            )
        elif e.entity_type in _NUMERIC_TYPES:
            pos = _find_normalized(e.original, anonymized)
            if pos is not None:
                run = re.search(r"\d[\d\s./-]*\d|\d+", anonymized[pos:])
                end = pos + len(run.group(0)) if run else pos + 1
                leak = Leak(
                    original=e.original,
                    matched_text=anonymized[pos:end],
                    match_type="normalized",
                    entity_type=e.entity_type,
                    position=pos,
                    context=_context(anonymized, pos, end),
                )
        elif fuzzy:
            hit = _fuzzy_find(e.original, tokens, fuzzy_threshold)
            if hit is not None:
                matched_text, start, end = hit
                leak = Leak(
                    original=e.original,
                    matched_text=matched_text,
                    match_type="fuzzy",
                    entity_type=e.entity_type,
                    position=start,
                    context=_context(anonymized, start, end),
                )

        if leak is not None:
            leaks.append(leak)

    return ValidationResult(
        is_anonymous=not leaks,
        leaks=leaks,
        checked_entities=len(distinct),
        fuzzy=fuzzy,
    )
