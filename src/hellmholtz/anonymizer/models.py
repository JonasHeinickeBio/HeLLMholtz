"""Data models for PII anonymization results.

These dataclasses form the stable public API of the ``hellmholtz.anonymizer``
module. They are deliberately free of dependencies on the LLM client so that
they can be used, tested and serialized independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Entity types observed in ShinrAI PII 1.3 model responses.
ENTITY_TYPES: tuple[str, ...] = (
    "PERSON",
    "STREET",
    "CITY",
    "CARD",
    "IBAN",
    "PHONE",
    "EMAIL",
    "ORG",
    "DATE",
)


@dataclass
class AnonymizedEntity:
    """A single PII entity that was replaced in the output text.

    Attributes:
        entity_type: PII category reported by the model (e.g. ``PERSON``).
        original: Original PII value as it appeared in the input text.
        replacement: Replacement value used in the anonymized text.
        source: Source of the replacement as reported by the model
            (e.g. ``replaced``, ``normalized``).
        confidence: Model confidence, 0.0-1.0. ``None`` if not reported.
    """

    entity_type: str
    original: str
    replacement: str
    source: str = ""
    confidence: float | None = None

    def __str__(self) -> str:
        """Return the canonical one-line entity representation."""
        parts = [self.source] if self.source else []
        if self.confidence is not None:
            parts.append(f"{self.confidence:.2f}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"{self.entity_type}: {self.original} -> {self.replacement}{suffix}"


@dataclass
class Leak:
    """A suspected PII leak: an original value (or a close variant of it)
    that survived in the anonymized text.

    Attributes:
        original: The original PII value that should have been removed.
        matched_text: The exact text in the anonymized output that matched.
        match_type: How the match was found (``exact``, ``normalized``,
            ``fuzzy``).
        entity_type: PII category of the leaked value.
        position: Character offset of the match within the anonymized text.
        context: Short context window around the match (anonymized text).
    """

    original: str
    matched_text: str
    match_type: str
    entity_type: str
    position: int
    context: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.match_type}] {self.entity_type} '{self.original}' -> "
            f"'{self.matched_text}' at {self.position}"
        )


@dataclass
class ValidationResult:
    """Outcome of an anonymity validation run.

    Attributes:
        is_anonymous: ``True`` if no leaks were detected.
        leaks: Detected leaks (empty if ``is_anonymous``).
        checked_entities: Number of distinct original values checked.
        fuzzy: Whether fuzzy matching was enabled.
        summary: Human-readable one-line summary.
    """

    is_anonymous: bool
    leaks: list[Leak] = field(default_factory=list)
    checked_entities: int = 0
    fuzzy: bool = True

    @property
    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        if self.is_anonymous:
            return f"anonymous ({self.checked_entities} values checked)"
        return (
            f"NOT anonymous: {len(self.leaks)} leak(s) of {self.checked_entities} checked values"
        )


@dataclass
class AnonymizationResult:
    """Full result of anonymizing a text (or a single chunk of one).

    Attributes:
        text: The anonymized text.
        entities: PII entities found and replaced, in order of the footer.
        original: The original (pre-anonymization) input text. Empty for
            intermediate chunks of a multi-chunk run; set on the final result.
        model: Model string that produced the output.
        duration_ms: Wall-clock latency of the model call(s) in milliseconds.
        model_reported_ms: Latency reported by the model footer, if present.
        validation: Validation outcome (``None`` if validation was skipped).
        raw_response: The raw model response text, for debugging.
    """

    text: str
    entities: list[AnonymizedEntity] = field(default_factory=list)
    original: str = ""
    model: str = ""
    duration_ms: float = 0.0
    model_reported_ms: float | None = None
    validation: ValidationResult | None = None
    raw_response: str = ""

    def mapping(self) -> dict[str, str]:
        """Return the original->replacement mapping for every entity.

        First occurrence wins when the same original value was replaced
        differently (should be rare; indicates an inconsistency in the model
        output).
        """
        out: dict[str, str] = {}
        for e in self.entities:
            out.setdefault(e.original, e.replacement)
        return out

    def restore(self) -> str:
        """Restore the original text from the anonymized text.

        Applies the stored entity mapping in longest-first order so that
        multi-word replacements are not clobbered by shorter ones.

        Returns:
            The (best-effort) de-anonymized text.
        """
        text = self.text
        # Longest originals first prevents partial clobbering.
        for original, replacement in sorted(
            self.mapping().items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            text = text.replace(replacement, original)
        return text

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation (for reports/JSON output)."""
        return {
            "text": self.text,
            "entities": [
                {
                    "entity_type": e.entity_type,
                    "original": e.original,
                    "replacement": e.replacement,
                    "source": e.source,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "model": self.model,
            "duration_ms": round(self.duration_ms, 1),
            "model_reported_ms": self.model_reported_ms,
            "validation": (
                {
                    "is_anonymous": self.validation.is_anonymous,
                    "checked_entities": self.validation.checked_entities,
                    "leaks": [
                        {
                            "original": leak.original,
                            "matched_text": leak.matched_text,
                            "match_type": leak.match_type,
                            "entity_type": leak.entity_type,
                            "position": leak.position,
                            "context": leak.context,
                        }
                        for leak in self.validation.leaks
                    ],
                }
                if self.validation
                else None
            ),
            "n_entities": len(self.entities),
        }
