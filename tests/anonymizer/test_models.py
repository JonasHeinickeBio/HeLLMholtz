"""Unit tests for the anonymizer data models."""

import json

import pytest

from hellmholtz.anonymizer.models import (
    AnonymizationResult,
    AnonymizedEntity,
    ENTITY_TYPES,
    Leak,
    ValidationResult,
)


class TestAnonymizedEntity:
    def test_str_with_source_and_confidence(self):
        e = AnonymizedEntity("PERSON", "John Doe", "DR. A. SMITH", "replaced", 0.98)
        assert str(e) == "PERSON: John Doe -> DR. A. SMITH (replaced, 0.98)"

    def test_str_without_metadata(self):
        e = AnonymizedEntity("CITY", "Berlin", "NORTHBROOK")
        assert str(e) == "CITY: Berlin -> NORTHBROOK"

    def test_str_without_confidence_only(self):
        e = AnonymizedEntity("CITY", "Berlin", "NORTHBROOK", source="replaced")
        assert str(e) == "CITY: Berlin -> NORTHBROOK (replaced)"


class TestValidationResult:
    def test_summary_anonymous(self):
        v = ValidationResult(is_anonymous=True, checked_entities=5)
        assert v.summary == "anonymous (5 values checked)"

    def test_summary_with_leaks(self):
        leak = Leak(
            original="John Doe",
            matched_text="john doe",
            match_type="exact",
            entity_type="PERSON",
            position=0,
        )
        v = ValidationResult(is_anonymous=False, leaks=[leak, leak], checked_entities=5)
        assert v.summary == "NOT anonymous: 2 leak(s) of 5 checked values"


class TestAnonymizationResult:
    def _result(self, entities=None, text=""):
        return AnonymizationResult(text=text, entities=entities or [])

    def test_mapping_first_wins(self):
        r = self._result(
            [
                AnonymizedEntity("PERSON", "John Smith", "DR. A. SMITH"),
                AnonymizedEntity("PERSON", "John Smith", "MR. B. SMITH"),
                AnonymizedEntity("CITY", "Berlin", "NORTHBROOK"),
            ]
        )
        assert r.mapping() == {
            "John Smith": "DR. A. SMITH",
            "Berlin": "NORTHBROOK",
        }

    def test_restore_roundtrip(self):
        entities = [
            AnonymizedEntity("PERSON", "John Smith", "DR. A. SMITH"),
            AnonymizedEntity("CITY", "Berlin", "NORTHBROOK"),
        ]
        anonymized = "DR. A. SMITH visited NORTHBROOK and DR. A. SMITH again."
        r = self._result(entities, anonymized)
        assert r.restore() == "John Smith visited Berlin and John Smith again."

    def test_restore_longest_first(self):
        # A shorter original that is a substring of a longer one.
        entities = [
            AnonymizedEntity("PERSON", "Smith", "X. Y."),
            AnonymizedEntity("PERSON", "John Smith", "DR. A. SMITH"),
        ]
        anonymized = "DR. A. SMITH met X. Y."
        r = self._result(entities, anonymized)
        assert r.restore() == "John Smith met Smith"

    def test_to_dict_is_json_serializable(self):
        v = ValidationResult(is_anonymous=True, checked_entities=1)
        r = self._result(
            [AnonymizedEntity("PERSON", "John Doe", "DR. A. SMITH", "replaced", 0.9)],
            "DR. A. SMITH",
        )
        r.validation = v
        r.duration_ms = 123.456
        payload = json.dumps(r.to_dict(), ensure_ascii=False)
        assert "DR. A. SMITH" in payload
        assert json.loads(payload)["n_entities"] == 1

    def test_to_dict_without_validation(self):
        r = self._result([], "text")
        assert r.to_dict()["validation"] is None


class TestEntityTypes:
    def test_known_types(self):
        assert "PERSON" in ENTITY_TYPES
        assert "IBAN" in ENTITY_TYPES
        assert len(ENTITY_TYPES) == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
