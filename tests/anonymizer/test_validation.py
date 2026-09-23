"""Unit tests for post-anonymization leak validation."""

import pytest

from hellmholtz.anonymizer.models import AnonymizedEntity
from hellmholtz.anonymizer.validation import validate_anonymity


def _entity(etype: str, original: str) -> AnonymizedEntity:
    return AnonymizedEntity(etype, original, "REPL")


class TestExactMatching:
    def test_case_insensitive(self):
        v = validate_anonymity("orig", "john smith was here", [_entity("PERSON", "John Smith")])
        assert v.is_anonymous is False
        assert v.leaks[0].match_type == "exact"
        assert v.leaks[0].matched_text == "john smith"

    def test_diacritics_folded(self):
        v = validate_anonymity("orig", "muller visited", [_entity("PERSON", "M\u00fcller")])
        assert v.is_anonymous is False
        assert v.leaks[0].match_type == "exact"
        assert v.leaks[0].matched_text == "muller"

    def test_no_leak(self):
        v = validate_anonymity(
            "orig", "DR. A. SMITH was here", [_entity("PERSON", "John Smith")]
        )
        assert v.is_anonymous is True
        assert v.leaks == []

    def test_position_and_context(self):
        anonymized = "xx john smith yy"
        v = validate_anonymity("orig", anonymized, [_entity("PERSON", "John Smith")])
        leak = v.leaks[0]
        assert anonymized[leak.position : leak.position + len(leak.matched_text)] == leak.matched_text
        assert "john smith" in leak.context


class TestNumericNormalized:
    def test_phone_reformatted(self):
        v = validate_anonymity(
            "orig", "call 030 123456 now", [_entity("PHONE", "030-123456")]
        )
        assert v.is_anonymous is False
        assert v.leaks[0].match_type == "normalized"

    def test_iban_with_spaces(self):
        v = validate_anonymity(
            "orig",
            "IBAN: DE89 3704 0044 0532 0130 00",
            [_entity("IBAN", "DE89370400440532013000")],
        )
        assert v.is_anonymous is False
        assert v.leaks[0].match_type == "normalized"

    def test_numeric_types_skip_fuzzy(self):
        # Digit run differs in the last digit: normalized check misses it and
        # numeric types never fall through to fuzzy matching.
        v = validate_anonymity(
            "orig",
            "4111 1111 1111 1112",
            [_entity("CARD", "4111 1111 1111 1111")],
            fuzzy=True,
        )
        assert v.is_anonymous is True


class TestFuzzyMatching:
    def test_multitoken_near_match(self):
        v = validate_anonymity(
            "orig",
            "Dr. John Michael Smyth presented.",
            [_entity("PERSON", "John Michael Smith")],
        )
        assert v.is_anonymous is False
        assert v.leaks[0].match_type == "fuzzy"
        assert v.leaks[0].matched_text == "John Michael Smyth"

    def test_fuzzy_disabled(self):
        v = validate_anonymity(
            "orig",
            "Dr. John Michael Smyth presented.",
            [_entity("PERSON", "John Michael Smith")],
            fuzzy=False,
        )
        assert v.is_anonymous is True

    def test_higher_threshold_no_leak(self):
        v = validate_anonymity(
            "orig",
            "Dr. John Michael Smyth presented.",
            [_entity("PERSON", "John Michael Smith")],
            fuzzy_threshold=0.95,
        )
        assert v.is_anonymous is True

    def test_single_token_values_not_fuzzied(self):
        v = validate_anonymity(
            "orig", "Smyth was here", [_entity("PERSON", "Smith")]
        )
        assert v.is_anonymous is True


class TestGeneral:
    def test_distinct_values_deduplicated(self):
        entities = [
            _entity("PERSON", "John Smith"),
            _entity("PERSON", "john smith"),
        ]
        v = validate_anonymity("orig", "john smith here", entities)
        assert v.checked_entities == 1
        assert len(v.leaks) == 1

    def test_empty_entities(self):
        v = validate_anonymity("orig", "anything", [])
        assert v.is_anonymous is True
        assert v.checked_entities == 0
        assert v.fuzzy is True

    def test_summary_reflects_state(self):
        clean = validate_anonymity("o", "x", [_entity("CITY", "Berlin")])
        assert clean.summary.startswith("anonymous")
        dirty = validate_anonymity("o", "berlin", [_entity("CITY", "Berlin")])
        assert "NOT anonymous" in dirty.summary

    def test_fuzzy_flag_recorded(self):
        v = validate_anonymity("o", "x", [_entity("CITY", "Berlin")], fuzzy=False)
        assert v.fuzzy is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
