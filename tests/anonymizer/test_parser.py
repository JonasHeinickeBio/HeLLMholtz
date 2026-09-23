"""Unit tests for the ShinrAI PII response parser."""

import pytest

from hellmholtz.anonymizer.parser import parse_model_response, strip_code_fences

STANDARD = (
    "Dr. DR. A. SMITH called from 030-000000.\n"
    "\u2014 2 entities (123 ms) \u2014\n"
    "\u2022 PERSON: John Smith \u2192 DR. A. SMITH (replaced, 0.98)\n"
    "\u2022 PHONE: 030-123456 \u2192 030-000000 (replaced, 0.91)"
)


class TestParseModelResponse:
    def test_standard_footer(self):
        parsed = parse_model_response(STANDARD)
        assert parsed.footer_found is True
        assert parsed.body == "Dr. DR. A. SMITH called from 030-000000."
        assert parsed.model_reported_ms == 123.0
        assert [e.entity_type for e in parsed.entities] == ["PERSON", "PHONE"]
        assert parsed.entities[0].original == "John Smith"
        assert parsed.entities[0].replacement == "DR. A. SMITH"
        assert parsed.entities[0].source == "replaced"
        assert parsed.entities[0].confidence == 0.98
        assert parsed.entities[1].confidence == 0.91

    def test_no_footer(self):
        parsed = parse_model_response("Just some anonymized text.\nNothing else.")
        assert parsed.footer_found is False
        assert parsed.entities == []
        assert parsed.model_reported_ms is None
        assert parsed.body == "Just some anonymized text.\nNothing else."

    def test_ascii_hyphen_footer(self):
        text = "Text here.\n- 1 entities (5 ms) -\n- PERSON: John Doe -> DR. A. SMITH"
        parsed = parse_model_response(text)
        assert parsed.footer_found is True
        assert parsed.model_reported_ms == 5.0
        assert parsed.entities[0].original == "John Doe"
        assert parsed.entities[0].replacement == "DR. A. SMITH"

    def test_single_entity_wording(self):
        text = "Body.\n\u2014 1 entity (10 ms) \u2014\n\u2022 CITY: Berlin \u2192 NORTHBROOK"
        parsed = parse_model_response(text)
        assert parsed.footer_found is True
        assert len(parsed.entities) == 1
        assert parsed.entities[0].entity_type == "CITY"

    def test_ascii_arrow_and_meta_without_confidence(self):
        text = "Body.\n\u2014 1 entities (10 ms) \u2014\n\u2022 CITY: Berlin -> NORTHBROOK (normalized)"
        parsed = parse_model_response(text)
        assert parsed.entities[0].replacement == "NORTHBROOK"
        assert parsed.entities[0].source == "normalized"
        assert parsed.entities[0].confidence is None

    def test_entity_without_meta(self):
        text = "Body.\n\u2014 1 entities (10 ms) \u2014\n\u2022 CITY: Berlin \u2192 NORTHBROOK"
        parsed = parse_model_response(text)
        assert parsed.entities[0].source == ""
        assert parsed.entities[0].confidence is None

    def test_malformed_entity_line_skipped(self):
        text = (
            "Body.\n"
            "\u2014 2 entities (10 ms) \u2014\n"
            "\u2022 broken line without arrow\n"
            "\u2022 CITY: Berlin \u2192 NORTHBROOK (replaced, 0.9)"
        )
        parsed = parse_model_response(text)
        assert len(parsed.entities) == 1
        assert parsed.entities[0].entity_type == "CITY"

    def test_footer_at_start_of_response(self):
        text = "\u2014 0 entities (3 ms) \u2014"
        parsed = parse_model_response(text)
        assert parsed.footer_found is True
        assert parsed.body == ""
        assert parsed.entities == []

    def test_multiline_body_preserved(self):
        text = "line one\nline two\n\u2014 1 entities (1 ms) \u2014\n\u2022 CITY: Berlin \u2192 NORTHBROOK"
        parsed = parse_model_response(text)
        assert parsed.body == "line one\nline two"

    def test_whitespace_around_body(self):
        parsed = parse_model_response("   \nBody text.\n\n\u2014 0 entities (1 ms) \u2014\n")
        assert parsed.body == "Body text."

    def test_footer_count_mismatch_does_not_raise(self):
        # Footer claims 2 entities but only one line follows: parse what is there.
        text = "Body.\n\u2014 2 entities (10 ms) \u2014\n\u2022 CITY: Berlin \u2192 NORTHBROOK"
        parsed = parse_model_response(text)
        assert len(parsed.entities) == 1


class TestStripCodeFences:
    def test_strips_fences(self):
        text = "```\nBody.\n\u2014 0 entities (1 ms) \u2014\n```"
        assert strip_code_fences(text) == "Body.\n\u2014 0 entities (1 ms) \u2014"

    def test_keeps_plain_text(self):
        text = "Body.\n```inline code```\nmore"
        assert strip_code_fences(text) == text

    def test_single_backtick_line_untouched(self):
        text = "```\nBody."
        assert strip_code_fences(text) == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
