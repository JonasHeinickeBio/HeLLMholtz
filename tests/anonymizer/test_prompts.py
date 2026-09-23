"""Unit tests for the anonymizer prompt variants."""

import pytest

from hellmholtz.anonymizer.prompts import PROMPTS, PromptVariant, get_prompt


class TestGetPrompt:
    def test_all_variants_registered(self):
        assert set(PROMPTS) == {"default", "framed", "strict", "multilingual", "footer_off"}

    def test_unknown_variant(self):
        with pytest.raises(KeyError, match="Unknown prompt variant"):
            get_prompt("nonexistent")


class TestMessageShapes:
    def test_default_is_user_only_with_text_marker(self):
        messages = get_prompt("default").build_messages("MY TEXT")
        assert [m["role"] for m in messages] == ["user"]
        user = messages[0]["content"]
        assert "MY TEXT" in user
        assert "Text:\nMY TEXT" in user
        assert "\u2014 N entities (T ms) \u2014" in user

    def test_framed_system_carrying_contract(self):
        messages = get_prompt("framed").build_messages("MY TEXT")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "MY TEXT"
        assert "\u2014 N entities (T ms) \u2014" in messages[0]["content"]

    def test_strict_user_prefix(self):
        messages = get_prompt("strict").build_messages("MY TEXT")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "Anonymize the text below.\n\nMY TEXT"

    def test_footer_off_has_no_footer_contract(self):
        variant = get_prompt("footer_off")
        assert variant.footer is False
        messages = variant.build_messages("MY TEXT")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "entities (T ms)" not in messages[0]["content"]

    def test_footer_flag_true_by_default(self):
        for name in ("default", "framed", "strict", "multilingual"):
            assert PROMPTS[name].footer is True

    def test_multilingual_mentions_languages(self):
        messages = get_prompt("multilingual").build_messages("x")
        system = next(m for m in messages if m["role"] == "system")
        assert "German" in system["content"]
        assert "English" in system["content"]


class TestConsistencyBlock:
    def test_no_system_without_mappings_in_default(self):
        messages = get_prompt("default").build_messages("text")
        assert all(m["role"] == "user" for m in messages)

    def test_system_injected_with_mappings_for_default(self):
        messages = get_prompt("default").build_messages(
            "text", known_mappings={"John Smith": "DR. A. SMITH"}
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "John Smith -> DR. A. SMITH" in system["content"]
        assert "Context from earlier sections" in system["content"]
        # Without presubstitution, no "keep unchanged" rule.
        assert "keep them unchanged" not in system["content"]

    def test_presubstituted_rule_added(self):
        messages = get_prompt("default").build_messages(
            "text", known_mappings={"John Smith": "DR. A. SMITH"}, presubstituted=True
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "keep them unchanged" in system["content"]

    def test_framed_appends_to_existing_system(self):
        messages = get_prompt("framed").build_messages(
            "text", known_mappings={"Berlin": "NORTHBROOK"}
        )
        system = next(m for m in messages if m["role"] == "system")
        assert system["content"].startswith("You are a PII anonymization engine.")
        assert "Berlin -> NORTHBROOK" in system["content"]

    def test_mappings_capped_at_100(self):
        mappings = {f"orig{i}": f"repl{i}" for i in range(150)}
        messages = get_prompt("default").build_messages("text", known_mappings=mappings)
        system = next(m for m in messages if m["role"] == "system")
        assert "orig99 -> repl99" in system["content"]
        assert "orig149 -> repl149" not in system["content"]

    def test_frozen_dataclass(self):
        variant = PROMPTS["default"]
        with pytest.raises(Exception):
            variant.name = "other"  # type: ignore[misc]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
