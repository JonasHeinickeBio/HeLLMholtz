"""Unit tests for the core Anonymizer engine (fake model, no network)."""

import json
import logging

import pytest

from hellmholtz.anonymizer.anonymizer import DEFAULT_MODEL, Anonymizer
from hellmholtz.anonymizer.benchmark import is_consistent
from tests.anonymizer.fake_model import FakeAnonymizerModel, make_mapping

LOGGER = "hellmholtz.anonymizer.anonymizer"

SMALL_TEXT = "John Smith lives in Berlin."
SMALL_MAPPING = {"John Smith": "REPL-000", "Berlin": "REPL-001"}


@pytest.fixture
def small_fake() -> FakeAnonymizerModel:
    return FakeAnonymizerModel(SMALL_MAPPING, types={"John Smith": "PERSON", "Berlin": "CITY"})


def _gt_mapping(gt: list[dict[str, str]], **kwargs) -> FakeAnonymizerModel:
    """Fake model that knows every value of a ground-truth annotation."""
    return FakeAnonymizerModel(
        make_mapping([g["value"] for g in gt]),
        types={g["value"]: g["type"] for g in gt},
        **kwargs,
    )


class TestDefaults:
    def test_default_model_string(self):
        assert Anonymizer().model == DEFAULT_MODEL == "blablador:alias-anonymizer"

    def test_unknown_prompt_raises(self):
        with pytest.raises(KeyError):
            Anonymizer(prompt="nope")

    def test_empty_text_returns_early(self, fake_model):
        result = Anonymizer(chat_fn=fake_model).anonymize("   \n\t ")
        assert result.text == ""
        assert result.original == ""
        assert result.entities == []
        assert result.validation is None
        assert fake_model.calls == []


class TestSingleChunk:
    def test_full_roundtrip(self, fake_model, fixture_texts):
        text = fixture_texts["sample_short_en"]
        result = Anonymizer(temperature=0.0, chat_fn=fake_model).anonymize(text)

        assert len(result.entities) == 9
        assert result.model_reported_ms == 42.0
        assert result.original == text.strip()
        # Every PII value is gone, every substitute is present.
        for e in result.entities:
            assert e.original not in result.text
            assert e.replacement in result.text
        # Validation ran and found no leaks.
        assert result.validation is not None
        assert result.validation.is_anonymous is True
        assert result.validation.checked_entities == 9
        # Reversibility: the mapping restores the original text exactly.
        assert result.restore() == result.original
        # Mapping is stable (first occurrence wins).
        assert result.mapping() == {e.original: e.replacement for e in result.entities}

    def test_validation_skipped(self, fake_model, fixture_texts):
        result = Anonymizer(validate=False, chat_fn=fake_model).anonymize(
            fixture_texts["sample_short_en"]
        )
        assert result.validation is None

    def test_anonymize_dict_is_json_serializable(self, fake_model, fixture_texts):
        d = Anonymizer(chat_fn=fake_model).anonymize_dict(fixture_texts["sample_short_en"])
        data = json.loads(json.dumps(d))
        assert data["n_entities"] == 9
        assert data["validation"]["is_anonymous"] is True
        assert data["model"] == DEFAULT_MODEL
        assert len(data["entities"]) == 9


class TestMultiChunk:
    LONG_TEXT_MAPPING = {
        "John Smith": "REPL-000",
        "030-123456": "REPL-001",
        "Musterstra\u00dfe 12": "REPL-002",
        "john.smith@example.com": "REPL-003",
        "Berlin": "REPL-004",
    }

    def test_chunks_are_split_and_stay_consistent(self, long_text):
        types = {
            "John Smith": "PERSON",
            "030-123456": "PHONE",
            "Musterstra\u00dfe 12": "STREET",
            "john.smith@example.com": "EMAIL",
            "Berlin": "CITY",
        }
        fake = FakeAnonymizerModel(dict(self.LONG_TEXT_MAPPING), types=types)
        result = Anonymizer(max_chunk_chars=5000, chat_fn=fake).anonymize(long_text)

        assert len(fake.calls) > 1
        # Only the first chunk discovers entities; later chunks are
        # presubstituted and report no footer.
        assert len(result.entities) == 5
        assert is_consistent(result) is True
        assert result.model_reported_ms == 42.0
        assert result.validation is not None
        assert result.validation.is_anonymous is True

        # First call: bare user message, no system message, raw PII present.
        first = fake.calls[0]
        assert all(m["role"] == "user" for m in first)
        assert "John Smith" in first[0]["content"]

        # Second call: system message carries the consistency block, user
        # message contains presubstituted values instead of the originals.
        second = fake.calls[1]
        system = next(m for m in second if m["role"] == "system")
        user = next(m for m in second if m["role"] == "user")
        assert "Context from earlier sections" in system["content"]
        assert "John Smith -> REPL-000" in system["content"]
        assert "keep them unchanged" in system["content"]
        assert "REPL-000" in user["content"]
        assert "John Smith" not in user["content"]

        # Every value is replaced in every chunk and the full text restores.
        for value in ("John Smith", "Berlin", "030-123456"):
            assert value not in result.text
        assert result.restore() == long_text.strip()


class TestRetries:
    def test_recovers_after_transient_failures(self, ground_truth, fixture_texts):
        flaky = _gt_mapping(ground_truth["sample_short_en"], failures=2)
        result = Anonymizer(max_retries=2, chat_fn=flaky).anonymize(
            fixture_texts["sample_short_en"]
        )
        assert len(flaky.calls) == 3  # 2 failures + 1 success
        assert len(result.entities) == 9

    def test_exhausts_retries(self, ground_truth, fixture_texts):
        flaky = _gt_mapping(ground_truth["sample_short_en"], failures=3)
        with pytest.raises(RuntimeError, match="simulated transient API failure"):
            Anonymizer(max_retries=2, chat_fn=flaky).anonymize(
                fixture_texts["sample_short_en"]
            )
        assert len(flaky.calls) == 3  # max_retries + 1 attempts


class TestFooterContract:
    def test_missing_footer_warns(self, caplog, ground_truth, fixture_texts):
        silent = _gt_mapping(ground_truth["sample_short_en"], footer=False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = Anonymizer(chat_fn=silent).anonymize(fixture_texts["sample_short_en"])
        assert result.entities == []
        assert any("no entity footer found" in r.message for r in caplog.records)

    def test_footer_off_variant_does_not_warn(self, caplog, ground_truth, fixture_texts):
        silent = _gt_mapping(ground_truth["sample_short_en"], footer=False)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = Anonymizer(prompt="footer_off", chat_fn=silent).anonymize(
                fixture_texts["sample_short_en"]
            )
        assert result.entities == []
        assert not any("no entity footer found" in r.message for r in caplog.records)


class TestMessageShapes:
    def _user(self, messages):
        return next(m["content"] for m in messages if m["role"] == "user")

    def _system(self, messages):
        return next((m["content"] for m in messages if m["role"] == "system"), None)

    @pytest.mark.parametrize(
        ("prompt", "has_system"),
        [("default", False), ("framed", True), ("strict", True), ("multilingual", True)],
    )
    def test_first_call_layout(self, small_fake, prompt, has_system):
        Anonymizer(prompt=prompt, chat_fn=small_fake).anonymize(SMALL_TEXT)
        messages = small_fake.calls[0]
        assert bool(self._system(messages)) is has_system
        assert "John Smith" in self._user(messages)

    def test_default_user_carries_footer_contract(self, small_fake):
        Anonymizer(prompt="default", chat_fn=small_fake).anonymize(SMALL_TEXT)
        user = self._user(small_fake.calls[0])
        assert "Text:\n" in user
        assert "entities (T ms)" in user

    def test_framed_user_is_bare_text(self, small_fake):
        Anonymizer(prompt="framed", chat_fn=small_fake).anonymize(SMALL_TEXT)
        assert self._user(small_fake.calls[0]) == SMALL_TEXT
        assert self._system(small_fake.calls[0]).startswith(
            "You are a PII anonymization engine."
        )

    def test_strict_user_prefix(self, small_fake):
        Anonymizer(prompt="strict", chat_fn=small_fake).anonymize(SMALL_TEXT)
        assert self._user(small_fake.calls[0]).startswith("Anonymize the text below.\n\n")
        assert "deterministic PII anonymization engine" in self._system(small_fake.calls[0])

    def test_multilingual_system(self, small_fake):
        Anonymizer(prompt="multilingual", chat_fn=small_fake).anonymize(SMALL_TEXT)
        assert "multilingual research documents" in self._system(small_fake.calls[0])

    def test_footer_off_system(self, small_fake):
        Anonymizer(prompt="footer_off", chat_fn=small_fake).anonymize(SMALL_TEXT)
        system = self._system(small_fake.calls[0])
        assert "Output the anonymized text and nothing else." in system
        assert "entities (T ms)" not in system


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
