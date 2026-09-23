"""Live (network) tests for the anonymizer against the real Blablador gateway.

Skipped unless ``BLABLADOR_API_KEY`` is set (e.g. ``set -a; source .env; set +a``).
Run explicitly with ``pytest tests/anonymizer/test_integration_anonymizer.py -m network``.
"""

import os

import pytest

from hellmholtz.anonymizer import Anonymizer, DEFAULT_MODEL

requires_api_key = pytest.mark.skipif(
    not os.getenv("BLABLADOR_API_KEY"),
    reason="BLABLADOR_API_KEY not set (source .env for live runs)",
)

LIVE_TEXT = (
    "Dr. John Smith, Helmholtz Centre, called from 030-123456. "
    "His email is john.smith@example.com and the meeting is on 12.03.2025 "
    "at Musterstra\u00dfe 12, Berlin."
)


@pytest.mark.network
@pytest.mark.slow
@requires_api_key
def test_live_anonymize_text():
    """The default model config anonymizes a multi-entity document."""
    anon = Anonymizer(temperature=0.0)
    assert anon.model == DEFAULT_MODEL
    result = anon.anonymize(LIVE_TEXT)

    assert result.text
    assert result.text != result.original
    assert result.entities, "model reported no entities"
    assert result.model_reported_ms is not None
    # The validation pass always runs by default.
    assert result.validation is not None
    assert result.validation.checked_entities == len(result.entities)
    # Every reported entity is a non-trivial replacement.
    for e in result.entities:
        assert e.entity_type
        assert e.original
        assert e.replacement
        assert e.replacement != e.original


@pytest.mark.network
@pytest.mark.slow
@requires_api_key
def test_live_restore_roundtrip():
    """restore() brings every reported original back into the text."""
    result = Anonymizer(temperature=0.0).anonymize(LIVE_TEXT)
    restored = result.restore()
    for e in result.entities:
        assert e.original in restored


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
