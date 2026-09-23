"""Shared fixtures and a deterministic fake for the ShinrAI PII model."""

import json
from pathlib import Path

import pytest

from tests.anonymizer.fake_model import FakeAnonymizerModel

FIXTURES = Path(__file__).parent / "fixtures"
GROUND_TRUTH = json.loads((FIXTURES / "ground_truth.json").read_text(encoding="utf-8"))


@pytest.fixture
def fixture_texts() -> dict[str, str]:
    """Load the text fixtures by stem."""
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(FIXTURES.glob("sample_*"))
        if p.suffix in {".md", ".txt"}
    }


@pytest.fixture
def ground_truth() -> dict[str, list[dict[str, str]]]:
    return GROUND_TRUTH


@pytest.fixture
def fake_model() -> FakeAnonymizerModel:
    """A fake model that knows every PII value in the English fixture."""
    gt = GROUND_TRUTH["sample_short_en"]
    mapping = {g["value"]: f"REPL-{i:03d}" for i, g in enumerate(gt)}
    types = {g["value"]: g["type"] for g in gt}
    return FakeAnonymizerModel(mapping, types=types)


@pytest.fixture
def long_text() -> str:
    """A deterministic document large enough to require chunking."""
    paragraph = (
        "The contact for this project is John Smith, who works in Berlin. "
        "His office is at Musterstra\u00dfe 12 and he can be reached at "
        "john.smith@example.com or 030-123456."
    )
    return "\n\n".join([paragraph] * 200)
