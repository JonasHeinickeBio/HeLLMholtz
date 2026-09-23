"""Unit tests for the one-call convenience wrappers."""

import json

from hellmholtz.anonymizer.convenience import (
    anonymize_documents,
    anonymize_file,
    anonymize_files,
    anonymize_text,
)
from tests.anonymizer.fake_model import FakeAnonymizerModel

MAPPING = {"John Smith": "REPL-000", "Berlin": "REPL-001"}
TYPES = {"John Smith": "PERSON", "Berlin": "CITY"}


def _fake(**kwargs) -> FakeAnonymizerModel:
    return FakeAnonymizerModel(dict(MAPPING), types=dict(TYPES), **kwargs)


class TestAnonymizeText:
    def test_returns_result(self):
        result = anonymize_text("John Smith lives in Berlin.", chat_fn=_fake())
        assert result.text == "REPL-000 lives in REPL-001."
        assert len(result.entities) == 2

    def test_kwargs_forwarded(self):
        fake = _fake()
        result = anonymize_text("John Smith", validate=False, chat_fn=fake)
        assert result.validation is None


class TestAnonymizeFile:
    def test_write_creates_output_and_sidecar(self, tmp_path):
        src = tmp_path / "doc.md"
        src.write_text("# Note\n\nJohn Smith lives in Berlin.", encoding="utf-8")

        result = anonymize_file(src, write=True, chat_fn=_fake())

        out = tmp_path / "doc.anonymized.md"
        assert out.exists()
        assert "REPL-000" in out.read_text(encoding="utf-8")
        assert "John Smith" not in out.read_text(encoding="utf-8")

        sidecar = tmp_path / "doc.anonymized.json"
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["n_entities"] == 2
        assert data["validation"]["is_anonymous"] is True
        assert result.text == out.read_text(encoding="utf-8").rstrip("\n")

    def test_custom_out_path_and_parent_dirs(self, tmp_path):
        src = tmp_path / "note.txt"
        src.write_text("Berlin and John Smith.", encoding="utf-8")
        target = tmp_path / "nested" / "dir" / "custom.md"

        anonymize_file(src, out_path=target, write=True, chat_fn=_fake())

        assert target.exists()
        assert target.with_suffix(".json").exists()

    def test_write_false_writes_nothing(self, tmp_path):
        src = tmp_path / "doc.md"
        src.write_text("John Smith", encoding="utf-8")
        result = anonymize_file(src, write=False, chat_fn=_fake())
        assert result.text == "REPL-000"
        assert list(tmp_path.iterdir()) == [src]


class TestAnonymizeFiles:
    def test_batch_with_out_dir(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.txt"
        a.write_text("John Smith here.", encoding="utf-8")
        b.write_text("Berlin there.", encoding="utf-8")
        out_dir = tmp_path / "out"

        results = anonymize_files([a, b], out_dir=out_dir, chat_fn=_fake())

        assert len(results) == 2
        (path_a, res_a), (path_b, res_b) = results
        assert path_a == out_dir / "a.anonymized.md"
        assert path_b == out_dir / "b.anonymized.txt"
        assert path_a.exists() and path_b.exists()
        assert (out_dir / "a.anonymized.json").exists()
        assert res_a.text == "REPL-000 here."
        assert res_b.text == "REPL-001 there."

    def test_write_false_returns_none_paths(self, tmp_path):
        a = tmp_path / "a.md"
        a.write_text("John Smith", encoding="utf-8")
        results = anonymize_files([a], write=False, chat_fn=_fake())
        assert results[0][0] is None
        assert results[0][1].text == "REPL-000"


class TestAnonymizeDocuments:
    def test_keys_preserved(self):
        docs = {"d1.md": "John Smith", "d2.txt": "Berlin"}
        out = anonymize_documents(docs, chat_fn=_fake())
        assert list(out.keys()) == ["d1.md", "d2.txt"]
        assert out["d1.md"].text == "REPL-000"
        assert out["d2.txt"].text == "REPL-001"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
