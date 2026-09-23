"""Unit tests for word-safe document chunking."""

import pytest

from hellmholtz.anonymizer.chunking import DEFAULT_MAX_CHUNK_CHARS, split_document


def _words(text: str) -> list[str]:
    return [w for chunk in text.split("\n") for w in chunk.split()]


class TestSplitDocument:
    def test_short_text_single_chunk(self):
        assert split_document("hello world") == ["hello world"]

    def test_empty_text(self):
        assert split_document("") == []
        assert split_document("   \n  \n ") == []

    def test_whitespace_stripped(self):
        assert split_document("  padded  ") == ["padded"]

    def test_invalid_limit(self):
        with pytest.raises(ValueError, match="must be positive"):
            split_document("text", max_chunk_chars=0)

    def test_word_coverage_no_loss_no_dup(self):
        text = " ".join(f"word{i}" for i in range(3000))
        chunks = split_document(text, max_chunk_chars=500)
        assert len(chunks) > 1
        # Joining with spaces recovers every word exactly once.
        assert " ".join(chunks).split() == text.split()

    def test_paragraphs_preferred(self):
        text = "\n\n".join(f"para {i} " + "x" * 40 for i in range(6))
        chunks = split_document(text, max_chunk_chars=200)
        assert len(chunks) > 1
        # Whole paragraphs should stay intact: each chunk is 1+ full paragraphs.
        for chunk in chunks:
            assert "para" in chunk
            for part in chunk.split("\n\n"):
                assert part.startswith("para ")

    def test_sentences_within_oversized_paragraph(self):
        # One giant paragraph of short sentences.
        text = " ".join(f"Sentence number {i} is here." for i in range(50))
        chunks = split_document(text, max_chunk_chars=150)
        assert len(chunks) > 1
        assert " ".join(chunks).split() == text.split()
        # Sentence boundaries respected: no chunk starts mid-sentence.
        for chunk in chunks:
            assert chunk.split(" ")[0][0].isupper() or chunk.split(" ")[0].isdigit()

    def test_words_for_giant_sentence(self):
        # One "sentence" without sentence-final punctuation.
        text = " ".join(f"token{i}" for i in range(200))
        chunks = split_document(text, max_chunk_chars=100)
        assert len(chunks) > 1
        assert " ".join(chunks).split() == text.split()

    def test_no_word_split(self):
        long_word = "a" * 80
        text = f"start {long_word} end {long_word} tail"
        chunks = split_document(text, max_chunk_chars=50)
        # The oversized word gets its own chunk rather than being split.
        assert long_word in chunks
        assert " ".join(chunks).split() == text.split()

    def test_oversized_word_own_chunk(self):
        long_word = "z" * 300
        chunks = split_document(f"before {long_word} after", max_chunk_chars=50)
        assert long_word in chunks
        assert "before" in " ".join(chunks)
        assert "after" in " ".join(chunks)

    def test_default_limit_is_large(self):
        assert DEFAULT_MAX_CHUNK_CHARS >= 10000

    def test_german_umlauts_preserved(self):
        text = "M\u00fcnsterrstra\u00dfe 12, Berlin " * 200
        chunks = split_document(text, max_chunk_chars=300)
        assert " ".join(chunks).split() == text.split()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
