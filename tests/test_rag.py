"""Tests for the RAG pipeline (rag extra).

All external dependencies (langchain-community, chromadb, sentence-transformers,
pypdf, aisuite, tqdm, …) are mocked so the test suite runs without the full
optional extras.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stub for LangChain Document used throughout
# ---------------------------------------------------------------------------


class _Document:
    """Minimal Document stub that mirrors langchain_core.documents.Document."""

    def __init__(self, page_content: str = "", metadata: dict[str, Any] | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# Mock all optional/missing dependencies before any hellmholtz imports
# ---------------------------------------------------------------------------

_MOCKED_MODULES = [
    # RAG deps
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.embeddings",
    "langchain_community.vectorstores",
    "langchain_core",
    "langchain_core.documents",
    "langchain_core.vectorstores",
    "langchain_text_splitters",
    "chromadb",
    "sentence_transformers",
    "pypdf",
    # Core hellmholtz deps not installed in minimal test env
    "aisuite",
    "aisuite.provider",
    "tqdm",
]

for _mod in _MOCKED_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Expose Document from the langchain_core stub
sys.modules["langchain_core.documents"].Document = _Document  # type: ignore[attr-defined]

# tqdm.tqdm must be callable and return an iterable
sys.modules["tqdm"].tqdm = lambda iterable, **kw: iterable  # type: ignore[attr-defined]

# aisuite.Client must be importable
sys.modules["aisuite"].Client = MagicMock()  # type: ignore[attr-defined]
sys.modules["aisuite.provider"].ProviderFactory = MagicMock()  # type: ignore[attr-defined]

# Now import the modules under test
from hellmholtz.rag import check_rag_deps  # noqa: E402
from hellmholtz.rag.loader import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    load_directory,
    load_file,
    load_paths,
)
from hellmholtz.rag.pipeline import RAGPipeline  # noqa: E402
from hellmholtz.rag.store import build_vector_store, load_vector_store  # noqa: E402


# ---------------------------------------------------------------------------
# check_rag_deps
# ---------------------------------------------------------------------------


class TestCheckRagDeps:
    def test_passes_when_all_present(self) -> None:
        """Should not raise when all deps are importable (they are mocked above)."""
        check_rag_deps()  # No exception expected

    def test_raises_when_dep_missing(self) -> None:
        """Should raise ImportError listing the missing package."""
        original = sys.modules.pop("langchain_community", None)
        try:
            with pytest.raises(ImportError, match="langchain-community"):
                check_rag_deps()
        finally:
            if original is not None:
                sys.modules["langchain_community"] = original
            else:
                sys.modules["langchain_community"] = MagicMock()


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


class TestLoader:
    def test_supported_extensions_contains_expected(self) -> None:
        assert {".pdf", ".txt", ".md", ".markdown"} <= SUPPORTED_EXTENSIONS

    def test_load_file_unsupported_ext_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(f)

    def test_load_file_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("hello world")

        mock_instance = MagicMock()
        mock_instance.load.return_value = [_Document("hello world")]
        mock_loader_cls = MagicMock(return_value=mock_instance)

        with patch("langchain_community.document_loaders.TextLoader", mock_loader_cls):
            docs = load_file(f)

        assert len(docs) == 1
        assert docs[0].page_content == "hello world"

    def test_load_file_markdown_sets_format(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Hello")

        mock_instance = MagicMock()
        mock_instance.load.return_value = [_Document("# Hello")]
        mock_loader_cls = MagicMock(return_value=mock_instance)

        with patch("langchain_community.document_loaders.TextLoader", mock_loader_cls):
            docs = load_file(f)

        assert docs[0].metadata.get("format") == "markdown"

    def test_load_file_pdf(self, tmp_path: Path) -> None:
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-fake")

        mock_instance = MagicMock()
        mock_instance.load.return_value = [_Document("PDF content")]
        mock_loader_cls = MagicMock(return_value=mock_instance)

        with patch("langchain_community.document_loaders.PyPDFLoader", mock_loader_cls):
            docs = load_file(f)

        assert len(docs) == 1

    def test_load_directory_skips_unsupported(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.csv").write_text("x,y")

        mock_instance = MagicMock()
        mock_instance.load.return_value = [_Document("hello")]
        mock_loader_cls = MagicMock(return_value=mock_instance)

        with patch("langchain_community.document_loaders.TextLoader", mock_loader_cls):
            docs = load_directory(tmp_path)

        # Only .txt is processed; .csv is silently skipped
        assert len(docs) == 1

    def test_load_directory_not_a_dir_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="Not a directory"):
            load_directory(f)

    def test_load_paths_mixes_files_and_dirs(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "a.txt").write_text("a")
        standalone = tmp_path / "b.txt"
        standalone.write_text("b")

        call_count = 0

        def _make_instance(content: str) -> MagicMock:
            inst = MagicMock()
            inst.load.return_value = [_Document(content)]
            return inst

        # Each call to TextLoader() returns a new instance
        def _loader_side_effect(path: str, **kw: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return _make_instance(f"content_{call_count}")

        mock_loader_cls = MagicMock(side_effect=_loader_side_effect)

        with patch("langchain_community.document_loaders.TextLoader", mock_loader_cls):
            docs = load_paths([subdir, standalone])

        assert len(docs) == 2

    def test_load_paths_warns_on_missing(self, tmp_path: Path, caplog: Any) -> None:
        import logging

        missing = tmp_path / "nonexistent.txt"
        with caplog.at_level(logging.WARNING, logger="hellmholtz.rag.loader"):
            docs = load_paths([missing])
        assert docs == []
        assert "not found" in caplog.text.lower()


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TestStore:
    def test_build_vector_store_calls_from_documents(self) -> None:
        docs = [_Document("chunk 1"), _Document("chunk 2")]

        mock_chroma_cls = MagicMock()
        mock_chroma_cls.from_documents.return_value = MagicMock()

        with patch("langchain_community.vectorstores.Chroma", mock_chroma_cls), patch(
            "hellmholtz.rag.store._get_embeddings", return_value=MagicMock()
        ):
            store = build_vector_store(docs)

        mock_chroma_cls.from_documents.assert_called_once()
        assert store is mock_chroma_cls.from_documents.return_value

    def test_build_vector_store_with_persist_dir(self, tmp_path: Path) -> None:
        docs = [_Document("chunk")]
        mock_chroma_cls = MagicMock()
        mock_chroma_cls.from_documents.return_value = MagicMock()

        with patch("langchain_community.vectorstores.Chroma", mock_chroma_cls), patch(
            "hellmholtz.rag.store._get_embeddings", return_value=MagicMock()
        ):
            build_vector_store(docs, persist_dir=tmp_path)

        call_kwargs = mock_chroma_cls.from_documents.call_args[1]
        assert "persist_directory" in call_kwargs
        assert call_kwargs["persist_directory"] == str(tmp_path)

    def test_load_vector_store_constructs_chroma(self, tmp_path: Path) -> None:
        mock_chroma_cls = MagicMock()

        with patch("langchain_community.vectorstores.Chroma", mock_chroma_cls), patch(
            "hellmholtz.rag.store._get_embeddings", return_value=MagicMock()
        ):
            load_vector_store(tmp_path)

        mock_chroma_cls.assert_called_once()
        kwargs = mock_chroma_cls.call_args[1]
        assert kwargs["persist_directory"] == str(tmp_path)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    def _make_pipeline(self, **kwargs: Any) -> RAGPipeline:
        return RAGPipeline(model="openai:gpt-4o", **kwargs)

    def test_is_ready_false_by_default(self) -> None:
        pipeline = self._make_pipeline()
        assert not pipeline.is_ready

    def test_query_raises_when_not_ingested(self) -> None:
        pipeline = self._make_pipeline()
        with pytest.raises(RuntimeError, match="No vector store"):
            pipeline.query("test question")

    def test_ingest_returns_chunk_count(self, tmp_path: Path) -> None:
        txt = tmp_path / "doc.txt"
        txt.write_text("hello world")

        mock_docs = [_Document("hello world")]
        mock_chunks = [_Document("hello"), _Document("world")]
        mock_store = MagicMock()

        mock_splitter_cls = MagicMock()
        mock_splitter_cls.return_value.split_documents.return_value = mock_chunks

        with patch("hellmholtz.rag.loader.load_paths", return_value=mock_docs), patch(
            "langchain_text_splitters.RecursiveCharacterTextSplitter", mock_splitter_cls
        ), patch("hellmholtz.rag.store.build_vector_store", return_value=mock_store):
            pipeline = self._make_pipeline()
            n = pipeline.ingest([txt])

        assert n == 2
        assert pipeline.is_ready

    def test_ingest_empty_paths_returns_zero(self) -> None:
        with patch("hellmholtz.rag.loader.load_paths", return_value=[]):
            pipeline = self._make_pipeline()
            n = pipeline.ingest([])
        assert n == 0
        assert not pipeline.is_ready

    def test_query_calls_retriever_and_chat(self) -> None:
        pipeline = self._make_pipeline()

        mock_doc = _Document("relevant context")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        pipeline._store = mock_store

        with patch("hellmholtz.client.chat", return_value="The answer.") as mock_chat:
            answer = pipeline.query("What is X?")

        assert answer == "The answer."
        mock_retriever.invoke.assert_called_once_with("What is X?")
        # Verify context was included in the user message
        call_args = mock_chat.call_args
        messages = call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "relevant context" in user_msg["content"]

    def test_query_respects_custom_system_prompt(self) -> None:
        pipeline = self._make_pipeline()

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [_Document("ctx")]
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        pipeline._store = mock_store

        with patch("hellmholtz.client.chat", return_value="ok") as mock_chat:
            pipeline.query("Q?", system_prompt="Custom instructions.")

        messages = mock_chat.call_args[1]["messages"]
        sys_msg = next(m for m in messages if m["role"] == "system")
        assert sys_msg["content"] == "Custom instructions."

    def test_from_store_loads_existing(self, tmp_path: Path) -> None:
        mock_store = MagicMock()

        with patch("hellmholtz.rag.store.load_vector_store", return_value=mock_store):
            pipeline = RAGPipeline.from_store(tmp_path, model="openai:gpt-4o")

        assert pipeline.is_ready
        assert pipeline.model == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# CLI – rag commands
# ---------------------------------------------------------------------------


class TestRagCLI:
    @pytest.fixture(autouse=True)
    def _mock_rag_deps(self) -> Any:
        """Ensure check_rag_deps never raises in CLI tests."""
        with patch("hellmholtz.cli.rag._require_rag_deps"):
            yield

    @pytest.fixture
    def runner(self) -> Any:
        from typer.testing import CliRunner

        return CliRunner()

    @pytest.fixture
    def app(self) -> Any:
        from hellmholtz.cli import app

        return app

    def test_rag_help_available(self, runner: Any, app: Any) -> None:
        result = runner.invoke(app, ["rag", "--help"])
        assert result.exit_code == 0
        assert "rag" in result.output.lower() or "retrieval" in result.output.lower()

    def test_rag_chat_help(self, runner: Any, app: Any) -> None:
        result = runner.invoke(app, ["rag", "chat", "--help"])
        assert result.exit_code == 0

    def test_rag_ingest_help(self, runner: Any, app: Any) -> None:
        result = runner.invoke(app, ["rag", "ingest", "--help"])
        assert result.exit_code == 0

    def test_rag_ingest_no_docs_exits(self, runner: Any, app: Any) -> None:
        result = runner.invoke(app, ["rag", "ingest"])
        assert result.exit_code != 0

    def test_rag_chat_no_source_exits(self, runner: Any, app: Any) -> None:
        result = runner.invoke(app, ["rag", "chat", "--model", "openai:gpt-4o", "hello"])
        assert result.exit_code != 0

    def test_rag_ingest_calls_pipeline(self, runner: Any, app: Any, tmp_path: Path) -> None:
        txt = tmp_path / "doc.txt"
        txt.write_text("hello")

        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = 3

        with patch("hellmholtz.rag.pipeline.RAGPipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["rag", "ingest", "--docs", str(txt)])

        assert result.exit_code == 0
        mock_pipeline.ingest.assert_called_once()

    def test_rag_chat_with_docs_calls_pipeline(
        self, runner: Any, app: Any, tmp_path: Path
    ) -> None:
        txt = tmp_path / "doc.txt"
        txt.write_text("hello")

        mock_pipeline = MagicMock()
        mock_pipeline.ingest.return_value = 2
        mock_pipeline.query.return_value = "The answer is 42."

        with patch("hellmholtz.rag.pipeline.RAGPipeline", return_value=mock_pipeline):
            result = runner.invoke(
                app,
                [
                    "rag",
                    "chat",
                    "--docs",
                    str(txt),
                    "--model",
                    "openai:gpt-4o",
                    "What is the answer?",
                ],
            )

        assert result.exit_code == 0
        assert "The answer is 42." in result.output
        mock_pipeline.ingest.assert_called_once()
        mock_pipeline.query.assert_called_once()

    def test_rag_chat_with_store_uses_from_store(
        self, runner: Any, app: Any, tmp_path: Path
    ) -> None:
        store_dir = tmp_path / "store"
        store_dir.mkdir()

        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = "Stored answer."

        with patch("hellmholtz.rag.pipeline.RAGPipeline") as mock_cls:
            mock_cls.from_store.return_value = mock_pipeline
            result = runner.invoke(
                app,
                [
                    "rag",
                    "chat",
                    "--store",
                    str(store_dir),
                    "--model",
                    "openai:gpt-4o",
                    "Question?",
                ],
            )

        assert result.exit_code == 0
        assert "Stored answer." in result.output
        mock_cls.from_store.assert_called_once()
