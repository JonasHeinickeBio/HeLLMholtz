"""Vector-store helpers built on ChromaDB + sentence-transformers.

All heavy imports are deferred so that the module can be imported even
when the *rag* extra is not installed – a helpful ``ImportError`` is raised
only at call time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "hellmholtz_rag"
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_embeddings(model_name: str = _DEFAULT_EMBEDDING_MODEL) -> Any:
    """Return a LangChain HuggingFaceEmbeddings instance."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def _chroma_kwargs(
    collection_name: str,
    embeddings: Any,
    persist_dir: Path | None,
) -> dict[str, Any]:
    """Build keyword arguments for Chroma construction."""
    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "embedding_function": embeddings,
    }
    if persist_dir is not None:
        kwargs["persist_directory"] = str(persist_dir)
    return kwargs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_vector_store(
    documents: list[Document],
    *,
    persist_dir: Path | None = None,
    collection_name: str = _DEFAULT_COLLECTION,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> VectorStore:
    """Embed *documents* and store them in a Chroma collection.

    Args:
        documents:       LangChain ``Document`` objects to index.
        persist_dir:     Optional directory for on-disk persistence.
                         When *None* the store is kept only in memory.
        collection_name: ChromaDB collection name.
        embedding_model: sentence-transformers model identifier.

    Returns:
        A ready-to-query LangChain ``VectorStore``.
    """
    from langchain_community.vectorstores import Chroma

    embeddings = _get_embeddings(embedding_model)
    kwargs = _chroma_kwargs(collection_name, embeddings, persist_dir)
    store = Chroma.from_documents(documents, **kwargs)
    logger.info(
        "Built vector store with %d document(s) (collection=%s)",
        len(documents),
        collection_name,
    )
    return store


def load_vector_store(
    persist_dir: Path,
    *,
    collection_name: str = _DEFAULT_COLLECTION,
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
) -> VectorStore:
    """Load an existing Chroma collection from disk.

    Args:
        persist_dir:     Directory previously passed to :func:`build_vector_store`.
        collection_name: ChromaDB collection name.
        embedding_model: sentence-transformers model identifier.

    Returns:
        A ready-to-query LangChain ``VectorStore``.
    """
    from langchain_community.vectorstores import Chroma

    embeddings = _get_embeddings(embedding_model)
    kwargs = _chroma_kwargs(collection_name, embeddings, persist_dir)
    store = Chroma(**kwargs)
    logger.info("Loaded vector store from %s (collection=%s)", persist_dir, collection_name)
    return store
