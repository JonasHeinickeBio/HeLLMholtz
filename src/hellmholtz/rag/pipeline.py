"""High-level RAG pipeline: ingest documents → answer questions.

Typical usage::

    from pathlib import Path
    from hellmholtz.rag import RAGPipeline

    pipeline = RAGPipeline(model="openai:gpt-4o")
    pipeline.ingest([Path("./docs/")])
    answer = pipeline.query("What is the main finding?")
    print(answer)

Or, with a persisted index::

    # First run – build and persist the index
    pipeline = RAGPipeline(model="openai:gpt-4o", persist_dir=Path("./.rag_store"))
    pipeline.ingest([Path("./docs/")])

    # Subsequent runs – reuse the persisted index
    pipeline = RAGPipeline.from_store(Path("./.rag_store"), model="openai:gpt-4o")
    answer = pipeline.query("What is X?")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_CHUNK_OVERLAP = 200
_DEFAULT_TOP_K = 4
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use ONLY the context provided below to answer the question. "
    "If the answer cannot be found in the context, say so honestly."
)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Parameters
    ----------
    model:
        LLM identifier in ``provider:name`` format (e.g. ``openai:gpt-4o``).
    persist_dir:
        Optional directory for on-disk ChromaDB persistence.
    collection_name:
        ChromaDB collection name.
    embedding_model:
        sentence-transformers model to use for embeddings.
    chunk_size:
        Character budget per text chunk.
    chunk_overlap:
        Overlap between adjacent chunks.
    top_k:
        Number of context chunks retrieved per query.
    """

    def __init__(
        self,
        model: str,
        *,
        persist_dir: Path | None = None,
        collection_name: str = "hellmholtz_rag",
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self.model = model
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self._store: VectorStore | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_store(
        cls,
        persist_dir: Path,
        model: str,
        *,
        collection_name: str = "hellmholtz_rag",
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        top_k: int = _DEFAULT_TOP_K,
    ) -> RAGPipeline:
        """Load a :class:`RAGPipeline` from a persisted ChromaDB directory.

        Args:
            persist_dir:     Directory of the existing store.
            model:           LLM identifier for queries.
            collection_name: ChromaDB collection name.
            embedding_model: sentence-transformers model identifier.
            top_k:           Number of context chunks per query.

        Returns:
            A :class:`RAGPipeline` with the store pre-loaded.
        """
        from hellmholtz.rag.store import load_vector_store

        pipeline = cls(
            model=model,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            top_k=top_k,
        )
        pipeline._store = load_vector_store(
            persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        return pipeline

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def ingest(self, paths: list[Path]) -> int:
        """Load documents from *paths*, chunk them, and build the vector store.

        Args:
            paths: File or directory paths to ingest.

        Returns:
            Number of text chunks indexed.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from hellmholtz.rag.loader import load_paths
        from hellmholtz.rag.store import build_vector_store

        logger.info("Ingesting %d path(s) …", len(paths))
        docs = load_paths(paths)
        if not docs:
            logger.warning("No documents found in the provided paths.")
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        logger.info("Split into %d chunk(s).", len(chunks))

        self._store = build_vector_store(
            chunks,
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
        )
        return len(chunks)

    def query(
        self,
        question: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Retrieve relevant context and generate a grounded answer.

        Args:
            question:      User's question.
            system_prompt: Override the default RAG system prompt.
            temperature:   LLM temperature (0 = deterministic).
            max_tokens:    Optional token cap for the response.

        Returns:
            The LLM's text answer.

        Raises:
            RuntimeError: If no vector store has been built or loaded yet.
        """
        if self._store is None:
            raise RuntimeError(
                "No vector store available. Call ingest() first, "
                "or load an existing store with RAGPipeline.from_store()."
            )

        from hellmholtz.client import chat

        # 1. Retrieve the most relevant chunks
        retriever = self._store.as_retriever(search_kwargs={"k": self.top_k})
        context_docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

        # 2. Build the prompt
        sys_prompt = system_prompt or _SYSTEM_PROMPT
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        # 3. Call the LLM
        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        logger.debug("Querying %s with %d context chunks.", self.model, len(context_docs))
        return chat(model=self.model, messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """``True`` if a vector store has been built or loaded."""
        return self._store is not None
