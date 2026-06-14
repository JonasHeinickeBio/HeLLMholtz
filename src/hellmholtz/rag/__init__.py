"""RAG (Retrieval-Augmented Generation) pipeline.

Optional extra – install with::

    pip install hellmholtz[rag]
    # or, with Poetry:
    poetry install --with rag
"""

from hellmholtz.rag.pipeline import RAGPipeline

__all__ = ["RAGPipeline"]


def check_rag_deps() -> None:
    """Raise ImportError with a helpful message when RAG deps are missing."""
    missing: list[str] = []
    for pkg, import_name in [
        ("langchain-community", "langchain_community"),
        ("langchain-text-splitters", "langchain_text_splitters"),
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
        ("pypdf", "pypdf"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        pkgs = ", ".join(missing)
        raise ImportError(
            f"RAG dependencies not installed: {pkgs}. "
            "Install with: pip install hellmholtz[rag]"
        )
