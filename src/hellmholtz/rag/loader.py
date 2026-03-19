"""Document loader registry for PDF, Markdown, and plain-text files.

Loaders are registered per file-extension so adding a new format only
requires defining a new function and mapping it in ``_LOADER_REGISTRY``.
All loaders return a list of LangChain ``Document`` objects.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private loader functions
# ---------------------------------------------------------------------------


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF file via LangChain's PyPDFLoader (requires *pypdf*)."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    return loader.load()


def _load_text(path: Path) -> list[Document]:
    """Load a plain-text file via LangChain's TextLoader."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def _load_markdown(path: Path) -> list[Document]:
    """Load a Markdown file as plain text (preserves raw Markdown)."""
    docs = _load_text(path)
    for doc in docs:
        doc.metadata["format"] = "markdown"
    return docs


# ---------------------------------------------------------------------------
# Registry: extension → loader function
# ---------------------------------------------------------------------------

_LOADER_REGISTRY: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
}

#: Public set of supported file extensions.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_LOADER_REGISTRY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_file(path: Path) -> list[Document]:
    """Load a single file, dispatching by extension.

    Args:
        path: Path to the file.

    Returns:
        List of LangChain ``Document`` objects.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = path.suffix.lower()
    loader_fn = _LOADER_REGISTRY.get(ext)
    if loader_fn is None:
        supported = sorted(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported extensions: {supported}"
        )
    logger.info("Loading %s file: %s", ext, path)
    docs = loader_fn(path)
    logger.debug("Loaded %d document(s) from %s", len(docs), path)
    return docs


def load_directory(path: Path, *, recursive: bool = True) -> list[Document]:
    """Load all supported files from a directory.

    Args:
        path:      Directory to scan.
        recursive: If *True* (default) also descend into sub-directories.

    Returns:
        Concatenated list of ``Document`` objects from all loaded files.
    """
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    pattern = "**/*" if recursive else "*"
    docs: list[Document] = []
    for file_path in sorted(path.glob(pattern)):
        if file_path.is_file() and file_path.suffix.lower() in _LOADER_REGISTRY:
            try:
                docs.extend(load_file(file_path))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", file_path, exc)

    logger.info("Loaded %d document(s) from directory %s", len(docs), path)
    return docs


def load_paths(paths: list[Path]) -> list[Document]:
    """Load documents from an arbitrary mix of file and directory paths.

    Args:
        paths: List of :class:`~pathlib.Path` objects (files or directories).

    Returns:
        Concatenated list of ``Document`` objects.
    """
    docs: list[Document] = []
    for path in paths:
        if path.is_dir():
            docs.extend(load_directory(path))
        elif path.is_file():
            docs.extend(load_file(path))
        else:
            logger.warning("Path not found, skipping: %s", path)
    return docs
