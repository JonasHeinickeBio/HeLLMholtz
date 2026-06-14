"""RAG command group: ``hellm rag ingest`` and ``hellm rag chat``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from hellmholtz.cli.common import handle_error

logger = logging.getLogger(__name__)

rag_app = typer.Typer(
    name="rag",
    help="Retrieval-Augmented Generation – chat with your local documents.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Shared options (defined once to keep DRY)
# ---------------------------------------------------------------------------

_DOCS_OPTION = typer.Option(
    None,
    "--docs",
    help=(
        "One or more paths to documents or directories to ingest "
        "(comma-separated, e.g. './papers/,report.pdf')."
    ),
)
_STORE_OPTION = typer.Option(
    None,
    "--store",
    help="Path to a persisted ChromaDB directory (created by `hellm rag ingest`).",
)
_MODEL_OPTION = typer.Option(..., "--model", help="LLM identifier (e.g. openai:gpt-4o).")
_TOP_K_OPTION = typer.Option(4, "--top-k", help="Number of context chunks to retrieve.")
_CHUNK_SIZE_OPTION = typer.Option(1000, "--chunk-size", help="Character budget per text chunk.")
_CHUNK_OVERLAP_OPTION = typer.Option(
    200, "--chunk-overlap", help="Character overlap between adjacent chunks."
)
_TEMPERATURE_OPTION = typer.Option(0.0, "--temperature", help="LLM temperature.")
_MAX_TOKENS_OPTION = typer.Option(None, "--max-tokens", help="Maximum tokens for the response.")
_EMBEDDING_MODEL_OPTION = typer.Option(
    "all-MiniLM-L6-v2",
    "--embedding-model",
    help="sentence-transformers model used for embeddings.",
)
_COLLECTION_OPTION = typer.Option(
    "hellmholtz_rag",
    "--collection",
    help="ChromaDB collection name.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_paths(raw: str) -> list[Path]:
    """Split a comma-separated string of paths and return ``Path`` objects."""
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _require_rag_deps() -> None:
    """Exit with a friendly message when the *rag* extra is missing."""
    try:
        from hellmholtz.rag import check_rag_deps  # noqa: F401

        check_rag_deps()
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@rag_app.command(name="ingest")
def ingest_cmd(
    docs: str = _DOCS_OPTION,
    store: Path | None = _STORE_OPTION,
    chunk_size: int = _CHUNK_SIZE_OPTION,
    chunk_overlap: int = _CHUNK_OVERLAP_OPTION,
    embedding_model: str = _EMBEDDING_MODEL_OPTION,
    collection: str = _COLLECTION_OPTION,
) -> None:
    """Index local documents into a ChromaDB vector store.

    The index is persisted to *--store* so it can be reused across sessions.
    Supported file types: PDF, Markdown (.md / .markdown), plain text (.txt).

    Examples::

        hellm rag ingest --docs ./papers/ --store ./.rag_store
        hellm rag ingest --docs report.pdf,notes.md --store ./.rag_store
    """
    _require_rag_deps()

    if not docs:
        typer.echo("Error: --docs is required for ingest.", err=True)
        raise typer.Exit(1)

    from hellmholtz.rag.pipeline import RAGPipeline

    paths = _parse_paths(docs)
    typer.echo(f"Ingesting {len(paths)} path(s): {[str(p) for p in paths]}")

    try:
        pipeline = RAGPipeline(
            model="",  # Not needed for ingest-only
            persist_dir=store,
            collection_name=collection,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        n_chunks = pipeline.ingest(paths)
        msg = f"Indexed {n_chunks} chunk(s)."
        if store:
            msg += f" Store persisted at: {store}"
        typer.echo(msg)
    except Exception as exc:
        handle_error(exc, "Ingest error")


@rag_app.command(name="chat")
def chat_cmd(
    question: str = typer.Argument(..., help="Question to ask about the documents."),
    model: str = _MODEL_OPTION,
    docs: str | None = _DOCS_OPTION,
    store: Path | None = _STORE_OPTION,
    top_k: int = _TOP_K_OPTION,
    chunk_size: int = _CHUNK_SIZE_OPTION,
    chunk_overlap: int = _CHUNK_OVERLAP_OPTION,
    temperature: float = _TEMPERATURE_OPTION,
    max_tokens: int | None = _MAX_TOKENS_OPTION,
    embedding_model: str = _EMBEDDING_MODEL_OPTION,
    collection: str = _COLLECTION_OPTION,
    system_prompt: str | None = typer.Option(
        None, "--system-prompt", help="Override the default RAG system prompt."
    ),
) -> None:
    """Answer a question grounded in your local documents.

    Supply documents via *--docs* (ingested on the fly) or point to a
    pre-built store via *--store*.  Both options can be combined: new
    documents are added to the existing store.

    Examples::

        # One-shot query (documents indexed in memory)
        hellm rag chat --docs ./papers/ --model openai:gpt-4o "What is the main finding?"

        # Use a persisted index built with `rag ingest`
        hellm rag chat --store ./.rag_store --model openai:gpt-4o "Summarise chapter 3"

        # Ingest on the fly and also persist for later
        hellm rag chat --docs ./papers/ --store ./.rag_store --model openai:gpt-4o "What is X?"
    """
    _require_rag_deps()

    if not docs and not store:
        typer.echo("Error: supply --docs, --store, or both.", err=True)
        raise typer.Exit(1)

    from hellmholtz.rag.pipeline import RAGPipeline

    try:
        if store and store.exists() and not docs:
            # Reuse persisted store
            typer.echo(f"Loading vector store from: {store}")
            pipeline = RAGPipeline.from_store(
                store,
                model=model,
                collection_name=collection,
                embedding_model=embedding_model,
                top_k=top_k,
            )
        else:
            # Build pipeline (with optional persistence)
            pipeline = RAGPipeline(
                model=model,
                persist_dir=store,
                collection_name=collection,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
            )
            if docs:
                paths = _parse_paths(docs)
                typer.echo(f"Ingesting {len(paths)} path(s) …")
                n_chunks = pipeline.ingest(paths)
                typer.echo(f"Indexed {n_chunks} chunk(s).")

        typer.echo("\nGenerating answer …\n")
        answer = pipeline.query(
            question,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        typer.echo(answer)

    except Exception as exc:
        handle_error(exc, "RAG chat error")


# ---------------------------------------------------------------------------
# Registration helper (consistent with other CLI modules)
# ---------------------------------------------------------------------------


def register_rag_commands(app: typer.Typer) -> None:
    """Attach the ``rag`` sub-application to *app*."""
    app.add_typer(rag_app, name="rag")
