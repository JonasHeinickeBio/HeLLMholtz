"""One-call convenience functions for PII anonymization.

These are thin wrappers around :class:`~hellmholtz.anonymizer.anonymizer.Anonymizer`
for the common cases:

* :func:`anonymize_text` - anonymize a string,
* :func:`anonymize_file` - anonymize one file (optionally write output),
* :func:`anonymize_files` - anonymize a batch of files,
* :func:`anonymize_documents` - anonymize a mapping of name->text.

All keyword arguments are forwarded to the :class:`Anonymizer` constructor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import logging
from pathlib import Path
from typing import Any

from hellmholtz.anonymizer.anonymizer import DEFAULT_MODEL, Anonymizer
from hellmholtz.anonymizer.models import AnonymizationResult

logger = logging.getLogger(__name__)


def anonymize_text(text: str, *, model: str = DEFAULT_MODEL, **kwargs: Any) -> AnonymizationResult:
    """Anonymize ``text`` with a fresh :class:`Anonymizer`.

    Args:
        text: Document text.
        model: Model string (``provider:model``).
        **kwargs: Forwarded to the :class:`Anonymizer` constructor
            (``temperature``, ``validate``, ``prompt``, ``max_chunk_chars``, ...).

    Returns:
        The full AnonymizationResult.
    """
    return Anonymizer(model=model, **kwargs).anonymize(text)


def _default_out_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.anonymized{path.suffix}")


def anonymize_file(
    path: str | Path,
    *,
    out_path: str | Path | None = None,
    write: bool = False,
    encoding: str = "utf-8",
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> AnonymizationResult:
    """Anonymize the text of a file.

    Args:
        path: Input file (any text format the model can read: .md, .txt, ...).
        out_path: Where to write the anonymized text when ``write`` is True.
            Defaults to ``<stem>.anonymized<suffix>`` next to the input.
        write: If True, write the anonymized text and a JSON sidecar
            (entities, mapping, validation) next to the output.
        encoding: Text encoding of the input file.
        model: Model string.
        **kwargs: Forwarded to the :class:`Anonymizer` constructor.

    Returns:
        The AnonymizationResult. When ``write`` is True, the result's
        ``text`` is also written to disk (see module docstring for the
        sidecar naming).
    """
    src = Path(path)
    text = src.read_text(encoding=encoding)
    result = Anonymizer(model=model, **kwargs).anonymize(text)

    if write:
        target = Path(out_path) if out_path is not None else _default_out_path(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.text + "\n", encoding=encoding)
        sidecar = target.with_suffix(".json")
        sidecar.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Wrote anonymized text to %s (sidecar: %s)", target, sidecar)

    return result


def anonymize_files(
    paths: Iterable[str | Path],
    *,
    out_dir: str | Path | None = None,
    write: bool = True,
    encoding: str = "utf-8",
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> list[tuple[Path | None, AnonymizationResult]]:
    """Anonymize a batch of files with a single shared :class:`Anonymizer`.

    Args:
        paths: Input file paths.
        out_dir: Directory for outputs; defaults to each input's directory.
        write: If True, write anonymized text + JSON sidecar per file.
        encoding: Text encoding.
        model: Model string.
        **kwargs: Forwarded to the :class:`Anonymizer` constructor.

    Returns:
        List of (written_path or None, result) tuples in input order.
    """
    anon = Anonymizer(model=model, **kwargs)
    results: list[tuple[Path | None, AnonymizationResult]] = []
    for path in paths:
        src = Path(path)
        text = src.read_text(encoding=encoding)
        result = anon.anonymize(text)
        written: Path | None = None
        if write:
            base = Path(out_dir) if out_dir is not None else src.parent
            target = base / _default_out_path(src).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.text + "\n", encoding=encoding)
            sidecar = target.with_suffix(".json")
            sidecar.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written = target
        results.append((written, result))
    return results


def anonymize_documents(
    documents: Mapping[str, str],
    *,
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> dict[str, AnonymizationResult]:
    """Anonymize a mapping of document name -> text.

    Args:
        documents: Named texts (e.g. ``{"paper.md": "...", "notes.txt": "..."}`).
        model: Model string.
        **kwargs: Forwarded to the :class:`Anonymizer` constructor.

    Returns:
        Dict of document name -> AnonymizationResult (same key order).
    """
    anon = Anonymizer(model=model, **kwargs)
    return {name: anon.anonymize(text) for name, text in documents.items()}
