"""PII anonymization for research documents.

This module wraps the ``Anonymizer: M-size ShinrAI PII 1.3`` model (served on
the Blablador gateway) into a reusable API:

* :class:`~hellmholtz.anonymizer.anonymizer.Anonymizer` - the engine
  (chunking, cross-chunk consistency, retries, validation),
* one-call helpers in :mod:`~hellmholtz.anonymizer.convenience`,
* :class:`~hellmholtz.anonymizer.models.AnonymizationResult` and friends for
  structured results (entity mapping, reversibility, leak reports),
* benchmarking and prompt comparison in
  :mod:`~hellmholtz.anonymizer.benchmark`.

Example::

    from hellmholtz.anonymizer import anonymize_text

    result = anonymize_text("Dr. Jane Smith called from 030-123456.")
    print(result.text)
    for e in result.entities:
        print(e)
    assert result.validation.is_anonymous
"""

from hellmholtz.anonymizer.anonymizer import DEFAULT_MODEL, Anonymizer
from hellmholtz.anonymizer.benchmark import (
    AccuracyReport,
    BenchmarkRun,
    compare_prompts,
    evaluate_accuracy,
    is_consistent,
    run_benchmark,
    write_reports,
)
from hellmholtz.anonymizer.chunking import DEFAULT_MAX_CHUNK_CHARS, split_document
from hellmholtz.anonymizer.convenience import (
    anonymize_documents,
    anonymize_file,
    anonymize_files,
    anonymize_text,
)
from hellmholtz.anonymizer.models import (
    ENTITY_TYPES,
    AnonymizationResult,
    AnonymizedEntity,
    Leak,
    ValidationResult,
)
from hellmholtz.anonymizer.parser import ParsedResponse, parse_model_response
from hellmholtz.anonymizer.prompts import DEFAULT_PROMPT, PROMPTS, PromptVariant, get_prompt
from hellmholtz.anonymizer.validation import validate_anonymity

__all__ = [
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "AccuracyReport",
    "AnonymizationResult",
    "AnonymizedEntity",
    "Anonymizer",
    "BenchmarkRun",
    "ENTITY_TYPES",
    "Leak",
    "PROMPTS",
    "ParsedResponse",
    "PromptVariant",
    "ValidationResult",
    "anonymize_documents",
    "anonymize_file",
    "anonymize_files",
    "anonymize_text",
    "compare_prompts",
    "evaluate_accuracy",
    "get_prompt",
    "is_consistent",
    "parse_model_response",
    "run_benchmark",
    "split_document",
    "validate_anonymity",
    "write_reports",
]
