"""Core anonymization engine.

Wraps the ``Anonymizer: M-size ShinrAI PII 1.3`` model (served on the
Blablador gateway) with:

* long-document support via word-safe chunking and a cross-chunk
  original->replacement mapping (consistency),
* retry logic for transient failures,
* post-hoc anonymity validation,
* reversibility via :meth:`AnonymizationResult.restore`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
import re
import time
from typing import Any

from hellmholtz.anonymizer.chunking import DEFAULT_MAX_CHUNK_CHARS, split_document
from hellmholtz.anonymizer.models import AnonymizationResult, AnonymizedEntity
from hellmholtz.anonymizer.parser import parse_model_response
from hellmholtz.anonymizer.prompts import DEFAULT_PROMPT, get_prompt
from hellmholtz.anonymizer.validation import validate_anonymity

logger = logging.getLogger(__name__)

#: Default anonymizer model on the Blablador gateway.
DEFAULT_MODEL = "blablador:alias-anonymizer"

#: Type of the injectable chat function: (messages) -> response text.
ChatFn = Callable[[Sequence[Mapping[str, str]]], str]


class Anonymizer:
    """Anonymize documents with the ShinrAI PII 1.3 model.

    Args:
        model: Model string (``provider:model``). Defaults to the ShinrAI PII
            1.3 anonymizer on the Blablador gateway.
        temperature: Sampling temperature; 0.0 for deterministic output.
        max_retries: Retries per chunk after a failed model call.
        validate: Run leak validation on the final result.
        fuzzy: Enable fuzzy leak matching (see
            :func:`hellmholtz.anonymizer.validation.validate_anonymity`).
        fuzzy_threshold: Similarity threshold for fuzzy leak matching.
        max_chunk_chars: Soft size limit per model call.
        prompt: Prompt variant name (see ``hellmholtz.anonymizer.prompts``).
        chat_fn: Optional callable replacing the default LLM call
            ``hellmholtz.client.chat``. Useful for tests and for swapping in
            another model with the same response contract.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_retries: int = 2,
        validate: bool = True,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.85,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        prompt: str = DEFAULT_PROMPT,
        chat_fn: ChatFn | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.validate = validate
        self.fuzzy = fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self.max_chunk_chars = max_chunk_chars
        self.variant = get_prompt(prompt)
        self._chat_fn: ChatFn = chat_fn if chat_fn is not None else self._default_chat_fn

    def _default_chat_fn(self, messages: Sequence[Mapping[str, str]]) -> str:
        from hellmholtz.client import chat

        return chat(self.model, messages, temperature=self.temperature)

    def _call_chunk(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Call the model for one chunk, retrying on failure."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._chat_fn(messages)
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last_error = exc
                logger.warning(
                    "Anonymizer call failed (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    self.model,
                    exc,
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _presubstitute(text: str, known: dict[str, str]) -> str:
        """Replace known originals with their established substitutes.

        Longest originals first so that multi-word values are not clobbered
        by shorter substrings. Falls back to a case-insensitive match.
        """
        for original, repl in sorted(known.items(), key=lambda kv: len(kv[0]), reverse=True):
            if original in text:
                text = text.replace(original, repl)
                continue
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(lambda _m, r=repl: r, text)
        return text

    def anonymize(self, text: str) -> AnonymizationResult:
        """Anonymize ``text`` and return the full result.

        Args:
            text: The document text.

        Returns:
            AnonymizationResult with the anonymized text, all entities found,
            timing, the entity mapping (for restoration) and the validation
            outcome.
        """
        text = text.strip()
        if not text:
            return AnonymizationResult(text="", original="", model=self.model)

        chunks = split_document(text, self.max_chunk_chars)
        logger.debug("Anonymizing %d chars in %d chunk(s)", len(text), len(chunks))

        all_entities: list[AnonymizedEntity] = []
        known: dict[str, str] = {}  # raw original -> replacement (first wins)
        known_cf: dict[str, str] = {}  # casefolded original -> replacement
        bodies: list[str] = []
        raw_responses: list[str] = []
        total_ms = 0.0
        reported_ms: list[float] = []

        for i, chunk in enumerate(chunks):
            chunk_to_send = self._presubstitute(chunk, known) if known else chunk
            messages = self.variant.build_messages(
                chunk_to_send,
                known_mappings=known or None,
                presubstituted=bool(known),
            )

            started = time.perf_counter()
            raw = self._call_chunk(messages)
            total_ms += (time.perf_counter() - started) * 1000.0

            parsed = parse_model_response(raw)
            raw_responses.append(raw)
            if parsed.model_reported_ms is not None:
                reported_ms.append(parsed.model_reported_ms)
            if not parsed.footer_found and self.variant.footer:
                logger.warning(
                    "Chunk %d/%d: no entity footer found; entities for this "
                    "chunk may be incomplete",
                    i + 1,
                    len(chunks),
                )

            for e in parsed.entities:
                all_entities.append(e)
                key = e.original.casefold()
                if key not in known_cf:
                    known_cf[key] = e.replacement
                    known[e.original] = e.replacement

            bodies.append(parsed.body)

        anonymized = "\n\n".join(bodies)
        validation = None
        if self.validate:
            validation = validate_anonymity(
                text,
                anonymized,
                all_entities,
                fuzzy=self.fuzzy,
                fuzzy_threshold=self.fuzzy_threshold,
            )

        return AnonymizationResult(
            text=anonymized,
            entities=all_entities,
            original=text,
            model=self.model,
            duration_ms=total_ms,
            model_reported_ms=sum(reported_ms) if reported_ms else None,
            validation=validation,
            raw_response="\n---\n".join(raw_responses),
        )

    def anonymize_dict(self, text: str) -> dict[str, Any]:
        """Convenience: :meth:`anonymize` as a JSON-serializable dict."""
        return self.anonymize(text).to_dict()
