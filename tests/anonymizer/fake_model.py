"""Deterministic fake of the ShinrAI PII model for offline tests."""

from __future__ import annotations


class FakeAnonymizerModel:
    """Deterministic stand-in for the ShinrAI PII model.

    Replaces every key of ``mapping`` (longest first) in the user text and
    emits a realistic model response including the footer. Used to test the
    parser, chunking, consistency and validation plumbing without network.

    Args:
        mapping: original -> replacement values to detect and replace.
        types: original -> entity type used in the footer lines.
        failures: Raise ``RuntimeError`` for the first N calls (retry tests).
        footer: When False (or nothing is found), return the text without a
            footer (footer-contract / warning tests).
    """

    def __init__(
        self,
        mapping: dict[str, str],
        types: dict[str, str] | None = None,
        failures: int = 0,
        footer: bool = True,
    ) -> None:
        self.mapping = mapping
        self.types = types or {}
        self.failures = failures
        self.footer = footer
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if len(self.calls) <= self.failures:
            raise RuntimeError("simulated transient API failure")
        user = next(m["content"] for m in messages if m["role"] == "user")
        if "\n\nText:\n" in user:
            text = user.split("\n\nText:\n", 1)[1]
        elif user.startswith("Anonymize the text below.\n\n"):
            text = user[len("Anonymize the text below.\n\n") :]
        else:
            text = user

        found = {o: self.mapping[o] for o in self.mapping if o in text}

        for original in sorted(found, key=len, reverse=True):
            text = text.replace(original, found[original])

        if not self.footer or not found:
            return text

        lines = [f"\u2014 {len(found)} entities (42 ms) \u2014"]
        for original, replacement in found.items():
            etype = self.types.get(original, "OTHER")
            lines.append(f"\u2022 {etype}: {original} \u2192 {replacement} (model, 0.97)")
        return f"{text}\n\n" + "\n".join(lines)


def make_mapping(values: list[str]) -> dict[str, str]:
    """Build a fixed replacement map for a list of PII values."""
    return {v: f"REPL-{i:03d}" for i, v in enumerate(values)}
