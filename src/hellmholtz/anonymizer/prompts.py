"""Prompt templates for PII anonymization experiments.

Each variant produces the list of chat messages for a given text. The
``default`` variant matches the format contract observed in the smoke test of
the ``Anonymizer: M-size ShinrAI PII 1.3`` model (body + entity footer).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_PROMPT = "default"

_ENTITY_TYPES = "PERSON, STREET, CITY, CARD, IBAN, PHONE, EMAIL, ORG, DATE"

#: Minimal system line injected for user-only variants when a consistency
#: block must be carried across chunks (keeps the text message untouched).
_MINIMAL_SYSTEM = "You are a PII anonymization engine."

_BASE_INSTRUCTIONS = """Anonymize the text below. Replace every occurrence of personally
identifiable information - person names, street addresses, cities, credit card
numbers, IBANs, phone numbers, email addresses, organization names, dates -
with realistic but clearly fictional substitutes. Do not translate, summarize,
reorder or comment. Keep the language, formatting and structure of the text
exactly as they are."""

_FOOTER_CONTRACT = f"""

After the anonymized text, add exactly one footer line in this format:
\u2014 N entities (T ms) \u2014
where N is the number of replaced values and T is the time you spent in
milliseconds. Then add one line per replaced entity, in this format:
\u2022 TYPE: original \u2192 replacement (source, confidence)
where TYPE is one of: {_ENTITY_TYPES}; source is the word "replaced";
confidence is a number between 0 and 1.
Output nothing before the text and nothing after the last entity line."""


def _consistency_block(known_mappings: dict[str, str] | None, presubstituted: bool = False) -> str:
    """Render the cross-chunk consistency context for the prompt.

    Args:
        known_mappings: original->replacement pairs from earlier chunks.
        presubstituted: When ``True``, the originals have already been
            replaced with their known substitutes in the text, so the model
            must be told to keep the replacement values unchanged.
    """
    if not known_mappings:
        return ""
    lines = [f"  {orig} -> {repl}" for orig, repl in list(known_mappings.items())[:100]]
    block = "\n\nContext from earlier sections of this document:\n" + "\n".join(lines)
    block += "\nRules for this section:"
    block += (
        "\n  - If you encounter one of the left-hand values, "
        "use exactly the right-hand substitute."
    )
    if presubstituted:
        block += (
            "\n  - The right-hand values are already anonymized and may already "
            "appear in this section: keep them unchanged, do not re-anonymize them."
        )
    return block


@dataclass(frozen=True)
class PromptVariant:
    """A named prompt variant for anonymization."""

    name: str
    description: str
    user_template: str = "{text}"
    system: str | None = None
    #: Whether the model is instructed to emit the entity footer.
    footer: bool = True

    def build_messages(
        self,
        text: str,
        known_mappings: dict[str, str] | None = None,
        presubstituted: bool = False,
    ) -> list[dict[str, str]]:
        """Build the chat messages for one text.

        Args:
            text: The text to anonymize.
            known_mappings: original->replacement pairs established by earlier
                chunks; injected into the system message so the model reuses
                the same substitutes. For user-only variants a minimal system
                message is created on demand.
            presubstituted: Whether ``text`` already has known originals
                replaced by their substitutes (cross-chunk consistency).

        Returns:
            List of OpenAI-style message dicts (system + user).
        """
        consistency = _consistency_block(known_mappings, presubstituted)
        messages: list[dict[str, str]] = []
        if self.system is not None or consistency:
            base_system = self.system if self.system is not None else _MINIMAL_SYSTEM
            messages.append({"role": "system", "content": base_system + consistency})
        messages.append({"role": "user", "content": self.user_template.format(text=text)})
        return messages


PROMPTS: dict[str, PromptVariant] = {
    "default": PromptVariant(
        name="default",
        description="Body + entity footer, single user message (smoke-test contract).",
        user_template=_BASE_INSTRUCTIONS + _FOOTER_CONTRACT + "\n\nText:\n{text}",
    ),
    "framed": PromptVariant(
        name="framed",
        description="System message carries the contract; user message is the bare text.",
        system=(
            "You are a PII anonymization engine. " + _BASE_INSTRUCTIONS + ". " + _FOOTER_CONTRACT
        ),
        user_template="{text}",
    ),
    "strict": PromptVariant(
        name="strict",
        description="Hardened wording: verbatim preservation, no commentary, no additions.",
        system=(
            "You are a deterministic PII anonymization engine. "
            "Rules: "
            "1. Replace every PII value with a fictional substitute. "
            f"2. Recognized categories: {_ENTITY_TYPES}. "
            "3. Preserve every non-PII character verbatim: same words, same "
            "language, same line breaks, same punctuation. "
            "4. Never add, remove or translate content. "
            "5. Use the same substitute for the same value throughout. " + _FOOTER_CONTRACT
        ),
        user_template="Anonymize the text below.\n\n{text}",
    ),
    "multilingual": PromptVariant(
        name="multilingual",
        description="Explicit language-preservation instructions for mixed-language documents.",
        system=(
            "You are a PII anonymization engine for multilingual research documents. "
            "Detect the language of each passage and keep it: German text stays "
            "German, English text stays English, and so on. "
            + _BASE_INSTRUCTIONS
            + ". "
            + _FOOTER_CONTRACT
        ),
        user_template="{text}",
    ),
    "footer_off": PromptVariant(
        name="footer_off",
        description="No entity footer: measures the latency/cost of the footer contract.",
        system=_BASE_INSTRUCTIONS + ". Output the anonymized text and nothing else.",
        user_template="{text}",
        footer=False,
    ),
}


def get_prompt(name: str) -> PromptVariant:
    """Return the named prompt variant.

    Args:
        name: One of ``default``, ``framed``, ``strict``, ``multilingual``,
            ``footer_off``.

    Raises:
        KeyError: If the variant name is unknown.
    """
    if name not in PROMPTS:
        raise KeyError(f"Unknown prompt variant {name!r}; expected one of {sorted(PROMPTS)}")
    return PROMPTS[name]
