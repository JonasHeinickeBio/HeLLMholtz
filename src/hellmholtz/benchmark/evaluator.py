"""LLM-as-a-Judge evaluation.

The judge is blind to which model wrote the response, sees a reference answer
and per-prompt rubric when the prompt defines them, and must answer in JSON so
parsing is unambiguous. Several judges can form a panel: their scores are
averaged, and a judge never grades its own model (self-preference bias).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
from typing import Any

from hellmholtz.benchmark.checks import strip_reasoning
from hellmholtz.benchmark.runner import BenchmarkResult, _retry_with_backoff
from hellmholtz.benchmark.scoring import final_score, judge_score, rating_from_score
from hellmholtz.client import chat_raw
from hellmholtz.core.prompts import Prompt

logger = logging.getLogger(__name__)

# Panel used when no judge is given. Chosen for strong instruction following and
# non-overlapping model families; a judge equal to the evaluated model is skipped.
DEFAULT_JUDGES = ["blablador:alias-huge", "blablador:alias-deepseek-v41-flash-booster"]

JUDGE_SYSTEM = (
    "You are a strict, impartial grader for LLM benchmark answers. You do not know which "
    "model wrote the answer. Judge only what is in the answer. Ignore length, tone and "
    "formatting unless the task asks for them. Never reward confident wording over "
    "correctness. Respond with a single JSON object and nothing else."
)

JUDGE_PROMPT_TEMPLATE = """\
TASK GIVEN TO THE MODEL:
<task>
{prompt}
</task>
{reference_block}{rubric_block}
ANSWER TO GRADE:
<answer>
{response}
</answer>

Score each dimension as an integer 1-5 using these anchors:
  5 = fully correct / complete / compliant, nothing to fix
  4 = minor flaw that does not change the outcome
  3 = partly right, one significant flaw or omission
  2 = mostly wrong or missing key parts
  1 = wrong, off-task, or unusable

Dimensions:
  correctness            facts, logic, calculations, code behaviour
  completeness           covers everything the task asks for
  instruction_following  honours every explicit constraint (format, length, language, count)
  clarity                well organised and easy to use for the stated purpose

Check the answer step by step against the reference and rubric before scoring. First write
a brief "analysis" (max 80 words), then the scores.{rubric_hint}

Respond with exactly this JSON shape:
{{"analysis": "<max 80 words>", "correctness": <1-5>, "completeness": <1-5>, \
"instruction_following": <1-5>, "clarity": <1-5>,{rubric_field} "critique": "<one sentence>"}}"""


def build_judge_prompt(prompt: Prompt, response: str) -> str:
    """Render the judge prompt for one prompt/response pair."""
    reference_block = (
        f"\nREFERENCE (ground truth / model solution; the answer may differ in wording):\n"
        f"<reference>\n{prompt.reference}\n</reference>\n"
        if prompt.reference
        else ""
    )
    if prompt.rubric:
        items = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(prompt.rubric))
        rubric_block = f"\nRUBRIC (each item must be verified independently):\n{items}\n"
        rubric_hint = (
            f" Also set rubric_met to a list of exactly {len(prompt.rubric)} booleans, "
            "one per rubric item, in order."
        )
        rubric_field = ' "rubric_met": [<true|false>, ...],'
    else:
        rubric_block, rubric_hint, rubric_field = "", "", ""
    return JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt.user_message,
        response=strip_reasoning(response)[:12000],
        reference_block=reference_block,
        rubric_block=rubric_block,
        rubric_hint=rubric_hint,
        rubric_field=rubric_field,
    )


def parse_verdict(text: str, n_rubric: int = 0) -> dict[str, Any] | None:
    """Parse the judge's JSON verdict; tolerant of fences and surrounding prose."""
    if not text:
        return None
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for dim in ("correctness", "completeness", "instruction_following", "clarity"):
            v = data.get(dim)
            data[dim] = int(round(v)) if isinstance(v, (int, float)) and 1 <= v <= 5 else None
        rubric = data.get("rubric_met")
        if n_rubric and not (
            isinstance(rubric, list)
            and len(rubric) == n_rubric
            and all(isinstance(r, bool) for r in rubric)
        ):
            data["rubric_met"] = None  # malformed: fall back to the dimension scores
        if judge_score(data) is not None:
            return data
    return None


def judge_one(
    prompt: Prompt, response: str, judge_model: str, max_tokens: int = 1500
) -> dict[str, Any] | None:
    """Ask one judge for a verdict; retries unparsable answers. None on failure."""
    user = build_judge_prompt(prompt, response)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    for attempt in range(3):
        try:
            raw = _retry_with_backoff(
                lambda: chat_raw(
                    model=judge_model, messages=messages, temperature=0.0, max_tokens=max_tokens
                )
            )
        except Exception as exc:
            logger.warning("Judge %s failed on %s: %s", judge_model, prompt.id, exc)
            return None
        text = ""
        if raw is not None and getattr(raw, "choices", None):
            message = raw.choices[0].message
            text = message.content or ""
            if "{" not in text:
                text = getattr(message, "reasoning_content", None) or text
        verdict = parse_verdict(strip_reasoning(text) or text, len(prompt.rubric))
        if verdict is not None:
            verdict["judge"] = judge_model
            return verdict
        logger.warning(
            "Unparsable verdict from %s (attempt %d): %r", judge_model, attempt + 1, text[:100]
        )
    return None


def _same_model(judge: str, evaluated: str) -> bool:
    def norm(name: str) -> str:
        return name.split(":", 1)[-1].strip().lower()

    return norm(judge) == norm(evaluated)


def judge_panel(
    prompt: Prompt, response: str, evaluated_model: str, judges: list[str]
) -> dict[str, Any] | None:
    """Average the verdicts of every judge that is not the evaluated model."""
    verdicts = [
        v
        for j in judges
        if not _same_model(j, evaluated_model)
        for v in [judge_one(prompt, response, j)]
        if v is not None
    ]
    if not verdicts:
        return None
    scores = [s for v in verdicts for s in [judge_score(v)] if s is not None]
    return {
        "score": sum(scores) / len(scores),
        "spread": (max(scores) - min(scores)) if len(scores) > 1 else 0.0,
        "critique": verdicts[0].get("critique"),
        "verdicts": verdicts,
    }


def needs_judge(prompt: Prompt) -> bool:
    """Prompts fully covered by deterministic checks keep their auto score only."""
    return bool(prompt.rubric) or not prompt.checks


def evaluate_responses(
    results: list[BenchmarkResult],
    judge_model: str | list[str],
    prompts: list[Prompt],
    workers: int = 4,
    checkpoint: Callable[[], None] | None = None,
    checkpoint_every: int = 25,
) -> list[BenchmarkResult]:
    """Judge benchmark results in place and recompute their final scores.

    ``judge_model`` may be one identifier, a comma-separated string or a list
    (a panel). Already-judged results are skipped, so the call is resumable.
    """
    judges = (
        [j.strip() for j in judge_model.split(",") if j.strip()]
        if isinstance(judge_model, str)
        else list(judge_model)
    )
    by_id = {p.id: p for p in prompts}
    todo = [
        r
        for r in results
        if r.success
        and r.response_text
        and r.prompt_id in by_id
        and r.judge_score is None
        and needs_judge(by_id[r.prompt_id])
    ]
    logger.info("Judging %d results with panel %s", len(todo), judges)

    def work(result: BenchmarkResult) -> tuple[BenchmarkResult, dict[str, Any] | None]:
        return result, judge_panel(
            by_id[result.prompt_id], result.response_text or "", result.model, judges
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, r) for r in todo]):
            result, panel = fut.result()
            if panel is None:
                continue
            result.judge_score = round(panel["score"], 4)
            result.judge_spread = round(panel["spread"], 4)
            result.rating = rating_from_score(panel["score"])
            result.critique = panel["critique"]
            result.judge_details = panel["verdicts"]
            result.final_score = final_score(result.auto_score, result.judge_score)
            done += 1
            if checkpoint and done % checkpoint_every == 0:
                checkpoint()
    return results
