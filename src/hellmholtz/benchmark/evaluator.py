import logging
import re

from hellmholtz.benchmark.runner import BenchmarkResult
from hellmholtz.client import chat
from hellmholtz.core.prompts import Prompt

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """
You are an impartial judge evaluating the quality of an AI model's response.

PROMPT:
{prompt}

RESPONSE:
{response}

Please evaluate the response on a scale of 1 to 10 based on accuracy, helpfulness, and clarity.
Provide a short critique explaining your rating.

Format your output exactly as follows:
RATING: <number>
CRITIQUE: <text>
"""


def evaluate_responses(
    results: list[BenchmarkResult], judge_model: str, prompts: list[Prompt]
) -> list[BenchmarkResult]:
    """Evaluate benchmark responses using an LLM-as-a-Judge approach.

    Uses a specified judge model to rate and critique the responses from
    benchmark runs. Updates the BenchmarkResult objects in-place with
    rating (1-10 scale) and critique text.

    Args:
        results: List of benchmark results to evaluate
        judge_model: Model identifier to use as judge (e.g., "openai:gpt-4o")
        prompts: List of original prompts used in the benchmark

    Returns:
        The same results list with updated rating and critique fields
        for successful runs.

    Note:
        Only evaluates successful runs with response text. Failed runs
        and runs without responses are skipped.
    """
    logger.info(f"Starting evaluation with judge model: {judge_model}")

    for result in results:
        if not result.success or not result.response_text:
            continue

        # Find the prompt by ID
        prompt = next((p for p in prompts if p.id == result.prompt_id), None)
        if not prompt:
            logger.warning(f"Could not find prompt for ID {result.prompt_id}")
            continue

        prompt_text = prompt.user_message

        evaluation_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt_text, response=result.response_text
        )

        try:
            eval_response = chat(
                model=judge_model, messages=[{"role": "user", "content": evaluation_prompt}]
            )

            # Parse response
            rating_match = re.search(r"RATING:\s*(\d+(\.\d+)?)", eval_response)
            critique_match = re.search(r"CRITIQUE:\s*(.*)", eval_response, re.DOTALL)

            if rating_match:
                result.rating = float(rating_match.group(1))
            if critique_match:
                result.critique = critique_match.group(1).strip()

        except Exception as e:
            logger.exception(f"Evaluation failed for {result.model}, {result.prompt_id}: {e}")

    return results
