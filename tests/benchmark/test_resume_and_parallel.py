"""Resume and parallel execution of run_benchmarks."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.benchmark.runner import BenchmarkResult, run_benchmarks
from hellmholtz.core.prompts import Message, Prompt


def _prompt(pid: str, expect: str = "42") -> Prompt:
    return Prompt(
        id=pid,
        category="math",
        messages=[Message(role="user", content=f"question {pid}")],
        checks=[{"type": "contains", "values": [expect]}],
    )


def _response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].finish_reason = "stop"
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 7
    return resp


def _record(model: str, pid: str, success: bool) -> dict:
    return {
        "model": model, "prompt_id": pid, "latency_seconds": 1.0, "success": success,
        "timestamp": "t", "temperature": 0.2, "run_id": 0, "response_text": "42" if success else None,
        "category": "math", "auto_score": 1.0 if success else None,
        "final_score": 100.0 if success else 0.0,
    }


@pytest.mark.parametrize("workers", [1, 2])
def test_resume_reuses_successes_and_retries_failures(tmp_path, workers):
    partial = tmp_path / "benchmark_x.partial.jsonl"
    partial.write_text(
        json.dumps(_record("blablador:m", "p1", True)) + "\n"
        + json.dumps(_record("blablador:m", "p2", False)) + "\n"
        + '{"torn json'  # a line cut off by a killed process
    )
    prompts = [_prompt("p1"), _prompt("p2")]

    with patch("hellmholtz.benchmark.runner.chat_raw", return_value=_response("answer 42")) as chat:
        results = run_benchmarks(
            ["blablador:m"], prompts, results_dir=str(tmp_path), temperatures=[0.2],
            max_concurrent=workers, resume_from=partial,
        )

    assert chat.call_count == 1  # only the failed p2 was re-requested
    assert sorted(r.prompt_id for r in results) == ["p1", "p2"]
    assert all(r.success for r in results)
    appended = []
    for line in partial.read_text().splitlines():
        try:
            appended.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the torn line stays, but must not swallow the next record
    assert any(x["prompt_id"] == "p2" and x["success"] for x in appended)


def test_checks_are_scored_inline_and_open_prompts_wait_for_the_judge(tmp_path):
    verified = _prompt("v")
    open_ended = Prompt(
        id="o", category="creative", messages=[Message(role="user", content="poem")],
        rubric=["is a poem"],
    )
    with patch("hellmholtz.benchmark.runner.chat_raw", return_value=_response("no digits here")):
        results = run_benchmarks(
            ["blablador:m"], [verified, open_ended], results_dir=str(tmp_path), temperatures=[0.2],
        )

    by_id = {r.prompt_id: r for r in results}
    assert by_id["v"].auto_score == 0.0 and by_id["v"].final_score == 0.0
    assert by_id["v"].category == "math"
    assert by_id["o"].final_score is None  # waits for the judge


def test_parallel_run_spreads_models_and_saves_one_file(tmp_path):
    prompts = [_prompt(f"p{i}") for i in range(5)]
    with patch("hellmholtz.benchmark.runner.chat_raw", return_value=_response("42")):
        results = run_benchmarks(
            ["blablador:a", "blablador:b", "blablador:c"], prompts, results_dir=str(tmp_path),
            temperatures=[0.2], max_concurrent=4,
        )

    assert len(results) == 15
    assert {r.model for r in results} == {"blablador:a", "blablador:b", "blablador:c"}
    assert len(list(tmp_path.glob("benchmark_*.json"))) == 1
    partial = next(tmp_path.glob("*.partial.jsonl"))
    assert len(partial.read_text().splitlines()) == 15
    assert isinstance(results[0], BenchmarkResult)
