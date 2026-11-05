"""Regression tests for review findings on the v2 benchmark."""

import pytest

from hellmholtz.benchmark.checks import auto_score, run_checks, strip_reasoning
from hellmholtz.benchmark.evaluator import _same_model, build_judge_prompt, parse_verdict
from hellmholtz.benchmark.prompts import get_prompt_by_id
from hellmholtz.benchmark.runner import BenchmarkResult, _FailureStreaks, _score_with_checks
from hellmholtz.benchmark.scoring import dedupe_attempts, final_score
from hellmholtz.core.prompts import Message, Prompt

GOOD = {"dims": dict(correctness=5, completeness=5, instruction_following=5, clarity=5)}


def _score(prompt_id: str, answer: str) -> float:
    prompt = get_prompt_by_id(prompt_id)
    return auto_score(run_checks(prompt.checks, answer)) or 0.0


def test_judge_alias_is_resolved_to_the_evaluated_deployment():
    assert _same_model(
        "blablador:alias-deepseek-v41-flash-booster",
        "blablador:15 - DeepSeek-V4.1-Flash on JUWELS Booster",
    )
    assert not _same_model("blablador:alias-huge", "blablador:alias-large")


def test_partial_verdict_is_rejected():
    assert parse_verdict('{"clarity": 5}') is None
    assert parse_verdict('{"correctness":5,"completeness":5,"instruction_following":5}') is None
    assert parse_verdict('{"correctness":5,"completeness":4,"instruction_following":5,"clarity":4}')


def test_judge_prompt_includes_system_message():
    prompt = Prompt(
        id="x",
        category="instruction",
        messages=[Message(role="system", content="Answer in French."), Message(role="user", content="Hi")],
    )
    assert "Answer in French." in build_judge_prompt(prompt, "Hello")


def test_unclosed_thinking_tag_hides_the_answer():
    assert strip_reasoning("<thinking>maybe\nFinal answer: 5") == ""


def test_negative_check_weight_is_rejected():
    with pytest.raises(ValueError):
        Prompt(
            id="x",
            category="c",
            messages=[Message(role="user", content="q")],
            checks=[{"type": "contains", "values": ["a"], "weight": -1}],
        )


def test_prompts_with_rubric_stay_unscored_until_judged():
    assert final_score(0.8, None, needs_judge=True) is None
    assert final_score(0.8, 0.5, needs_judge=True) == pytest.approx(68.0)
    prompt = get_prompt_by_id("coding_007")
    res = BenchmarkResult(
        model="m",
        prompt_id="coding_007",
        latency_seconds=1,
        success=True,
        timestamp="t",
        response_text="package main http.ListenAndServe Hello, World! 8080",
    )
    _score_with_checks(res, prompt)
    assert res.auto_score == 1.0 and res.final_score is None
    res.judge_score = 0.5
    _score_with_checks(res, prompt)  # rescoring keeps the verdict
    assert res.final_score == pytest.approx(80.0)


def test_json_types_are_enforced():
    ok = '{"name":"Ada","age":34,"skills":["Go","SQL","Rust"]}'
    assert _score("instruction_003", ok) == 1.0
    for bad in (
        '{"name":"Ada","age":"old","skills":["Go","SQL","Rust"]}',
        '{"name":"Ada","age":34,"skills":[1,2,3]}',
        '{"name":"Ada","age":34,"skills":["Go",1,"Rust"]}',
        '{"name":"Ada","age":true,"skills":["Go","SQL","Rust"]}',
    ):
        assert _score("instruction_003", bad) < 1.0, bad


def test_extraction_001_rejects_prose_around_json():
    body = '{"invoice_number": "INV-2026-0417", "total_eur": 392.0, "due_date": "2026-04-02"}'
    assert _score("extraction_001", body) == 1.0
    assert _score("extraction_001", f"Here is the invoice: {body}") < 1.0


def test_photosynthesis_requires_the_glucose_product():
    assert _score("knowledge_005", "6 CO2 + 6 H2O -> C6H12O6 + 6 O2") == 1.0
    assert _score("knowledge_005", "6 CO2 + 6 H2O -> 6 O2 + C6H12O6") == 1.0
    assert _score("knowledge_005", "6 CO2 + 6 H2O -> salt + 6 O2") < 1.0


def test_dedupe_keeps_the_final_attempt():
    recs = [
        {"model": "m", "prompt_id": "p", "temperature": 0.0, "run_id": 0, "success": False},
        {"model": "m", "prompt_id": "p", "temperature": 0.0, "run_id": 0, "success": True},
        {"model": "m", "prompt_id": "q", "temperature": 0.0, "run_id": 0, "success": False},
    ]
    out = dedupe_attempts(recs)
    assert [(r["prompt_id"], r["success"]) for r in out] == [("p", True), ("q", False)]


def test_failure_streaks_are_shared_per_model():
    s = _FailureStreaks()
    s.hit("a")
    s.hit("a")
    s.hit("b")
    assert (s.get("a"), s.get("b")) == (2, 1)
    s.reset("a")
    assert s.get("a") == 0


def test_python_tests_cannot_read_repository_files():
    from pathlib import Path

    from hellmholtz.benchmark.checks import _run_python, sandbox_available

    if not sandbox_available():
        pytest.skip("bubblewrap not usable here")
    secret = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert secret.exists()
    code = f"import os\nassert not os.path.exists({str(secret)!r}), 'secret visible'\n"
    ok, detail = _run_python(code, "", 10)
    assert ok, detail
    net = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), 2)\nexcept OSError:\n    pass\nelse:\n    raise SystemExit('network reachable')\n"
    ok, detail = _run_python(net, "", 10)
    assert ok, detail


def test_python_tests_still_pass_and_fail_correctly():
    from hellmholtz.benchmark.checks import _run_python

    assert _run_python("def f(): return 2", "assert f() == 2", 10)[0]
    assert not _run_python("def f(): return 3", "assert f() == 2", 10)[0]
    assert not _run_python("while True: pass", "", 2)[0]
