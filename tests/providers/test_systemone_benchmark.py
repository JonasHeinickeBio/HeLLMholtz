"""Offline tests for the System-One reliability benchmark.

These tests exercise the benchmark pipeline with an injected fake ``route_fn``
so no network / API key is required.
"""

from __future__ import annotations

import json

import pytest

from hellmholtz.providers.systemone import (
    SystemOneAnswer,
    SystemOneError,
    SystemOneQuestion,
    SystemOneResponse,
    SystemOneRouting,
)
from hellmholtz.providers.systemone_benchmark import (
    DEFAULT_SCENARIOS,
    DecisionScenario,
    format_summary,
    load_scenarios,
    run_systemone_benchmark,
    write_reports,
)


def _answer(choice: str, confidence: float = 0.8) -> SystemOneAnswer:
    return SystemOneAnswer(
        type="choice",
        choice=choice,
        probabilities={choice: 0.9, "other": 0.1},
        confidence=confidence,
    )


def _response(
    choice_map: dict[str, str],
    routing_model: str = "laya-rl-agent",
    repo: str = "convaiinnovations/laya",
) -> SystemOneResponse:
    return SystemOneResponse(
        model="alias-laya",
        answers={name: _answer(choice) for name, choice in choice_map.items()},
        routing=SystemOneRouting(model=routing_model, repo=repo),
    )


def _single_question_scenario() -> DecisionScenario:
    return DecisionScenario(
        name="s1",
        state="A decision state.",
        questions={
            "q1": SystemOneQuestion(
                type="choice",
                instructions="Pick an option.",
                criteria={"yes": "yes option", "no": "no option"},
            )
        },
    )


def test_default_scenarios_present() -> None:
    assert DEFAULT_SCENARIOS
    for scenario in DEFAULT_SCENARIOS:
        assert scenario.state.strip()
        assert scenario.questions
        for question in scenario.questions.values():
            assert question.instructions
            assert question.criteria


def test_run_all_deterministic_is_fully_stable() -> None:
    def fake_route(state, questions, model=None, timeout=None):
        return _response({name: "yes" for name in questions})

    report = run_systemone_benchmark(
        [_single_question_scenario()],
        model="m",
        replications=4,
        route_fn=fake_route,
        endpoint="http://fake",
        progress=False,
    )

    assert report.total_attempts == 4
    assert report.overall_success_rate == 1.0
    assert report.overall_stability == 1.0
    assert report.all_stable is True

    scen = report.scenarios[0]
    assert scen.successful == 4
    assert scen.errors == 0
    assert len(scen.latencies_ms) == 4
    assert scen.stable is True
    assert scen.routing_model == "laya-rl-agent"
    assert scen.routing_repo == "convaiinnovations/laya"

    q = scen.questions[0]
    assert q.choice_counts == {"yes": 4}
    assert q.stability == 1.0
    assert q.majority_choice == "yes"
    assert q.answer_rate == 1.0
    assert q.mean_confidence == pytest.approx(0.8)


def test_run_records_errors_and_mixed_choices() -> None:
    calls = {"n": 0}

    def fake_route(state, questions, model=None, timeout=None):
        calls["n"] += 1
        n = calls["n"]
        if n % 3 == 0:
            raise SystemOneError("transient failure")
        choice = "yes" if n % 2 == 1 else "no"
        return _response({"q1": choice})

    report = run_systemone_benchmark(
        [_single_question_scenario()],
        model="m",
        replications=6,
        route_fn=fake_route,
        endpoint="http://fake",
        progress=False,
    )

    scen = report.scenarios[0]
    assert scen.attempts == 6
    # calls 3 and 6 fail -> 2 errors, 4 successes
    assert scen.errors == 2
    assert scen.successful == 4
    assert scen.success_rate == pytest.approx(4 / 6)
    assert scen.stable is False

    q = scen.questions[0]
    assert q.attempts == 6
    # successes at n=1(yes),2(no),4(no),5(yes) -> yes:2, no:2
    assert q.choice_counts == {"yes": 2, "no": 2}
    assert q.stability == pytest.approx(0.5)
    assert q.no_answer == 2
    assert q.answer_rate == pytest.approx(4 / 6)


def test_run_missing_answer_counts_as_no_answer() -> None:
    def fake_route(state, questions, model=None, timeout=None):
        # A successful response that omits the requested question entirely.
        return SystemOneResponse(model="alias-laya", answers={})

    report = run_systemone_benchmark(
        [_single_question_scenario()],
        model="m",
        replications=3,
        route_fn=fake_route,
        endpoint="http://fake",
        progress=False,
    )

    scen = report.scenarios[0]
    assert scen.successful == 3
    assert scen.errors == 0
    q = scen.questions[0]
    assert q.attempts == 3
    assert q.no_answer == 3
    assert q.choice_counts == {}
    assert q.stability == 0.0
    assert q.answer_rate == 0.0
    assert q.majority_choice is None


def test_report_to_dict_is_json_serializable() -> None:
    def fake_route(state, questions, model=None, timeout=None):
        return _response({name: "yes" for name in questions})

    report = run_systemone_benchmark(
        [_single_question_scenario()],
        model="m",
        replications=2,
        route_fn=fake_route,
        endpoint="http://fake",
        progress=False,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    data = json.loads(payload)
    assert data["model"] == "m"
    assert data["endpoint"] == "http://fake"
    assert len(data["scenarios"]) == 1
    assert data["scenarios"][0]["questions"][0]["choice_counts"] == {"yes": 2}


def test_format_summary_smoke() -> None:
    def fake_route(state, questions, model=None, timeout=None):
        return _response({name: "yes" for name in questions})

    report = run_systemone_benchmark(
        [_single_question_scenario()],
        model="m",
        replications=2,
        route_fn=fake_route,
        endpoint="http://fake",
        progress=False,
    )
    text = format_summary(report)
    assert "System-One reliability benchmark" in text
    assert "s1" in text
    assert "http://fake" in text


def test_write_reports_creates_files(tmp_path) -> None:
    def fake_route(state, questions, model=None, timeout=None):
        return _response({name: "yes" for name in questions})

    report = run_systemone_benchmark(
        [_single_question_scenario()],
        model="m",
        replications=2,
        route_fn=fake_route,
        endpoint="http://fake",
        progress=False,
    )
    md_path, json_path = write_reports(report, out_dir=tmp_path)

    assert md_path.exists()
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["model"] == "m"
    assert len(data["scenarios"]) == 1
    md_text = md_path.read_text(encoding="utf-8")
    assert "System-One reliability benchmark" in md_text
    assert "s1" in md_text


def test_load_scenarios_from_json(tmp_path) -> None:
    payload = [
        {
            "name": "js1",
            "state": "A JSON state.",
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Pick?",
                    "criteria": {"yes": "y", "no": "n"},
                }
            },
        }
    ]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scenarios = load_scenarios(path)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.name == "js1"
    assert "q1" in scenario.questions
    assert scenario.questions["q1"].criteria == {"yes": "y", "no": "n"}
    assert scenario.questions["q1"].type == "choice"


def test_load_scenarios_missing_file(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_scenarios(tmp_path / "nope.json")


def test_load_scenarios_rejects_non_list(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenarios(path)


def test_load_scenarios_rejects_missing_state(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps([{"name": "x", "state": "", "questions": {"q": {"instructions": "i"}}}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_scenarios(path)


def test_load_scenarios_rejects_no_questions(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"name": "x", "state": "s", "questions": {}}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenarios(path)
