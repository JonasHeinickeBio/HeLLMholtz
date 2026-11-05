"""Speed probe, reasoning-token capture and the speed fields of the leaderboard."""

from types import SimpleNamespace
from unittest.mock import patch

from hellmholtz.benchmark.runner import count_reasoning_tokens
from hellmholtz.benchmark.scoring import leaderboard, speed_score
from hellmholtz.benchmark.speed import is_speed_record, probe_models, probe_once


def _chunk(text=None, reasoning=None, usage=None):
    delta = SimpleNamespace(content=text, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(chunks)))


def _patched(chunks):
    return patch(
        "hellmholtz.benchmark.speed.ClientManager.get_client",
        return_value=(_FakeClient(chunks), "m"),
    )


def test_probe_measures_ttft_and_throughput():
    usage = SimpleNamespace(completion_tokens=50, prompt_tokens=20)
    chunks = [_chunk(None), _chunk("one, "), _chunk("two, "), _chunk(None, usage=usage)]
    with _patched(chunks):
        r = probe_once("blablador:x")
    assert r.success and r.category == "speed" and r.prompt_id == "speed_probe"
    assert r.ttft_seconds is not None and r.ttft_seconds >= 0
    assert r.output_tokens == 50 and r.input_tokens == 20
    assert is_speed_record(r) and is_speed_record({"category": "speed"})


def test_probe_counts_reasoning_stream_as_first_token():
    with _patched([_chunk(None, reasoning="thinking..."), _chunk("answer")]):
        r = probe_once("blablador:x")
    assert r.success and r.ttft_seconds is not None


def test_probe_failure_is_a_failed_result_not_an_exception():
    with patch(
        "hellmholtz.benchmark.speed.ClientManager.get_client", side_effect=RuntimeError("down")
    ):
        r = probe_once("blablador:x")
    assert not r.success and "down" in (r.error_message or "") and r.ttft_seconds is None


def test_probe_models_runs_each_model_twice():
    with _patched([_chunk("a"), _chunk("b")]):
        results = probe_models(["blablador:a", "blablador:b"], runs=2, workers=2)
    assert len(results) == 4 and {r.run_id for r in results} == {0, 1}


def test_reasoning_tokens_prefers_provider_count():
    usage = SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens=321))
    resp = SimpleNamespace(
        usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="x"))]
    )
    assert count_reasoning_tokens(resp) == 321


def test_reasoning_tokens_estimated_from_content():
    msg = SimpleNamespace(content="<think>" + "a" * 400 + "</think>42", reasoning_content=None)
    resp = SimpleNamespace(usage=None, choices=[SimpleNamespace(message=msg)])
    assert 100 <= (count_reasoning_tokens(resp) or 0) <= 105
    plain = SimpleNamespace(
        usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content="42"))]
    )
    assert count_reasoning_tokens(plain) is None


def test_speed_score_bounds_and_monotonic():
    assert speed_score(0.3, 200) == 100.0 and speed_score(10, 10) == 0.0
    assert speed_score(1, 100) > speed_score(3, 30)
    assert speed_score(None, None) is None


def _rec(**kw):
    base = {
        "model": "blablador:m",
        "prompt_id": "p1",
        "category": "math",
        "success": True,
        "final_score": 80.0,
        "latency_seconds": 2.0,
        "tokens_per_sec": 30.0,
        "output_tokens": 100,
        "finish_reason": "stop",
    }
    return {**base, **kw}


def test_leaderboard_keeps_probe_out_of_quality_and_reports_speed():
    recs = [
        _rec(reasoning_tokens=60),
        _rec(prompt_id="p2", finish_reason="length"),
        _rec(
            prompt_id="speed_probe",
            category="speed",
            final_score=None,
            ttft_seconds=0.5,
            tokens_per_sec=100.0,
        ),
        _rec(
            prompt_id="speed_probe",
            category="speed",
            final_score=None,
            ttft_seconds=0.7,
            tokens_per_sec=120.0,
        ),
    ]
    [row] = leaderboard(recs)
    assert row["score"] == 80.0 and "speed" not in row["categories"] and row["runs"] == 2
    assert row["ttft_s"] == 0.6 and row["decode_tps"] == 110.0
    assert row["speed_score"] == speed_score(0.6, 110.0)
    assert row["reasoning_share"] == 60.0 and row["truncated_pct"] == 50.0
