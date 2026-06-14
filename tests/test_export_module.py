"""Tests for hellmholtz.export module (select_best_model, get_default_model_config)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hellmholtz.benchmark.runner import BenchmarkResult
from hellmholtz.export import get_default_model_config, select_best_model


def _result(
    model: str = "m1",
    latency: float = 0.5,
    success: bool = True,
    prompt_id: str = "p1",
) -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        prompt_id=prompt_id,
        temperature=0.1,
        run_id=1,
        response_text="ok",
        success=success,
        input_tokens=10,
        output_tokens=20,
        latency_seconds=latency,
        timestamp="2024-01-01T00:00:00",
    )


def _write_results(path: Path, results: list[BenchmarkResult]) -> Path:
    from dataclasses import asdict

    path.write_text(json.dumps([asdict(r) for r in results]))
    return path


# ── select_best_model ─────────────────────────────────────────────────────────


class TestSelectBestModel:
    def test_by_latency(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [
            _result("fast", latency=0.1),
            _result("slow", latency=1.0),
        ])
        best = select_best_model(str(f), criterion="latency")
        assert best["model"] == "fast"

    def test_by_success_rate(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [
            _result("good", success=True, prompt_id="p1"),
            _result("good", success=True, prompt_id="p2"),
            _result("bad", success=False, prompt_id="p1"),
            _result("bad", success=True, prompt_id="p2"),
        ])
        best = select_best_model(str(f), criterion="success_rate")
        assert best["model"] == "good"

    def test_invalid_criterion_raises(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [_result()])
        with pytest.raises(ValueError, match="Invalid criterion"):
            select_best_model(str(f), criterion="token_efficiency")

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [])
        with pytest.raises(ValueError, match="No results found"):
            select_best_model(str(f), criterion="latency")

    def test_all_failed_returns_first_model(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [
            _result("model-a", success=False),
            _result("model-b", success=False),
        ])
        best = select_best_model(str(f), criterion="latency")
        assert best["model"] in ["model-a", "model-b"]

    def test_latency_ignores_failed_runs(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [
            _result("slow-but-ok", latency=2.0, success=True),
            _result("fast-but-fails", latency=0.1, success=False),
        ])
        best = select_best_model(str(f), criterion="latency")
        assert best["model"] == "slow-but-ok"

    def test_returns_model_key(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "r.json", [_result("x")])
        best = select_best_model(str(f), criterion="latency")
        assert "model" in best


# ── get_default_model_config ──────────────────────────────────────────────────


class TestGetDefaultModelConfig:
    def test_no_results_dir_returns_fallback(self, tmp_path: Path) -> None:
        result = get_default_model_config(str(tmp_path))
        assert result == {"model": "openai:gpt-4o"}

    def test_no_benchmark_files_returns_fallback(self, tmp_path: Path) -> None:
        (tmp_path / "other.json").write_text("[]")
        result = get_default_model_config(str(tmp_path))
        assert result == {"model": "openai:gpt-4o"}

    def test_with_benchmark_file(self, tmp_path: Path) -> None:
        f = _write_results(tmp_path / "benchmark_2024-01-01.json", [
            _result("best-model", latency=0.1),
            _result("worst-model", latency=1.0),
        ])
        result = get_default_model_config(str(tmp_path))
        assert result["model"] == "best-model"

    def test_multiple_benchmark_files_uses_latest(self, tmp_path: Path) -> None:
        f1 = _write_results(tmp_path / "benchmark_old.json", [_result("old-model", latency=1.0)])
        f2 = _write_results(tmp_path / "benchmark_new.json", [_result("new-model", latency=0.5)])
        with patch("hellmholtz.export.os.path.getctime") as mock_ctime:
            mock_ctime.side_effect = lambda p: 100 if "old" in str(p) else 200
            result = get_default_model_config(str(tmp_path))
        assert result["model"] == "new-model"

    @patch("hellmholtz.export.select_best_model")
    def test_select_best_model_error_returns_fallback(self, mock_select: MagicMock, tmp_path: Path) -> None:
        from unittest.mock import MagicMock
        mock_select.side_effect = Exception("boom")
        _write_results(tmp_path / "benchmark_2024-01-01.json", [_result()])
        result = get_default_model_config(str(tmp_path))
        assert result == {"model": "openai:gpt-4o"}
