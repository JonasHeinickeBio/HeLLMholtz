"""Tests for hellmholtz.reporting.chart module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hellmholtz.reporting.chart import (
    calculate_stats,
    generate_performance_chart,
    load_results,
    main,
)


class TestLoadResults:
    """Tests for load_results function."""

    def test_load_list_of_objects(self, tmp_path: Path) -> None:
        results = [{"model": "a", "score": 1}, {"model": "b", "score": 2}]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        loaded = load_results(str(f))
        assert loaded == results
        assert len(loaded) == 2

    def test_load_single_object_wrapped_in_list(self, tmp_path: Path) -> None:
        result = {"model": "a", "score": 1}
        f = tmp_path / "results.json"
        f.write_text(json.dumps(result))
        loaded = load_results(str(f))
        assert loaded == [result]
        assert isinstance(loaded, list)

    def test_load_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("[]")
        loaded = load_results(str(f))
        assert loaded == []

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_results(str(f))

    def test_load_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path.json")

    def test_load_numeric_list(self, tmp_path: Path) -> None:
        f = tmp_path / "nums.json"
        f.write_text(json.dumps([1, 2, 3]))
        loaded = load_results(str(f))
        assert loaded == [1, 2, 3]


class TestCalculateStats:
    """Tests for calculate_stats function."""

    EXPECTED_KEYS = {
        "mean", "std", "min", "max", "median",
        "q25", "q75", "ci_lower", "ci_upper", "count",
    }

    def test_empty_list(self) -> None:
        result = calculate_stats([])
        assert result["count"] == 0
        assert result["mean"] == 0.0
        assert result["std"] == 0.0
        assert result["min"] == 0.0
        assert result["max"] == 0.0
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_single_value(self) -> None:
        result = calculate_stats([5.0])
        assert result["mean"] == 5.0
        assert result["std"] == 0.0
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["median"] == 5.0
        assert result["count"] == 1
        assert result["ci_lower"] == 5.0
        assert result["ci_upper"] == 5.0

    def test_multiple_values(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_stats(data)
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["median"] == 3.0
        assert result["count"] == 5
        assert result["std"] > 0
        assert result["q25"] < result["median"] < result["q75"]
        assert result["ci_lower"] < result["mean"] < result["ci_upper"]

    def test_two_values(self) -> None:
        result = calculate_stats([10.0, 20.0])
        assert result["mean"] == 15.0
        assert result["count"] == 2
        assert result["ci_lower"] < 15.0 < result["ci_upper"]

    def test_identical_values(self) -> None:
        data = [7.0, 7.0, 7.0]
        result = calculate_stats(data)
        assert result["mean"] == 7.0
        assert result["std"] == 0.0
        assert result["min"] == 7.0
        assert result["max"] == 7.0

    def test_negative_values(self) -> None:
        data = [-3.0, -1.0, 2.0]
        result = calculate_stats(data)
        assert result["min"] == -3.0
        assert result["max"] == 2.0

    def test_keys_always_present(self) -> None:
        for data in [[], [1.0], [1.0, 2.0, 3.0]]:
            result = calculate_stats(data)
            assert set(result.keys()) == self.EXPECTED_KEYS


class TestGeneratePerformanceChart:
    """Tests for generate_performance_chart function."""

    @pytest.fixture
    def sample_results_file(self, tmp_path: Path) -> Path:
        results = [
            {
                "model": "openai:gpt-4o",
                "latency_seconds": 1.5,
                "success": True,
                "input_tokens": 10,
                "output_tokens": 20,
            },
            {
                "model": "openai:gpt-4o",
                "latency_seconds": 2.0,
                "success": True,
                "input_tokens": 15,
                "output_tokens": 25,
            },
            {
                "model": "blablador:llama-3",
                "latency_seconds": 3.0,
                "success": False,
                "input_tokens": 5,
                "output_tokens": 10,
            },
        ]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        return f

    def test_generates_chart_without_crash(self, sample_results_file: Path, tmp_path: Path) -> None:
        output = str(tmp_path / "chart.png")
        generate_performance_chart(str(sample_results_file), output)
        assert Path(output).exists()

    def test_chart_is_valid_image(self, sample_results_file: Path, tmp_path: Path) -> None:
        output = str(tmp_path / "chart.png")
        generate_performance_chart(str(sample_results_file), output)
        with open(output, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"

    def test_single_model_results(self, tmp_path: Path) -> None:
        results = [
            {"model": "openai:gpt-4o", "latency_seconds": 1.0, "success": True,
             "input_tokens": 5, "output_tokens": 10},
        ]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        output = str(tmp_path / "chart.png")
        generate_performance_chart(str(f), output)
        assert Path(output).exists()

    def test_results_with_missing_optional_fields(self, tmp_path: Path) -> None:
        results = [
            {"model": "test:model", "latency_seconds": 0.5, "success": True},
        ]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        output = str(tmp_path / "chart.png")
        generate_performance_chart(str(f), output)
        assert Path(output).exists()

    def test_all_failed_requests(self, tmp_path: Path) -> None:
        results = [
            {"model": "a:model", "latency_seconds": 1.0, "success": False,
             "input_tokens": 0, "output_tokens": 0},
            {"model": "a:model", "latency_seconds": 2.0, "success": False,
             "input_tokens": 0, "output_tokens": 0},
        ]
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        output = str(tmp_path / "chart.png")
        generate_performance_chart(str(f), output)
        assert Path(output).exists()

    def test_multiple_models(self, tmp_path: Path) -> None:
        results = []
        for i, model in enumerate(["m1", "m2", "m3"]):
            results.append({
                "model": f"prov:{model}",
                "latency_seconds": float(i + 1),
                "success": True,
                "input_tokens": 10 * (i + 1),
                "output_tokens": 5 * (i + 1),
            })
        f = tmp_path / "results.json"
        f.write_text(json.dumps(results))
        output = str(tmp_path / "chart.png")
        generate_performance_chart(str(f), output)
        assert Path(output).exists()


class TestMain:
    """Tests for the main entry point."""

    @patch("hellmholtz.reporting.chart.generate_performance_chart")
    def test_main_with_valid_args(self, mock_gen: MagicMock) -> None:
        with patch("hellmholtz.reporting.chart.sys.argv", ["script", "input.json", "out.png"]):
            main()
        mock_gen.assert_called_once_with("input.json", "out.png")

    def test_main_wrong_arg_count(self) -> None:
        with patch("hellmholtz.reporting.chart.sys.argv", ["script", "only_one_arg"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_no_args(self) -> None:
        with patch("hellmholtz.reporting.chart.sys.argv", ["script"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
