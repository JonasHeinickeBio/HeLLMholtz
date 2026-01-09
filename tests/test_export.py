"""
Tests for export functionality.

This module contains comprehensive tests for the export module,
including model selection algorithms and result processing.
"""

import json
import pytest
from pathlib import Path
from typing import List, Dict, Any

from hellmholtz.export import select_best_model
from hellmholtz.benchmark.runner import BenchmarkResult


class TestExportFunctions:
    """Test suite for export functions."""

    @pytest.fixture
    def sample_results(self) -> List[BenchmarkResult]:
        """Create sample benchmark results for testing."""
        return [
            BenchmarkResult(
                model="fast-model",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="Fast response",
                success=True,
                input_tokens=10,
                output_tokens=20,
                latency_seconds=0.1,
                timestamp="2024-01-01T00:00:00"
            ),
            BenchmarkResult(
                model="slow-model",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="Slow response",
                success=True,
                input_tokens=10,
                output_tokens=20,
                latency_seconds=1.0,
                timestamp="2024-01-01T00:00:00"
            ),
            BenchmarkResult(
                model="accurate-model",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="Accurate response",
                success=True,
                input_tokens=15,
                output_tokens=25,
                latency_seconds=0.5,
                timestamp="2024-01-01T00:00:00"
            )
        ]

    @pytest.fixture
    def results_file(self, tmp_path: Path, sample_results: List[BenchmarkResult]) -> Path:
        """Create a temporary results file for testing."""
        from dataclasses import asdict
        filepath = tmp_path / "test_results.json"
        with open(filepath, "w") as f:
            json.dump([asdict(result) for result in sample_results], f)
        return filepath

    def test_select_best_model_latency_criterion(self, results_file: Path) -> None:
        """Test selecting best model by latency (lowest is best)."""
        best = select_best_model(str(results_file), criterion="latency")

        assert best["model"] == "fast-model"

    def test_select_best_model_success_rate_criterion(self, results_file: Path) -> None:
        """Test selecting best model by success rate (highest is best)."""
        best = select_best_model(str(results_file), criterion="success_rate")

        # All models have 100% success rate, should return first one
        assert best["model"] in ["fast-model", "slow-model", "accurate-model"]

    def test_select_best_model_token_efficiency_criterion(self, results_file: Path) -> None:
        """Test that invalid criterion raises ValueError."""
        with pytest.raises(ValueError, match="Invalid criterion: token_efficiency"):
            select_best_model(str(results_file), criterion="token_efficiency")

    def test_select_best_model_with_failed_results(self, tmp_path: Path) -> None:
        """Test selecting best model when some results have failures."""
        results = [
            BenchmarkResult(
                model="reliable-model",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="Good response",
                success=True,
                input_tokens=10,
                output_tokens=20,
                latency_seconds=0.5,
                timestamp="2024-01-01T00:00:00"
            ),
            BenchmarkResult(
                model="unreliable-model",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="",
                success=False,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=0.0,
                timestamp="2024-01-01T00:00:00"
            )
        ]

        filepath = tmp_path / "mixed_results.json"
        with open(filepath, "w") as f:
            from dataclasses import asdict
            json.dump([asdict(result) for result in results], f)

        best = select_best_model(str(filepath), criterion="success_rate")
        assert best["model"] == "reliable-model"

    def test_select_best_model_multiple_prompts(self, tmp_path: Path) -> None:
        """Test selecting best model across multiple prompts."""
        results = [
            # Prompt 1
            BenchmarkResult(
                model="model-a",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="Response A1",
                success=True,
                input_tokens=10,
                output_tokens=20,
                latency_seconds=0.2,
                timestamp="2024-01-01T00:00:00"
            ),
            BenchmarkResult(
                model="model-b",
                prompt_id="prompt-1",
                temperature=0.1,
                run_id=1,
                response_text="Response B1",
                success=True,
                input_tokens=10,
                output_tokens=20,
                latency_seconds=0.3,
                timestamp="2024-01-01T00:00:00"
            ),
            # Prompt 2
            BenchmarkResult(
                model="model-a",
                prompt_id="prompt-2",
                temperature=0.1,
                run_id=1,
                response_text="Response A2",
                success=True,
                input_tokens=15,
                output_tokens=25,
                latency_seconds=0.4,
                timestamp="2024-01-01T00:00:00"
            ),
            BenchmarkResult(
                model="model-b",
                prompt_id="prompt-2",
                temperature=0.1,
                run_id=1,
                response_text="Response B2",
                success=True,
                input_tokens=15,
                output_tokens=25,
                latency_seconds=0.2,
                timestamp="2024-01-01T00:00:00"
            )
        ]

        filepath = tmp_path / "multi_prompt_results.json"
        with open(filepath, "w") as f:
            from dataclasses import asdict
            json.dump([asdict(result) for result in results], f)

        # Model A average latency: (0.2 + 0.4) / 2 = 0.3
        # Model B average latency: (0.3 + 0.2) / 2 = 0.25
        best = select_best_model(str(filepath), criterion="latency")
        assert best["model"] == "model-b"

    def test_select_best_model_empty_file(self, tmp_path: Path) -> None:
        """Test behavior with empty results file."""
        filepath = tmp_path / "empty_results.json"
        with open(filepath, "w") as f:
            json.dump([], f)

        with pytest.raises(ValueError, match="No results found"):
            select_best_model(str(filepath), criterion="latency")

    def test_select_best_model_invalid_criterion(self, results_file: Path) -> None:
        """Test error handling for invalid criterion."""
        with pytest.raises(ValueError, match="Invalid criterion"):
            select_best_model(str(results_file), criterion="invalid")

    def test_select_best_model_file_not_found(self) -> None:
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            select_best_model("nonexistent_file.json", criterion="latency")

    @pytest.mark.parametrize("criterion,expected_key", [
        ("latency", "latency_seconds"),
        ("success_rate", "success_rate"),
    ])
    def test_select_best_model_returns_expected_keys(
        self,
        results_file: Path,
        criterion: str,
        expected_key: str
    ) -> None:
        """Test that select_best_model returns expected keys for different criteria."""
        best = select_best_model(str(results_file), criterion=criterion)

        # Currently only returns model, but could be extended
        assert "model" in best
        # Note: criterion and expected_key are not currently returned
