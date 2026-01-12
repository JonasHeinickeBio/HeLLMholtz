"""
Tests for reporting functionality.

This module contains comprehensive tests for the reporting modules,
including unit tests for markdown, HTML, and statistics generation.
"""


import pytest

from hellmholtz.benchmark import BenchmarkResult
from hellmholtz.reporting.markdown import generate_markdown_report, summarize_results
from hellmholtz.reporting.stats import calculate_model_stats, calculate_overall_stats


class TestMarkdownReporting:
    """Test suite for markdown reporting functions."""

    @pytest.fixture
    def sample_results(self) -> list[BenchmarkResult]:
        """Create sample benchmark results for testing."""
        return [
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test_1",
                response_text="Response 1",
                latency_seconds=1.5,
                success=True,
                timestamp="2024-01-01T12:00:00",
            ),
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test_2",
                response_text="Response 2",
                latency_seconds=2.0,
                success=True,
                timestamp="2024-01-01T12:00:01",
            ),
            BenchmarkResult(
                model="anthropic:claude-3",
                prompt_id="test_1",
                response_text="Response 3",
                latency_seconds=1.8,
                success=False,
                timestamp="2024-01-01T12:00:02",
            ),
            BenchmarkResult(
                model="anthropic:claude-3",
                prompt_id="test_2",
                response_text="Response 4",
                latency_seconds=2.2,
                success=True,
                timestamp="2024-01-01T12:00:03",
            ),
        ]

    def test_generate_markdown_report_empty(self) -> None:
        """Test markdown report generation with empty results."""
        result = generate_markdown_report([])
        assert result == "No results to summarize."

    def test_generate_markdown_report_basic(self, sample_results: list[BenchmarkResult]) -> None:
        """Test basic markdown report generation."""
        result = generate_markdown_report(sample_results)

        assert "# Benchmark Summary" in result
        assert "**Total Runs**: 4" in result
        assert "openai:gpt-4o" in result
        assert "anthropic:claude-3" in result
        assert "| Model | Success Rate | Avg Latency (s) |" in result

    def test_generate_markdown_report_success_rates(
        self, sample_results: list[BenchmarkResult]
    ) -> None:
        """Test that success rates are calculated correctly."""
        result = generate_markdown_report(sample_results)

        # openai:gpt-4o should have 100% success rate (2/2)
        assert "100.0%" in result
        # anthropic:claude-3 should have 50% success rate (1/2)
        assert "50.0%" in result

    def test_generate_markdown_report_latency(self, sample_results: list[BenchmarkResult]) -> None:
        """Test that average latency is calculated correctly."""
        result = generate_markdown_report(sample_results)

        # openai:gpt-4o average: (1.5 + 2.0) / 2 = 1.75
        assert "1.7500" in result
        # anthropic:claude-3 average: (1.8 + 2.2) / 2 = 2.0
        assert "2.0000" in result

    def test_summarize_results_alias(self, sample_results: list[BenchmarkResult]) -> None:
        """Test that summarize_results is an alias for generate_markdown_report."""
        markdown_result = generate_markdown_report(sample_results)
        summary_result = summarize_results(sample_results)

        assert markdown_result == summary_result


class TestStatisticsReporting:
    """Test suite for statistics reporting functions."""

    @pytest.fixture
    def sample_results(self) -> list[BenchmarkResult]:
        """Create sample benchmark results for testing."""
        return [
            BenchmarkResult(
                model="model_a",
                prompt_id="prompt_1",
                response_text="Response 1",
                latency_seconds=1.0,
                success=True,
                timestamp="2024-01-01T12:00:00",
            ),
            BenchmarkResult(
                model="model_a",
                prompt_id="prompt_2",
                response_text="Response 2",
                latency_seconds=2.0,
                success=True,
                timestamp="2024-01-01T12:00:01",
            ),
            BenchmarkResult(
                model="model_b",
                prompt_id="prompt_1",
                response_text="Response 3",
                latency_seconds=1.5,
                success=False,
                timestamp="2024-01-01T12:00:02",
            ),
        ]

    def test_calculate_model_stats(self, sample_results: list[BenchmarkResult]) -> None:
        """Test model statistics calculation."""
        stats = calculate_model_stats(sample_results)

        assert "model_a" in stats
        assert "model_b" in stats

        model_a_stats = stats["model_a"]
        assert model_a_stats["total_runs"] == 2
        assert model_a_stats["success_rate"] == 1.0  # 2/2
        assert model_a_stats["avg_latency"] == 1.5  # (1.0 + 2.0) / 2

        model_b_stats = stats["model_b"]
        assert model_b_stats["total_runs"] == 1
        assert model_b_stats["success_rate"] == 0.0  # 0/1
        assert model_b_stats["avg_latency"] == 0.0  # No successful runs

    def test_calculate_overall_stats(self, sample_results: list[BenchmarkResult]) -> None:
        """Test overall statistics calculation."""
        stats = calculate_overall_stats(sample_results)

        assert stats["total_runs"] == 3
        assert stats["unique_models"] == 2
        assert stats["overall_success_rate"] == pytest.approx(2 / 3, rel=1e-2)  # 2/3 ≈ 0.667
        assert stats["avg_latency_all"] == pytest.approx(1.5, rel=1e-2)  # (1.0 + 2.0 + 1.5) / 3

    def test_calculate_model_stats_empty(self) -> None:
        """Test model stats calculation with empty results."""
        stats = calculate_model_stats([])
        assert stats == {}

    def test_calculate_overall_stats_empty(self) -> None:
        """Test overall stats calculation with empty results."""
        stats = calculate_overall_stats([])
        assert stats == {
            "total_runs": 0,
            "unique_models": 0,
            "overall_success_rate": 0.0,
            "avg_latency_all": 0.0,
        }
