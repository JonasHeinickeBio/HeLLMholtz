"""
Tests for reporting functionality.

This module contains comprehensive tests for the reporting modules,
including unit tests for markdown, HTML, and statistics generation.
"""

from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.benchmark import BenchmarkResult
from hellmholtz.reporting.markdown import generate_markdown_report, summarize_results
from hellmholtz.reporting.stats import (
    analyze_performance_trends,
    calculate_confidence_interval,
    calculate_model_stats,
    calculate_overall_stats,
    calculate_statistical_significance,
    detect_outliers,
    generate_insights,
)


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
        assert stats["avg_latency_all"] == pytest.approx(1.5, rel=1e-2)  # (1.0 + 2.0) / 2 = 1.5

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

    def test_calculate_confidence_interval(self) -> None:
        """Test confidence interval calculation."""
        from hellmholtz.reporting.stats import calculate_confidence_interval

        # Test with sufficient data
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = calculate_confidence_interval(data)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower < upper

        # Test with insufficient data
        small_data = [1.0]
        lower, upper = calculate_confidence_interval(small_data)
        assert lower != lower  # NaN
        assert upper != upper  # NaN

    def test_calculate_statistical_significance(self) -> None:
        """Test statistical significance calculation."""
        from hellmholtz.reporting.stats import calculate_statistical_significance

        group1 = [1.0, 2.0, 3.0]
        group2 = [2.0, 3.0, 4.0]
        result = calculate_statistical_significance(group1, group2)

        assert isinstance(result, dict)
        assert "significant" in result
        assert "p_value" in result
        assert "effect_size" in result

        # Test with insufficient data
        small_group1 = [1.0]
        small_group2 = [2.0]
        result = calculate_statistical_significance(small_group1, small_group2)
        assert result["significant"] is False

    def test_detect_outliers(self) -> None:
        """Test outlier detection."""
        from hellmholtz.reporting.stats import detect_outliers

        data = [1.0, 2.0, 3.0, 4.0, 100.0]  # 100 is an outlier
        outliers = detect_outliers(data, method="iqr")

        assert isinstance(outliers, list)
        assert 4 in outliers  # Index of the outlier

    def test_analyze_performance_trends(self, sample_results: list[BenchmarkResult]) -> None:
        """Test performance trends analysis."""
        trends = analyze_performance_trends(sample_results)

        assert isinstance(trends, dict)
        # Check for expected keys based on function implementation
        expected_keys = ["model_comparisons", "outlier_count", "outlier_percentage"]
        for key in expected_keys:
            assert key in trends

    def test_generate_insights(self, sample_results: list[BenchmarkResult]) -> None:
        """Test insights generation."""
        insights = generate_insights(sample_results)

        assert isinstance(insights, list)
        # Should generate some insights
        assert len(insights) > 0


class TestHTMLReporting:
    """Test suite for HTML reporting functions."""

    @pytest.fixture
    def sample_results(self) -> list[BenchmarkResult]:
        """Create sample benchmark results for HTML testing."""
        return [
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test_1",
                response_text="Response 1",
                latency_seconds=1.5,
                success=True,
                timestamp="2024-01-01T12:00:00",
                temperature=0.7,
                input_tokens=10,
                output_tokens=20,
            ),
            BenchmarkResult(
                model="openai:gpt-4o",
                prompt_id="test_2",
                response_text="Response 2",
                latency_seconds=2.0,
                success=True,
                timestamp="2024-01-01T12:00:01",
                temperature=1.0,
                input_tokens=15,
                output_tokens=25,
            ),
            BenchmarkResult(
                model="anthropic:claude-3",
                prompt_id="test_1",
                response_text="Response 3",
                latency_seconds=1.8,
                success=True,
                timestamp="2024-01-01T12:00:02",
                temperature=0.7,
                input_tokens=12,
                output_tokens=18,
            ),
        ]

    @patch("hellmholtz.reporting.html._load_template")
    def test_generate_html_report_simple(self, mock_load_template: MagicMock, sample_results: list[BenchmarkResult]) -> None:
        """Test simple HTML report generation."""
        from hellmholtz.reporting.html import generate_html_report_simple

        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Simple Report</html>"
        mock_load_template.return_value = mock_template

        result = generate_html_report_simple(sample_results)

        assert result == "<html>Simple Report</html>"
        mock_load_template.assert_called_once_with("simple")
        mock_template.render.assert_called_once()

    @patch("hellmholtz.reporting.html._load_template")
    def test_generate_html_report_detailed(self, mock_load_template: MagicMock, sample_results: list[BenchmarkResult]) -> None:
        """Test detailed HTML report generation."""
        from hellmholtz.reporting.html import generate_html_report_detailed

        mock_template = MagicMock()
        mock_template.render.return_value = "<html>Detailed Report</html>"
        mock_load_template.return_value = mock_template

        result = generate_html_report_detailed(sample_results)

        assert result == "<html>Detailed Report</html>"
        mock_load_template.assert_called_once_with("detailed")
        mock_template.render.assert_called_once()

    def test_generate_html_report_simple_empty_results(self) -> None:
        """Test simple HTML report with empty results."""
        from hellmholtz.reporting.html import generate_html_report_simple

        result = generate_html_report_simple([])

        assert result == "<p>No results to summarize.</p>"

    def test_generate_html_report_detailed_empty_results(self) -> None:
        """Test detailed HTML report with empty results."""
        from hellmholtz.reporting.html import generate_html_report_detailed

        result = generate_html_report_detailed([])

        assert result == "<p>No results to summarize.</p>"

    def test_generate_html_report_alias(self, sample_results: list[BenchmarkResult]) -> None:
        """Test that generate_html_report is an alias for detailed."""
        from hellmholtz.reporting.html import generate_html_report, generate_html_report_detailed

        # Both should return the same result
        result1 = generate_html_report(sample_results)
        result2 = generate_html_report_detailed(sample_results)

        assert result1 == result2

    def test_generate_html_report_full_placeholder(self, sample_results: list[BenchmarkResult]) -> None:
        """Test full HTML report placeholder."""
        from hellmholtz.reporting.html import generate_html_report_full

        result = generate_html_report_full(sample_results)

        assert result == "<p>Full report not yet implemented in modular format.</p>"

    @patch("hellmholtz.reporting.html._load_template")
    def test_load_template(self, mock_load_template: MagicMock) -> None:
        """Test template loading."""
        from hellmholtz.reporting.html import _load_template

        mock_template = MagicMock()
        mock_load_template.return_value = mock_template

        result = _load_template("test_template")

        assert result == mock_template
        mock_load_template.assert_called_once_with("test_template")

    def test_prepare_simple_report_data(self, sample_results: list[BenchmarkResult]) -> None:
        """Test preparation of simple report data."""
        from hellmholtz.reporting.html import _prepare_simple_report_data

        data = _prepare_simple_report_data(sample_results)

        assert isinstance(data, dict)
        assert "total_runs" in data
        assert "success_rate" in data
        assert "models_count" in data
        assert "temperatures_count" in data
        assert "timestamp" in data
        assert "model_labels" in data
        assert "success_rates" in data
        assert "avg_latencies" in data

        # Check calculated values
        assert data["total_runs"] == 3
        assert data["models_count"] == 2  # openai:gpt-4o and anthropic:claude-3
        assert data["temperatures_count"] == 2  # 0.7 and 1.0


class TestReportingUtils:
    """Test suite for reporting utility functions."""

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
                input_tokens=10,
                output_tokens=20,
                error_message="",
                temperature=0.7,
                max_tokens=100,
                run_id="run_1",
            ),
            BenchmarkResult(
                model="anthropic:claude-3",
                prompt_id="test_2",
                response_text="Response 2",
                latency_seconds=2.0,
                success=False,
                timestamp="2024-01-01T12:00:01",
                input_tokens=15,
                output_tokens=0,
                error_message="API Error",
                temperature=1.0,
                max_tokens=200,
                run_id="run_2",
            ),
        ]

    def test_export_to_csv_success(self, sample_results: list[BenchmarkResult], tmp_path) -> None:
        """Test successful CSV export."""
        from hellmholtz.reporting.utils import export_to_csv

        csv_file = tmp_path / "test_results.csv"
        export_to_csv(sample_results, str(csv_file))

        # Verify file was created and contains expected content
        assert csv_file.exists()

        content = csv_file.read_text()
        lines = content.strip().split("\n")

        # Check header
        assert lines[0] == "model,prompt_id,latency_seconds,success,timestamp,input_tokens,output_tokens,error_message,temperature,max_tokens,run_id"

        # Check data rows
        assert len(lines) == 3  # header + 2 data rows
        assert "openai:gpt-4o,test_1,1.5,True,2024-01-01T12:00:00,10,20,,0.7,100,run_1" in lines[1]
        assert "anthropic:claude-3,test_2,2.0,False,2024-01-01T12:00:01,15,0,API Error,1.0,200,run_2" in lines[2]

    def test_export_to_csv_empty_results(self, tmp_path) -> None:
        """Test CSV export with empty results list."""
        from hellmholtz.reporting.utils import export_to_csv

        csv_file = tmp_path / "empty_results.csv"
        export_to_csv([], str(csv_file))

        # Verify file was created with only header
        assert csv_file.exists()
        content = csv_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1  # only header
        assert lines[0] == "model,prompt_id,latency_seconds,success,timestamp,input_tokens,output_tokens,error_message,temperature,max_tokens,run_id"

    def test_load_results_success(self, sample_results: list[BenchmarkResult], tmp_path) -> None:
        """Test successful loading of results from JSON."""
        import json
        from hellmholtz.reporting.utils import load_results

        # Create test JSON file
        json_file = tmp_path / "test_results.json"
        json_data = [
            {
                "model": "openai:gpt-4o",
                "prompt_id": "test_1",
                "response_text": "Response 1",
                "latency_seconds": 1.5,
                "success": True,
                "timestamp": "2024-01-01T12:00:00",
                "input_tokens": 10,
                "output_tokens": 20,
                "error_message": "",
                "temperature": 0.7,
                "max_tokens": 100,
                "run_id": "run_1",
            },
            {
                "model": "anthropic:claude-3",
                "prompt_id": "test_2",
                "response_text": "Response 2",
                "latency_seconds": 2.0,
                "success": False,
                "timestamp": "2024-01-01T12:00:01",
                "input_tokens": 15,
                "output_tokens": 0,
                "error_message": "API Error",
                "temperature": 1.0,
                "max_tokens": 200,
                "run_id": "run_2",
            },
        ]

        with open(json_file, "w") as f:
            json.dump(json_data, f)

        # Load and verify results
        loaded_results = load_results(str(json_file))

        assert len(loaded_results) == 2
        assert loaded_results[0].model == "openai:gpt-4o"
        assert loaded_results[0].prompt_id == "test_1"
        assert loaded_results[0].latency_seconds == 1.5
        assert loaded_results[0].success is True
        assert loaded_results[1].model == "anthropic:claude-3"
        assert loaded_results[1].success is False
        assert loaded_results[1].error_message == "API Error"

    def test_load_results_invalid_json_structure(self, tmp_path) -> None:
        """Test loading results with invalid JSON structure."""
        import json
        from hellmholtz.reporting.utils import load_results

        json_file = tmp_path / "invalid_results.json"

        # Test with non-list JSON
        with open(json_file, "w") as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="Expected a JSON list of results"):
            load_results(str(json_file))

        # Test with list containing non-dict items
        with open(json_file, "w") as f:
            json.dump(["not", "dicts"], f)

        with pytest.raises(ValueError, match="Expected each result item to be an object"):
            load_results(str(json_file))

    def test_load_results_file_not_found(self) -> None:
        """Test loading results from non-existent file."""
        from hellmholtz.reporting.utils import load_results

        with pytest.raises(FileNotFoundError):
            load_results("non_existent_file.json")
