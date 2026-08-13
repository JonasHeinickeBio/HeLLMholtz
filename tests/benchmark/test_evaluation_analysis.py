"""Tests for the HeLLMholtz evaluation analysis module."""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.evaluation_analysis import EvaluationAnalyzer


class TestEvaluationAnalyzer:
    """Test suite for EvaluationAnalyzer class."""

    @pytest.fixture
    def sample_evaluation_data(self) -> list[dict]:
        """Sample evaluation data for testing."""
        return [
            {
                "model": "openai:gpt-4o",
                "prompt_id": "test-1",
                "success": True,
                "rating": 8,
                "response_text": "This is a good response",
                "critique": "Well-structured answer",
                "latency_seconds": 1.5,
            },
            {
                "model": "openai:gpt-4o",
                "prompt_id": "test-2",
                "success": True,
                "rating": 9,
                "response_text": "Excellent response",
                "critique": "Very detailed",
                "latency_seconds": 2.0,
            },
            {
                "model": "anthropic:claude-3-haiku",
                "prompt_id": "test-1",
                "success": True,
                "rating": 7,
                "response_text": "Decent response",
                "critique": "Good but could be better",
                "latency_seconds": 1.8,
            },
        ]

    @pytest.fixture
    def temp_results_file(self, sample_evaluation_data: list[dict]) -> str:
        """Create a temporary results file with sample data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_evaluation_data, f)
            return f.name

    def test_load_results(
        self, temp_results_file: str, sample_evaluation_data: list[dict]
    ) -> None:
        """Test loading results from JSON file."""
        analyzer = EvaluationAnalyzer()
        results = analyzer.load_results(temp_results_file)

        assert len(results) == len(sample_evaluation_data)
        assert results[0]["model"] == "openai:gpt-4o"
        assert results[0]["rating"] == 8

        # Clean up
        Path(temp_results_file).unlink()

    def test_load_results_file_not_found(self) -> None:
        """Test error handling when results file doesn't exist."""
        analyzer = EvaluationAnalyzer()

        with pytest.raises(FileNotFoundError, match="Results file not found"):
            analyzer.load_results("nonexistent_file.json")

    def test_analyze_evaluation_results(self, temp_results_file: str) -> None:
        """Test comprehensive evaluation analysis."""
        analyzer = EvaluationAnalyzer()
        analysis = analyzer.analyze_evaluation_results(temp_results_file)

        # Check structure
        assert "model_analysis" in analysis
        assert "prompt_analysis" in analysis
        assert "total_evaluations" in analysis
        assert "models_tested" in analysis
        assert "prompts_tested" in analysis

        # Check model analysis
        assert "openai:gpt-4o" in analysis["model_analysis"]
        assert "anthropic:claude-3-haiku" in analysis["model_analysis"]

        model_stats = analysis["model_analysis"]["openai:gpt-4o"]
        assert "avg_rating" in model_stats
        assert "success_rate" in model_stats
        assert "total_responses" in model_stats
        assert model_stats["avg_rating"] == 8.5  # (8 + 9) / 2

        # Check prompt analysis
        assert "test-1" in analysis["prompt_analysis"]
        assert "test-2" in analysis["prompt_analysis"]

        # Clean up
        Path(temp_results_file).unlink()

    def test_calculate_model_stats(self) -> None:
        """Test individual model statistics calculation."""
        analyzer = EvaluationAnalyzer()

        # Mock stats data
        stats = {
            "ratings": [7, 8, 9, 8],
            "latencies": [1.0, 1.5, 2.0, 1.2],
            "success_count": 4,
            "total_count": 4,
            "prompts": {"test-1", "test-2"},
        }

        result = analyzer._calculate_model_stats(stats)

        assert result["avg_rating"] == 8.0
        assert result["min_rating"] == 7
        assert result["max_rating"] == 9
        assert result["success_rate"] == 100.0
        assert result["total_responses"] == 4
        assert result["total_prompts"] == 2
        assert "rating_distribution" in result
        assert "rating_percentiles" in result

    def test_calculate_model_stats_empty_ratings(self) -> None:
        """Test model stats calculation with no ratings."""
        analyzer = EvaluationAnalyzer()

        stats = {
            "ratings": [],
            "latencies": [],
            "success_count": 0,
            "total_count": 1,
            "prompts": set(),
        }

        result = analyzer._calculate_model_stats(stats)

        assert result["avg_rating"] == 0
        assert result["success_rate"] == 0.0
        assert result["avg_latency"] == 0

    def test_calculate_rating_distribution(self) -> None:
        """Test rating distribution calculation."""
        analyzer = EvaluationAnalyzer()

        ratings = [1, 2, 2, 3, 3, 3, 4, 5, 7, 8, 9, 10]
        distribution = analyzer._calculate_rating_distribution(ratings)

        assert distribution["10"] == 1  # One rating of 10
        assert distribution["9"] == 1  # One rating of 9
        assert distribution["8"] == 1  # One rating of 8
        assert distribution["7"] == 1  # One rating of 7
        assert distribution["6"] == 0  # No rating of 6
        assert distribution["<6"] == 8  # Eight ratings below 6 (1,2,2,3,3,3,4,5)

    def test_calculate_percentiles(self) -> None:
        """Test percentile calculation."""
        analyzer = EvaluationAnalyzer()

        ratings = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        percentiles = analyzer._calculate_percentiles(ratings)

        assert percentiles["25th"] == 3  # 25th percentile (3rd item in sorted list)
        assert percentiles["75th"] == 7  # 75th percentile (7th item in sorted list)
        assert percentiles["90th"] == 9  # 90th percentile (9th item in sorted list)

    @patch("builtins.print")
    def test_print_analysis_summary(self, mock_print: MagicMock, temp_results_file: str) -> None:
        """Test printing analysis summary."""
        analyzer = EvaluationAnalyzer()
        analysis = analyzer.analyze_evaluation_results(temp_results_file)

        analyzer.print_analysis_summary(analysis)

        # Verify print was called (we don't check exact output for brevity)
        assert mock_print.called

        # Clean up
        Path(temp_results_file).unlink()

    @patch("hellmholtz.evaluation_analysis.EvaluationAnalyzer._generate_html_header")
    @patch("hellmholtz.evaluation_analysis.EvaluationAnalyzer._generate_html_stats_section")
    def test_create_enhanced_html_report(
        self, mock_stats_section: MagicMock, mock_header: MagicMock, temp_results_file: str
    ) -> None:
        """Test HTML report creation."""
        analyzer = EvaluationAnalyzer()

        mock_header.return_value = "<html><head></head><body>"
        mock_stats_section.return_value = "<div>Stats</div>"

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_file = f.name

        try:
            analysis = analyzer.analyze_evaluation_results(temp_results_file)
            analyzer.create_enhanced_html_report(analysis, output_file)

            # Check file was created
            assert Path(output_file).exists()

            # Check content
            with open(output_file) as f:
                content = f.read()
                assert "<html>" in content
                assert "<div>Stats</div>" in content

        finally:
            Path(output_file).unlink()
            Path(temp_results_file).unlink()
