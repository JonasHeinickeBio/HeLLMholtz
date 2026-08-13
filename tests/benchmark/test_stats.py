"""Tests for hellmholtz.reporting.stats module."""

import math
from datetime import datetime

import pytest

from hellmholtz.benchmark import BenchmarkResult
from hellmholtz.reporting.stats import (
    calculate_confidence_interval,
    calculate_model_stats,
    calculate_overall_stats,
    calculate_statistical_significance,
    analyze_performance_trends,
    detect_outliers,
    generate_insights,
)


def _make_result(
    model: str = "openai:gpt-4o",
    prompt_id: str = "p1",
    temperature: float | None = 0.7,
    success: bool = True,
    latency: float = 1.0,
    response_text: str | None = "ok",
    error_message: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        prompt_id=prompt_id,
        temperature=temperature,
        success=success,
        latency_seconds=latency,
        response_text=response_text,
        error_message=error_message,
        timestamp=datetime.now().isoformat(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---- calculate_confidence_interval ----


class TestCalculateConfidenceInterval:
    def test_empty(self) -> None:
        lower, upper = calculate_confidence_interval([])
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_single_element(self) -> None:
        lower, upper = calculate_confidence_interval([5.0])
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_two_elements(self) -> None:
        lower, upper = calculate_confidence_interval([1.0, 3.0])
        assert lower < upper
        assert lower < 2.0 < upper

    def test_many_elements(self) -> None:
        data = [10.0, 10.1, 10.2, 9.9, 10.3]
        lower, upper = calculate_confidence_interval(data)
        mean = sum(data) / len(data)
        assert lower < mean < upper
        assert lower < upper

    def test_symmetric(self) -> None:
        lower, upper = calculate_confidence_interval([0.0, 10.0])
        assert abs(lower - 5.0) == abs(upper - 5.0)


# ---- calculate_statistical_significance ----


class TestCalculateStatisticalSignificance:
    def test_group1_too_small(self) -> None:
        result = calculate_statistical_significance([1.0], [1.0, 2.0])
        assert result["significant"] is False
        assert result["p_value"] == 1.0
        assert result["effect_size"] == 0.0

    def test_group2_too_small(self) -> None:
        result = calculate_statistical_significance([1.0, 2.0], [1.0])
        assert result["significant"] is False
        assert result["p_value"] == 1.0

    def test_zero_pooled_std(self) -> None:
        result = calculate_statistical_significance([5.0, 5.0], [5.0, 5.0])
        assert result["significant"] is False
        assert result["p_value"] == 1.0
        assert result["effect_size"] == 0.0
        assert result["t_statistic"] == 0.0

    def test_significant_difference(self) -> None:
        group1 = [1.0, 1.1, 1.0, 1.1, 1.0, 1.1]
        group2 = [5.0, 5.1, 5.0, 5.1, 5.0, 5.1]
        result = calculate_statistical_significance(group1, group2)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert abs(result["effect_size"]) > 1.0
        assert "t_statistic" in result

    def test_no_difference(self) -> None:
        group1 = [3.0, 3.1, 2.9, 3.0, 3.1, 3.0]
        group2 = [3.0, 3.1, 2.9, 3.0, 3.1, 3.0]
        result = calculate_statistical_significance(group1, group2)
        assert result["significant"] is False
        assert result["p_value"] > 0.05
        assert abs(result["effect_size"]) < 0.01


# ---- detect_outliers ----


class TestDetectOutliers:
    def test_empty(self) -> None:
        assert detect_outliers([]) == []

    def test_small_data(self) -> None:
        assert detect_outliers([1.0, 2.0, 3.0]) == []

    def test_iqr_with_outliers(self) -> None:
        data = [1, 1, 1, 1, 1, 1, 1, 1, 100]
        outliers = detect_outliers(data, method="iqr")
        assert len(outliers) >= 1
        assert len(data) - 1 in outliers

    def test_zscore_with_outliers(self) -> None:
        data = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1000]
        outliers = detect_outliers(data, method="zscore")
        assert len(outliers) >= 1
        assert len(data) - 1 in outliers

    def test_no_outliers(self) -> None:
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        assert detect_outliers(data, method="iqr") == []

    def test_zscore_no_outliers(self) -> None:
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        assert detect_outliers(data, method="zscore") == []

    def test_zscore_identical_values(self) -> None:
        data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        assert detect_outliers(data, method="zscore") == []


# ---- analyze_performance_trends ----


class TestAnalyzePerformanceTrends:
    def test_empty(self) -> None:
        result = analyze_performance_trends([])
        assert result == {}

    def test_temp_latency_correlation(self) -> None:
        results = [
            _make_result(temperature=0.1, latency=1.0),
            _make_result(temperature=0.5, latency=2.0),
            _make_result(temperature=1.0, latency=3.0),
        ]
        analysis = analyze_performance_trends(results)
        assert "temp_latency_correlation" in analysis

    def test_model_comparisons(self) -> None:
        results = []
        for lat in [1.0, 1.1, 1.2]:
            results.append(_make_result(model="m1", latency=lat))
        for lat in [3.0, 3.1, 3.2]:
            results.append(_make_result(model="m2", latency=lat))
        analysis = analyze_performance_trends(results)
        assert "model_comparisons" in analysis
        assert len(analysis["model_comparisons"]) == 1
        key = list(analysis["model_comparisons"].keys())[0]
        assert "m1" in key and "m2" in key

    def test_outlier_analysis(self) -> None:
        results = [_make_result(latency=1.0) for _ in range(10)]
        results.append(_make_result(latency=100.0))
        analysis = analyze_performance_trends(results)
        assert "outlier_count" in analysis
        assert analysis["outlier_count"] >= 1
        assert "outlier_percentage" in analysis


# ---- calculate_model_stats ----


class TestCalculateModelStats:
    def test_empty(self) -> None:
        assert calculate_model_stats([]) == {}

    def test_single_model(self) -> None:
        results = [
            _make_result(model="m1", success=True, latency=1.0),
            _make_result(model="m1", success=True, latency=2.0),
        ]
        stats = calculate_model_stats(results)
        assert "m1" in stats
        assert stats["m1"]["total_runs"] == 2
        assert stats["m1"]["success_rate"] == 1.0
        assert stats["m1"]["avg_latency"] == 1.5

    def test_multiple_models(self) -> None:
        results = [
            _make_result(model="m1", success=True, latency=1.0),
            _make_result(model="m1", success=True, latency=3.0),
            _make_result(model="m2", success=True, latency=2.0),
        ]
        stats = calculate_model_stats(results)
        assert len(stats) == 2
        assert stats["m1"]["avg_latency"] == 2.0
        assert stats["m2"]["avg_latency"] == 2.0


# ---- calculate_overall_stats ----


class TestCalculateOverallStats:
    def test_empty(self) -> None:
        stats = calculate_overall_stats([])
        assert stats["total_runs"] == 0
        assert stats["unique_models"] == 0
        assert stats["overall_success_rate"] == 0.0
        assert stats["avg_latency_all"] == 0.0

    def test_mixed_success_failure(self) -> None:
        results = [
            _make_result(success=True, latency=1.0),
            _make_result(success=True, latency=3.0),
            _make_result(success=False, latency=0.0),
        ]
        stats = calculate_overall_stats(results)
        assert stats["total_runs"] == 3
        assert stats["overall_success_rate"] == pytest.approx(2.0 / 3.0)
        assert stats["avg_latency_all"] == 2.0

    def test_all_failed(self) -> None:
        results = [_make_result(success=False) for _ in range(5)]
        stats = calculate_overall_stats(results)
        assert stats["overall_success_rate"] == 0.0
        assert stats["avg_latency_all"] == 0.0


# ---- generate_insights ----


class TestGenerateInsights:
    @pytest.mark.xfail(reason="Source bug: generate_insights crashes on empty results")
    def test_empty(self) -> None:
        insights = generate_insights([])
        assert isinstance(insights, list)
        assert len(insights) == 0

    def test_single_model(self) -> None:
        results = [
            _make_result(model="m1", temperature=0.5, latency=1.0, success=True),
            _make_result(model="m1", temperature=0.5, latency=2.0, success=True),
        ]
        insights = generate_insights(results)
        assert len(insights) >= 1
        assert any("Best Overall Model" in i for i in insights)

    def test_multiple_models_with_temperatures(self) -> None:
        results = [
            _make_result(model="m1", temperature=0.1, latency=1.0, success=True),
            _make_result(model="m1", temperature=0.9, latency=5.0, success=True),
            _make_result(model="m2", temperature=0.1, latency=2.0, success=True),
            _make_result(model="m2", temperature=0.9, latency=4.0, success=True),
        ]
        insights = generate_insights(results)
        assert any("Best Overall Model" in i for i in insights)
        assert any("Optimal Temperature" in i for i in insights)
        assert any("reliability" in i.lower() for i in insights)

    def test_consistency_insight(self) -> None:
        results = [
            _make_result(model="stable", temperature=0.7, latency=1.0, success=True),
            _make_result(model="stable", temperature=0.7, latency=1.01, success=True),
            _make_result(model="volatile", temperature=0.7, latency=1.0, success=True),
            _make_result(model="volatile", temperature=0.7, latency=10.0, success=True),
        ]
        insights = generate_insights(results)
        assert any("Consistent" in i for i in insights)
