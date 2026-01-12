"""
Statistical analysis functions for benchmark reporting.
"""

import math
import statistics
from typing import Any

from hellmholtz.benchmark import BenchmarkResult


def calculate_confidence_interval(data: list[float]) -> tuple[float, float]:
    """Calculate confidence interval for a dataset."""
    if len(data) < 2:
        return (float("nan"), float("nan"))

    mean = statistics.mean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0
    n = len(data)

    # t-distribution approximation for 95% confidence
    t_value = 1.96  # approximately 1.96 for large n
    margin = t_value * (std / math.sqrt(n))

    return (mean - margin, mean + margin)


def calculate_statistical_significance(group1: list[float], group2: list[float]) -> dict[str, Any]:
    """Calculate statistical significance between two groups using t-test approximation."""
    if len(group1) < 2 or len(group2) < 2:
        return {"significant": False, "p_value": 1.0, "effect_size": 0.0}

    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    std1, std2 = statistics.stdev(group1), statistics.stdev(group2)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return {"significant": False, "p_value": 1.0, "effect_size": 0.0, "t_statistic": 0.0}

    # t-statistic
    t_stat = (mean1 - mean2) / (pooled_std * math.sqrt(1 / n1 + 1 / n2))

    # Approximate p-value (two-tailed)
    # Using normal distribution approximation
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    # Cohen's d effect size
    effect_size = (mean1 - mean2) / pooled_std

    return {
        "significant": p_value < 0.05,
        "p_value": p_value,
        "effect_size": effect_size,
        "t_statistic": t_stat,
    }


def detect_outliers(data: list[float], method: str = "iqr") -> list[int]:
    """Detect outlier indices using IQR or z-score method."""
    if len(data) < 4:
        return []

    if method == "iqr":
        q1 = statistics.quantiles(data, n=4)[0]
        q3 = statistics.quantiles(data, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    else:  # z-score method
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        if std_val == 0:
            return []  # No outliers when all values are identical
        outliers = [i for i, x in enumerate(data) if abs((x - mean_val) / std_val) > 3]

    return outliers


def analyze_performance_trends(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Analyze performance trends and correlations."""
    analysis: dict[str, Any] = {}

    # Temperature vs Performance correlation
    temp_data = [
        (r.temperature, r.latency_seconds)
        for r in results
        if r.temperature is not None and r.success
    ]
    if temp_data:
        temps, latencies = zip(*temp_data, strict=False)
        try:
            correlation = statistics.correlation(temps, latencies)
            analysis["temp_latency_correlation"] = correlation
        except (ValueError, statistics.StatisticsError):
            analysis["temp_latency_correlation"] = 0.0

    # Model comparison significance tests
    models = list(set(r.model for r in results))
    if len(models) >= 2:
        model_comparisons = {}
        for i, model1 in enumerate(models[:-1]):
            for model2 in models[i + 1 :]:
                latencies1 = [
                    r.latency_seconds for r in results if r.model == model1 and r.success
                ]
                latencies2 = [
                    r.latency_seconds for r in results if r.model == model2 and r.success
                ]

                if len(latencies1) >= 3 and len(latencies2) >= 3:
                    sig_test = calculate_statistical_significance(latencies1, latencies2)
                    model_comparisons[f"{model1}_vs_{model2}"] = sig_test

        analysis["model_comparisons"] = model_comparisons

    # Outlier analysis
    all_latencies = [r.latency_seconds for r in results if r.success]
    if all_latencies:
        outlier_indices = detect_outliers(all_latencies)
        analysis["outlier_count"] = len(outlier_indices)
        analysis["outlier_percentage"] = len(outlier_indices) / len(all_latencies) * 100

    return analysis


def calculate_model_stats(results: list[BenchmarkResult]) -> dict[str, dict[str, Any]]:
    """Calculate statistics for each model."""
    if not results:
        return {}

    model_stats = {}
    models = set(r.model for r in results)

    for model in models:
        model_results = [r for r in results if r.model == model]
        total_runs = len(model_results)
        successful_runs = [r for r in model_results if r.success]
        success_rate = len(successful_runs) / total_runs if total_runs > 0 else 0.0

        avg_latency = (
            statistics.mean([r.latency_seconds for r in successful_runs])
            if successful_runs
            else 0.0
        )

        model_stats[model] = {
            "total_runs": total_runs,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
        }

    return model_stats


def calculate_overall_stats(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Calculate overall statistics across all results."""
    if not results:
        return {
            "total_runs": 0,
            "unique_models": 0,
            "overall_success_rate": 0.0,
            "avg_latency_all": 0.0,
        }

    total_runs = len(results)
    unique_models = len(set(r.model for r in results))
    successful_runs = [r for r in results if r.success]
    overall_success_rate = len(successful_runs) / total_runs if total_runs > 0 else 0.0

    avg_latency_all = (
        statistics.mean([r.latency_seconds for r in successful_runs]) if successful_runs else 0.0
    )

    return {
        "total_runs": total_runs,
        "unique_models": unique_models,
        "overall_success_rate": overall_success_rate,
        "avg_latency_all": avg_latency_all,
    }


def generate_insights(results: list[BenchmarkResult]) -> list[str]:  # noqa: C901
    """Generate automated insights and recommendations."""
    insights = []

    models = list(set(r.model for r in results))
    temperatures = list(set(r.temperature for r in results if r.temperature is not None))

    # Model performance insights
    model_stats = {}
    for model in models:
        model_results = [r for r in results if r.model == model]
        success_rate = len([r for r in model_results if r.success]) / len(model_results) * 100
        avg_latency = (
            statistics.mean([r.latency_seconds for r in model_results if r.success])
            if any(r.success for r in model_results)
            else float("inf")
        )
        model_stats[model] = {"success_rate": success_rate, "avg_latency": avg_latency}

    # Best performing model
    best_model = min(
        model_stats.items(), key=lambda x: (x[1]["avg_latency"], -x[1]["success_rate"])
    )
    insights.append(
        f"🏆 **Best Overall Model**: {best_model[0]} "
        f"(avg {best_model[1]['avg_latency']:.3f}s, "
        f"{best_model[1]['success_rate']:.1f}% success)"
    )

    # Temperature insights
    if temperatures:
        temp_performance = {}
        for temp in temperatures:
            temp_results = [r for r in results if r.temperature == temp and r.success]
            if temp_results:
                avg_latency = statistics.mean([r.latency_seconds for r in temp_results])
                success_rate = (
                    len(temp_results) / len([r for r in results if r.temperature == temp]) * 100
                )
                temp_performance[temp] = {"latency": avg_latency, "success": success_rate}

        if temp_performance:
            best_temp = min(temp_performance.items(), key=lambda x: x[1]["latency"])
            insights.append(
                f"🌡️ **Optimal Temperature**: {best_temp[0]} "
                f"(fastest responses with {best_temp[1]['latency']:.3f}s avg latency)"
            )

    # Reliability insights
    reliability_scores = {}
    for model in models:
        model_results = [r for r in results if r.model == model]
        success_rate = len([r for r in model_results if r.success]) / len(model_results)
        reliability_scores[model] = success_rate

    most_reliable = max(reliability_scores.items(), key=lambda x: x[1])
    if most_reliable[1] < 1.0:
        insights.append(
            f"🛡️ **Most Reliable Model**: {most_reliable[0]} "
            f"({most_reliable[1] * 100:.1f}% success rate)"
        )
    else:
        insights.append("✅ **All models show 100% reliability** in this benchmark")

    # Performance variability
    latency_variability = {}
    for model in models:
        latencies = [r.latency_seconds for r in results if r.model == model and r.success]
        if len(latencies) > 1:
            cv = statistics.stdev(latencies) / statistics.mean(
                latencies
            )  # Coefficient of variation
            latency_variability[model] = cv

    if latency_variability:
        most_consistent = min(latency_variability.items(), key=lambda x: x[1])
        insights.append(
            f"📊 **Most Consistent Model**: {most_consistent[0]} (lowest latency variability)"
        )

    return insights
