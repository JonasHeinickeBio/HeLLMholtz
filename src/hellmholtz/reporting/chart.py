"""
Chart generation utilities for benchmark reporting.
"""

import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


def load_results(results_file: str) -> list[dict[str, Any]]:
    """Load benchmark results from JSON file."""
    with open(results_file) as f:
        data = json.load(f)
        return data if isinstance(data, list) else [data]


def calculate_stats(data: list[float]) -> dict[str, float]:
    """Calculate comprehensive statistics for a dataset."""
    if not data:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "count": 0,
        }

    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array, ddof=1) if len(data_array) > 1 else 0

    # Confidence interval (95%)
    if len(data_array) > 1:
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(data_array) - 1, loc=mean, scale=stats.sem(data_array)
        )
    else:
        ci_lower, ci_upper = mean, mean

    return {
        "mean": mean,
        "std": std,
        "min": np.min(data_array),
        "max": np.max(data_array),
        "median": np.median(data_array),
        "q25": np.percentile(data_array, 25),
        "q75": np.percentile(data_array, 75),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "count": len(data_array),
    }


def generate_performance_chart(results_file: str, output_path: str) -> None:
    """Generate comprehensive performance charts with seaborn."""
    results = load_results(results_file)

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Group results by model
    model_stats: dict[str, dict[str, Any]] = {}
    for result in results:
        model = result["model"]
        if model not in model_stats:
            model_stats[model] = {
                "latencies": [],
                "successes": [],
                "input_tokens": [],
                "output_tokens": [],
                "total": 0,
            }

        model_stats[model]["latencies"].append(result["latency_seconds"])
        model_stats[model]["successes"].append(result["success"])
        model_stats[model]["input_tokens"].append(result.get("input_tokens") or 0)
        model_stats[model]["output_tokens"].append(result.get("output_tokens") or 0)
        model_stats[model]["total"] += 1

    # Prepare data for plotting
    models: list[str] = []
    success_rates: list[float] = []
    success_stds: list[float] = []
    avg_latencies: list[float] = []
    latency_stds: list[float] = []
    latency_ci_lower: list[float] = []
    latency_ci_upper: list[float] = []
    throughput_data: list[float] = []  # tokens per second
    total_requests: list[int] = []

    for model, model_data in model_stats.items():
        models.append(model.split(":")[-1])  # Remove provider prefix

        # Success rate stats
        success_rate = (
            sum(model_data["successes"]) / len(model_data["successes"]) * 100
            if model_data["successes"]
            else 0
        )
        success_rates.append(success_rate)
        success_stds.append(
            np.std(model_data["successes"]) * 100 if len(model_data["successes"]) > 1 else 0
        )

        # Latency stats
        latency_stats = calculate_stats(model_data["latencies"])
        avg_latencies.append(latency_stats["mean"])
        latency_stds.append(latency_stats["std"])
        latency_ci_lower.append(latency_stats["ci_lower"])
        latency_ci_upper.append(latency_stats["ci_upper"])

        # Throughput calculation (total tokens per second)
        total_tokens = sum(model_data["input_tokens"]) + sum(model_data["output_tokens"])
        total_time = sum(model_data["latencies"])
        throughput = total_tokens / total_time if total_time > 0 else 0
        throughput_data.append(throughput)

        total_requests.append(model_data["total"])

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Success Rate Chart
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(
        models,
        success_rates,
        yerr=success_stds,
        capsize=5,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Model Success Rates (%)", fontsize=14, fontweight="bold", pad=20)
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, success_rates, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Latency Chart with Confidence Intervals
    ax2 = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(models))
    ax2.bar(
        x_pos,
        avg_latencies,
        yerr=[
            np.array(avg_latencies) - np.array(latency_ci_lower),
            np.array(latency_ci_upper) - np.array(avg_latencies),
        ],
        capsize=5,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Response Latency (seconds)", fontsize=14, fontweight="bold", pad=20)
    ax2.set_ylabel("Latency (seconds)", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, latency in enumerate(avg_latencies):
        ax2.text(
            i,
            latency + 0.02,
            f"{latency:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Throughput Chart
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.bar(models, throughput_data, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax3.set_title("Token Throughput (tokens/second)", fontsize=14, fontweight="bold", pad=20)
    ax3.set_ylabel("Tokens/second", fontsize=12)
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, throughput in zip(bars3, throughput_data, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(throughput_data) * 0.02,
            f"{throughput:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Request Count Chart
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.bar(
        models, total_requests, alpha=0.8, edgecolor="black", linewidth=0.5, color="lightgreen"
    )
    ax4.set_title("Total Requests per Model", fontsize=14, fontweight="bold", pad=20)
    ax4.set_ylabel("Request Count", fontsize=12)
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, count in zip(bars4, total_requests, strict=False):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(total_requests) * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Summary Statistics Table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    # Create summary table data
    table_data = []
    for i, model in enumerate(models):
        table_data.append(
            [
                model,
                f"{success_rates[i]:.1f}%",
                f"{avg_latencies[i]:.2f}s",
                f"{latency_stds[i]:.2f}s",
                f"{throughput_data[i]:.1f}",
                str(total_requests[i]),
            ]
        )

    table = ax5.table(
        cellText=table_data,
        colLabels=[
            "Model",
            "Success Rate",
            "Avg Latency",
            "Latency Std",
            "Throughput",
            "Requests",
        ],
        loc="center",
        cellLoc="center",
        colColours=["lightblue"] * 6,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax5.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=20)

    # Overall title
    fig.suptitle("LLM Benchmark Performance Report", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Enhanced performance chart saved to {output_path}")
    print(f"Generated charts for {len(models)} models with comprehensive statistics")


def main() -> None:
    """Main entry point for command line usage."""
    if len(sys.argv) != 3:
        print("Usage: python -m hellmholtz.reporting.chart <results_file> <output_image>")
        sys.exit(1)

    results_file = sys.argv[1]
    output_image = sys.argv[2]

    generate_performance_chart(results_file, output_image)


if __name__ == "__main__":
    main()
