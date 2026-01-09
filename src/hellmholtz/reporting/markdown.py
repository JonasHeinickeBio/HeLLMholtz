"""
Markdown report generation functions.
"""

from hellmholtz.benchmark import BenchmarkResult


def generate_markdown_report(results: list[BenchmarkResult]) -> str:
    """Generate a Markdown summary of benchmark results."""
    if not results:
        return "No results to summarize."

    models = sorted(list(set(r.model for r in results)))

    summary = ["# Benchmark Summary\n"]

    summary.append(f"**Total Runs**: {len(results)}")
    summary.append(f"**Models**: {', '.join(models)}\n")

    summary.append("## Performance by Model\n")
    summary.append("| Model | Success Rate | Avg Latency (s) |")
    summary.append("|-------|--------------|-----------------|")

    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        successes = len([r for r in model_results if r.success])
        avg_latency = sum(r.latency_seconds for r in model_results) / total if total > 0 else 0

        success_rate = (successes / total) * 100 if total > 0 else 0

        summary.append(f"| {model} | {success_rate:.1f}% | {avg_latency:.4f} |")

    return "\n".join(summary)


def summarize_results(results: list[BenchmarkResult]) -> str:
    """Summarize benchmark results (alias for markdown report)."""
    return generate_markdown_report(results)
