"""
Markdown report generation functions.
"""

from hellmholtz.benchmark import BenchmarkResult


def generate_markdown_report(results: list[BenchmarkResult]) -> str:
    """Generate a concise markdown summary with expanded metrics.

    Includes success rate, average latency, token usage, throughput, and optional rating.
    """
    if not results:
        return "No results to summarize."

    models = sorted(list(set(r.model for r in results)))

    summary = ["# Benchmark Summary\n"]

    summary.append(f"**Total Runs**: {len(results)}")
    summary.append(f"**Models**: {', '.join(models)}\n")

    summary.append("## Performance by Model\n")
    summary.append(
        "| Model | Success Rate | Avg Latency (s) | Avg Input Tokens |"
        " Avg Output Tokens | Tokens/sec | Avg Rating |"
    )
    summary.append(
        "|-------|--------------|-----------------|-------------------|--------------------|-----------|------------|"
    )
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        successes = len([r for r in model_results if r.success])
        avg_latency = (
            sum(r.latency_seconds or 0 for r in model_results) / total if total > 0 else 0
        )
        avg_input = sum(r.input_tokens or 0 for r in model_results) / total if total > 0 else 0
        avg_output = sum(r.output_tokens or 0 for r in model_results) / total if total > 0 else 0
        avg_tps = sum(r.tokens_per_sec or 0 for r in model_results) / total if total > 0 else 0
        rated = [r.rating for r in model_results if r.rating is not None]
        avg_rating = sum(rated) / len(rated) if rated else None
        rating_cell = f"{avg_rating:.2f}" if avg_rating is not None else "-"

        success_rate = (successes / total) * 100 if total > 0 else 0

        summary.append(
            f"| {model} | {success_rate:.1f}% | {avg_latency:.4f} | {avg_input:.1f} |"
            f" {avg_output:.1f} | {avg_tps:.2f} | {rating_cell} |"
        )

    return "\n".join(summary)


def summarize_results(results: list[BenchmarkResult]) -> str:
    """Summarize benchmark results (alias for markdown report)."""
    return generate_markdown_report(results)
