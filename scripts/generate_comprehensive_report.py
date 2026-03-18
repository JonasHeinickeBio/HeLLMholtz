#!/usr/bin/env python3
"""
Comprehensive Weekly Benchmark Report Generator

This script generates detailed reports combining benchmark results with model status data.
Used by the weekly benchmark GitHub Actions workflow.
"""

from datetime import datetime
import json
from pathlib import Path
import sys

import yaml


def load_benchmark_data(results_file: str) -> dict:
    """Load benchmark results from JSON file."""
    try:
        with open(results_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading benchmark data: {e}", file=sys.stderr)
        return {}


def load_model_status() -> dict:
    """Load model status from YAML file."""
    try:
        with open("models_status.yaml") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading model status: {e}", file=sys.stderr)
        return {}


def generate_comprehensive_markdown_report(
    benchmark_data: dict, model_status: dict, results_file: str
) -> str:
    """Generate comprehensive Markdown report."""
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Extract data
    models = model_status.get("models", {})
    total_models = len(models)
    available_models = sum(1 for m in models.values() if m.get("available", False))
    tested_models = sum(1 for m in models.values() if m.get("latency") is not None)

    # Build report
    lines = [
        f"# Weekly Benchmark Report - {report_date}",
        "",
        "## 📊 Executive Summary",
        "",
        f"**Report Generated:** {report_date}",
        f"**Benchmark Results:** {Path(results_file).name}",
        f'**Models Tested:** {len(benchmark_data.get("results", {}))}',
    ]

    # Calculate total prompts
    total_prompts = 0
    if benchmark_data.get("results"):
        first_model = next(iter(benchmark_data["results"].keys()), None)
        if first_model:
            total_prompts = len(benchmark_data["results"][first_model].get("prompt_results", []))
    lines.append(f"**Total Prompts:** {total_prompts}")
    lines.append("")

    # Model status section
    lines.extend(
        [
            "## 🔍 Model Availability Status",
            "",
            "### Status Summary",
            f"- **Total Models:** {total_models}",
            f"- **Available:** {available_models} ({available_models/total_models*100:.1f}%)"
            if total_models > 0
            else "- **Available:** 0",
            f"- **Tested Accessible:** {tested_models}",
            f'- **Last Updated:** {model_status.get("# Last updated", "Unknown")}',
            "",
            "### Model Categories",
            "",
        ]
    )

    # Group models by category
    categories = {}
    for name, info in models.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    for cat, model_list in categories.items():
        lines.append(f'#### {cat.replace("_", " ").title()} Models')
        for name, info in sorted(
            model_list, key=lambda x: (not x[1].get("available", False), x[0])
        ):
            status = "✅" if info.get("available", False) else "❌"
            latency = f", {info.get('latency', 'N/A')}s" if info.get("latency") else ""
            lines.append(f"- {status} {name}{latency}")
        lines.append("")

    # Benchmark results
    lines.extend(
        [
            "## 📈 Benchmark Results",
            "",
            "### Performance Summary",
            "",
            "| Model | Success Rate | Avg Latency | Total Requests |",
            "|-------|-------------|-------------|----------------|",
        ]
    )

    if benchmark_data.get("results"):
        for model_name, model_data in benchmark_data["results"].items():
            success_rate = model_data.get("success_rate", 0)
            avg_latency = model_data.get("avg_latency_seconds", 0)
            total_requests = len(model_data.get("prompt_results", []))
            lines.append(
                f"| {model_name} | {success_rate:.1%} | {avg_latency:.2f}s | {total_requests} |"
            )

    lines.append("")

    # Recommendations
    lines.extend(["## 📋 Recommendations", ""])

    # Generate recommendations
    if available_models < total_models * 0.5:
        lines.append(
            "- ⚠️  **Low Availability:** Less than 50% of models are available. Consider reviewing API access or model configurations."
        )

    if tested_models < available_models:
        lines.append(
            f"- 🔍 **Testing Gap:** {available_models - tested_models} available models were not tested for accessibility."
        )

    if benchmark_data.get("results"):
        successful_models = sum(
            1 for m in benchmark_data["results"].values() if m.get("success_rate", 0) > 0.8
        )
        if successful_models < len(benchmark_data["results"]) * 0.7:
            lines.append(
                "- 📉 **Performance Issues:** Some models show low success rates. Review model configurations or API issues."
            )

    # Links
    lines.extend(
        [
            "",
            "## 🔗 Links",
            f"- [Full Benchmark Results]({results_file})",
            "- [Model Status YAML](models_status.yaml)",
            "- [HTML Report](weekly_benchmark_report.html)",
            "- [Performance Chart](weekly_benchmark_chart.png)",
            "",
            "---",
            "*This report is automatically generated weekly by GitHub Actions.*",
        ]
    )

    return "\n".join(lines)


def generate_comprehensive_html_report(
    benchmark_data: dict, model_status: dict, results_file: str
) -> str:
    """Generate comprehensive HTML report."""
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Extract data
    models = model_status.get("models", {})
    total_models = len(models)
    available_models = sum(1 for m in models.values() if m.get("available", False))
    tested_models = sum(1 for m in models.values() if m.get("latency") is not None)

    # Group models by category
    categories = {}
    for name, info in models.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Weekly Benchmark Report - {report_date}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .status-card {{ background: white; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; }}
        .available {{ color: #28a745; }}
        .unavailable {{ color: #dc3545; }}
        .metric {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e9ecef; color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Weekly Benchmark Report</h1>
        <p><strong>Generated:</strong> {report_date}</p>
        <p><strong>Models Tested:</strong> {len(benchmark_data.get('results', {}))}</p>
    </div>

    <div class="summary">
        <h2>🔍 Model Availability Status</h2>
        <div class="status-grid">
            <div class="status-card">
                <h3>Total Models</h3>
                <div class="metric">{total_models}</div>
            </div>
            <div class="status-card">
                <h3>Available</h3>
                <div class="metric available">{available_models}</div>
                <small>{available_models/total_models*100:.1f}%</small>
            </div>
            <div class="status-card">
                <h3>Tested Accessible</h3>
                <div class="metric">{tested_models}</div>
            </div>
        </div>
    </div>
"""

    # Add model categories
    html += '<div class="summary"><h2>📋 Model Categories</h2>'
    for cat, model_list in categories.items():
        html += f'<h3>{cat.replace("_", " ").title()} Models</h3><ul>'
        for name, info in sorted(
            model_list, key=lambda x: (not x[1].get("available", False), x[0])
        ):
            css_class = "available" if info.get("available", False) else "unavailable"
            status_icon = "✅" if info.get("available", False) else "❌"
            latency = f" ({info.get('latency', 'N/A')}s)" if info.get("latency") else ""
            html += f'<li class="{css_class}">{status_icon} {name}{latency}</li>'
        html += "</ul>"
    html += "</div>"

    # Add benchmark results
    if benchmark_data.get("results"):
        html += '<div class="summary"><h2>📈 Benchmark Results</h2>'
        html += '<table class="table"><thead><tr><th>Model</th><th>Success Rate</th><th>Avg Latency</th><th>Total Requests</th></tr></thead><tbody>'

        for model_name, model_data in benchmark_data["results"].items():
            success_rate = model_data.get("success_rate", 0)
            avg_latency = model_data.get("avg_latency_seconds", 0)
            total_requests = len(model_data.get("prompt_results", []))
            html += f"<tr><td>{model_name}</td><td>{success_rate:.1%}</td><td>{avg_latency:.2f}s</td><td>{total_requests}</td></tr>"

        html += "</tbody></table></div>"

    # Add recommendations
    html += '<div class="recommendations"><h2>💡 Recommendations</h2><ul>'

    if available_models < total_models * 0.5:
        html += "<li>⚠️ <strong>Low Availability:</strong> Less than 50% of models are available. Consider reviewing API access or model configurations.</li>"

    if tested_models < available_models:
        html += f"<li>🔍 <strong>Testing Gap:</strong> {available_models - tested_models} available models were not tested for accessibility.</li>"

    if benchmark_data.get("results"):
        successful_models = sum(
            1 for m in benchmark_data["results"].values() if m.get("success_rate", 0) > 0.8
        )
        if successful_models < len(benchmark_data["results"]) * 0.7:
            html += "<li>📉 <strong>Performance Issues:</strong> Some models show low success rates. Review model configurations or API issues.</li>"

    html += "</ul></div>"

    html += f"""
    <div class="footer">
        <p><strong>Links:</strong>
            <a href="{results_file}">Full Benchmark Results</a> |
            <a href="models_status.yaml">Model Status YAML</a> |
            <a href="weekly_benchmark_report.html">HTML Report</a> |
            <a href="weekly_benchmark_chart.png">Performance Chart</a>
        </p>
        <p><em>This report is automatically generated weekly by GitHub Actions.</em></p>
    </div>
</body>
</html>"""

    return html


def main():
    """Main function to generate comprehensive reports."""
    if len(sys.argv) != 2:
        print("Usage: python generate_comprehensive_report.py <results_file>", file=sys.stderr)
        sys.exit(1)

    results_file = sys.argv[1]

    # Load data
    benchmark_data = load_benchmark_data(results_file)
    model_status = load_model_status()

    # Generate reports
    markdown_report = generate_comprehensive_markdown_report(
        benchmark_data, model_status, results_file
    )
    html_report = generate_comprehensive_html_report(benchmark_data, model_status, results_file)

    # Write reports
    with open("reports/weekly_benchmark_comprehensive.md", "w") as f:
        f.write(markdown_report)

    with open("reports/weekly_benchmark_comprehensive.html", "w") as f:
        f.write(html_report)

    print("✅ Comprehensive reports generated successfully")


if __name__ == "__main__":
    main()
