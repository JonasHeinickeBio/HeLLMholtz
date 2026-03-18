"""
HTML report generation functions.
"""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template

from hellmholtz.benchmark import BenchmarkResult
from hellmholtz.benchmark.prompts import PROMPTS


def _load_template(template_name: str) -> Template:
    """Load an HTML template from the templates directory."""
    template_path = Path(__file__).parent / "templates" / f"{template_name}.html"
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=True)
    return env.get_template(f"{template_name}.html")


def _prepare_simple_report_data(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Prepare data for the simple HTML report."""
    if not results:
        return {}

    models = sorted(list(set(r.model for r in results)))
    temperatures = sorted(list(set(r.temperature for r in results if r.temperature is not None)))

    total_runs = len(results)
    successful_runs = len([r for r in results if r.success])
    success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

    model_stats = []
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        successes = len([r for r in model_results if r.success])
        latencies = [r.latency_seconds for r in model_results if r.success]

        model_stats.append(
            {
                "model": model,
                "total": total,
                "successes": successes,
                "success_rate": (successes / total * 100) if total > 0 else 0,
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
            }
        )

    return {
        "total_runs": total_runs,
        "success_rate": f"{success_rate:.1f}",
        "models_count": len(models),
        "temperatures_count": len(temperatures) if temperatures else 0,
        "timestamp": results[0].timestamp if results else "Unknown",
        "model_labels": json.dumps([stat["model"] for stat in model_stats]),
        "success_rates": json.dumps([stat["success_rate"] for stat in model_stats]),
        "avg_latencies": json.dumps([stat["avg_latency"] for stat in model_stats]),
    }


def generate_html_report_simple(results: list[BenchmarkResult]) -> str:
    """Generate a simple HTML report with basic statistics and charts."""
    if not results:
        return "<p>No results to summarize.</p>"

    template = _load_template("simple")
    data = _prepare_simple_report_data(results)
    return str(template.render(**data))


def generate_html_report_detailed(results: list[BenchmarkResult]) -> str:  # noqa: C901
    """Generate a detailed HTML report with individual test results and statistics."""
    if not results:
        return "<p>No results to summarize.</p>"

    # Extract metadata
    models = sorted(list(set(r.model for r in results)))
    temperatures = sorted(list(set(r.temperature for r in results if r.temperature is not None)))

    # Calculate statistics
    total_runs = len(results)
    successful_runs = len([r for r in results if r.success])
    success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

    # Model performance data
    model_stats = []
    for model in models:
        model_results = [r for r in results if r.model == model]
        total = len(model_results)
        successes = len([r for r in model_results if r.success])
        latencies = [r.latency_seconds for r in model_results if r.success]

        model_stats.append(
            {
                "model": model,
                "total": total,
                "successes": successes,
                "success_rate": (successes / total * 100) if total > 0 else 0,
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
                "std_latency": (
                    sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies)
                    / len(latencies)
                )
                ** 0.5
                if latencies
                else 0,
            }
        )

    # Temperature analysis
    temp_stats = []
    if temperatures:
        for temp in temperatures:
            temp_results = [r for r in results if r.temperature == temp and r.success]
            if temp_results:
                latencies = [r.latency_seconds for r in temp_results]
                temp_stats.append(
                    {
                        "temperature": temp,
                        "count": len(temp_results),
                        "avg_latency": sum(latencies) / len(latencies),
                        "success_rate": len(
                            [r for r in results if r.temperature == temp and r.success]
                        )
                        / len([r for r in results if r.temperature == temp])
                        * 100,
                    }
                )

    # Group results by model and prompt for detailed view
    detailed_results: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        key = f"{result.model}_{result.prompt_id}"
        if key not in detailed_results:
            detailed_results[key] = []
        detailed_results[key].append(result)

    # Build HTML content for template
    model_stats_table = ""
    for stat in model_stats:
        model_stats_table += f"""
                        <tr>
                            <td><strong>{stat["model"]}</strong></td>
                            <td>{stat["total"]}</td>
                            <td class="success-rate">{stat["success_rate"]:.1f}%</td>
                            <td class="latency">{stat["avg_latency"]:.3f}s</td>
                            <td>{stat["min_latency"]:.3f}s</td>
                            <td>{stat["max_latency"]:.3f}s</td>
                            <td>{stat["std_latency"]:.3f}s</td>
                        </tr>
        """

    temperature_section = ""
    temperature_chart_script = ""
    if temperatures:
        temperature_section = """
            <div class="section">
                <h2>🌡️ Temperature Analysis</h2>
                <div class="chart-container">
                    <canvas id="temperatureChart"></canvas>
                </div>

                <div class="temperature-analysis">
        """

        for temp_stat in temp_stats:
            temperature_section += f"""
                    <div class="temp-card">
                        <h3>Temperature: {temp_stat["temperature"]}</h3>
                        <div class="metric">
                            <span class="metric-label">Tests:</span>
                            <span class="metric-value">{temp_stat["count"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Success Rate:</span>
                            <span class="metric-value success-rate">
                                {temp_stat["success_rate"]:.1f}%
                            </span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Avg Latency:</span>
                            <span class="metric-value latency">
                                {temp_stat["avg_latency"]:.3f}s
                            </span>
                        </div>
                    </div>
            """

        temperature_section += """
                </div>
            </div>
        """

        temperature_chart_script = f"""
            // Temperature Chart
            const tempCtx = document.getElementById('temperatureChart').getContext('2d');
            new Chart(tempCtx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps([stat["temperature"] for stat in temp_stats])},
                    datasets: [{{
                        label: 'Success Rate (%)',
                        data: {json.dumps([stat["success_rate"] for stat in temp_stats])},
                        borderColor: 'rgba(40, 167, 69, 1)',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4
                    }}, {{
                        label: 'Avg Latency (s)',
                        data: {json.dumps([stat["avg_latency"] for stat in temp_stats])},
                        borderColor: 'rgba(0, 123, 255, 1)',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Performance vs Temperature'
                        }}
                    }}
                }}
            }});
        """

    detailed_results_html = ""
    for key, results_list in detailed_results.items():
        model_name, prompt_id = key.split("_", 1)
        # Get prompt details
        prompt_info = next((p for p in PROMPTS if p.id == prompt_id), None)
        prompt_text = prompt_info.messages[0].content if prompt_info else "Unknown prompt"
        category = prompt_info.category if prompt_info else "unknown"

        detailed_results_html += f"""
                <div class="result-item">
                    <div class="result-header">
                        <div class="result-model">{model_name} - {prompt_id}</div>
                        <div class="result-meta">
                            Category: {category} | {len(results_list)} runs
                        </div>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>Prompt:</strong> {prompt_text[:200]}
                        {"..." if len(prompt_text) > 200 else ""}
                    </div>
                    <div class="result-metrics">
        """

        for i, result in enumerate(results_list):
            status_class = "success" if result.success else "error"
            status_text = "✓" if result.success else "✗"
            temp_info = f" T={result.temperature}" if result.temperature is not None else ""
            tokens_info = ""
            if result.input_tokens is not None and result.output_tokens is not None:
                tokens_info = f" | {result.input_tokens + result.output_tokens} tokens"

            detailed_results_html += f"""
                        <div class="metric-item">
                            <div class="{status_class}">{status_text} Run {i + 1}{temp_info}</div>
                            <div class="latency">{result.latency_seconds:.3f}s{tokens_info}</div>
                        </div>
            """

        detailed_results_html += """
                    </div>
                </div>
        """

    # Load template and substitute
    template = _load_template("detailed")
    data: dict[str, str] = {
        "total_runs": str(total_runs),
        "success_rate": f"{success_rate:.1f}",
        "models_count": str(len(models)),
        "temperatures_count": str(len(temperatures) if temperatures else 0),
        "timestamp": results[0].timestamp if results else "Unknown",
        "model_labels": str(json.dumps([stat["model"] for stat in model_stats])),
        "success_rates": str(json.dumps([stat["success_rate"] for stat in model_stats])),
        "avg_latencies": str(json.dumps([stat["avg_latency"] for stat in model_stats])),
        "model_stats_table": model_stats_table,
        "temperature_section": temperature_section,
        "detailed_results": detailed_results_html,
        "temperature_chart_script": temperature_chart_script,
    }
    return str(template.render(**data))


def generate_html_report(results: list[BenchmarkResult]) -> str:
    """Generate a comprehensive HTML report (alias for detailed report)."""
    return generate_html_report_detailed(results)


def generate_html_report_full(results: list[BenchmarkResult]) -> str:
    """Generate a comprehensive HTML report with advanced features."""
    # For now, return a placeholder - this would be the most complex template
    return "<p>Full report not yet implemented in modular format.</p>"
