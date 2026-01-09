"""
Evaluation Analysis for HeLLMholtz Benchmark Results

This module provides comprehensive analysis of benchmark evaluation results
with LLM-as-a-Judge scoring, including statistical analysis and visualization.
"""

from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, cast

from hellmholtz.core.config import get_settings


class EvaluationAnalyzer:
    """Analyzer for benchmark evaluation results with LLM-as-a-Judge scoring."""

    def __init__(self) -> None:
        """Initialize the evaluation analyzer."""
        self.settings = get_settings()

    def load_results(self, results_file: str) -> list[dict[str, Any]]:
        """Load evaluation results from JSON file.

        Args:
            results_file: Path to the JSON results file.

        Returns:
            List of evaluation result dictionaries.

        Raises:
            FileNotFoundError: If the results file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        results_path = Path(results_file)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)
            return cast(list[dict[str, Any]], data)

    def analyze_evaluation_results(self, results_file: str) -> dict[str, Any]:
        """Analyze evaluation results and return comprehensive statistics.

        Args:
            results_file: Path to the evaluation results JSON file.

        Returns:
            Dictionary containing analysis results with model and prompt statistics.
        """
        data = self.load_results(results_file)

        # Group by model
        model_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "ratings": [],
                "latencies": [],
                "success_count": 0,
                "total_count": 0,
                "responses": [],
                "critiques": [],
                "prompts": set[str](),
            }
        )

        # Group by prompt
        prompt_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"ratings": [], "responses": [], "models": set[str]()}
        )

        for result in data:
            model = result["model"]
            prompt_id = result["prompt_id"]
            success = result["success"]

            model_stats[model]["total_count"] += 1
            cast(set[str], model_stats[model]["prompts"]).add(prompt_id)

            if success:
                model_stats[model]["success_count"] += 1
                model_stats[model]["latencies"].append(result["latency_seconds"])

                if result.get("rating") is not None:
                    rating = result["rating"]
                    model_stats[model]["ratings"].append(rating)
                    model_stats[model]["responses"].append(result.get("response_text", ""))
                    model_stats[model]["critiques"].append(result.get("critique", ""))

                    prompt_stats[prompt_id]["ratings"].append(rating)
                    prompt_stats[prompt_id]["responses"].append(result.get("response_text", ""))
                    cast(set[str], prompt_stats[prompt_id]["models"]).add(model)

        # Calculate model statistics
        model_analysis = {}
        for model, stats in model_stats.items():
            if stats["ratings"]:
                model_analysis[model] = self._calculate_model_stats(stats)

        # Calculate prompt statistics
        prompt_analysis = {}
        for prompt_id, stats in prompt_stats.items():
            if stats["ratings"]:
                prompt_analysis[prompt_id] = self._calculate_prompt_stats(stats)

        return {
            "model_analysis": model_analysis,
            "prompt_analysis": prompt_analysis,
            "total_evaluations": sum(len(stats["ratings"]) for stats in model_stats.values()),
            "models_tested": len([m for m in model_stats if model_stats[m]["ratings"]]),
            "prompts_tested": len([p for p in prompt_stats if prompt_stats[p]["ratings"]]),
            "summary": self._generate_summary(model_analysis, prompt_analysis),
        }

    def _calculate_model_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive statistics for a single model.

        Args:
            stats: Raw statistics for the model.

        Returns:
            Dictionary with calculated statistics.
        """
        ratings = stats["ratings"]
        latencies = stats["latencies"]

        return {
            "avg_rating": round(statistics.mean(ratings), 2),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            "median_rating": round(statistics.median(ratings), 2),
            "rating_std": round(statistics.stdev(ratings), 2) if len(ratings) > 1 else 0,
            "success_rate": round(stats["success_count"] / stats["total_count"] * 100, 1),
            "avg_latency": round(statistics.mean(latencies), 3) if latencies else 0,
            "min_latency": round(min(latencies), 3) if latencies else 0,
            "max_latency": round(max(latencies), 3) if latencies else 0,
            "total_responses": len(ratings),
            "total_prompts": len(cast(set[str], stats["prompts"])),
            "rating_distribution": self._calculate_rating_distribution(ratings),
            "rating_percentiles": self._calculate_percentiles(ratings),
        }

    def _calculate_prompt_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Calculate statistics for a single prompt.

        Args:
            stats: Raw statistics for the prompt.

        Returns:
            Dictionary with calculated statistics.
        """
        ratings = stats["ratings"]

        return {
            "avg_rating": round(statistics.mean(ratings), 2),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            "median_rating": round(statistics.median(ratings), 2),
            "rating_std": round(statistics.stdev(ratings), 2) if len(ratings) > 1 else 0,
            "rating_range": f"{min(ratings)}-{max(ratings)}",
            "response_count": len(ratings),
            "models_tested": len(stats["models"]),
            "rating_distribution": self._calculate_rating_distribution(ratings),
        }

    def _calculate_rating_distribution(self, ratings: list[int]) -> dict[str, int]:
        """Calculate rating distribution across standard buckets.

        Args:
            ratings: List of rating values.

        Returns:
            Dictionary with rating distribution.
        """
        return {
            "10": ratings.count(10),
            "9": ratings.count(9),
            "8": ratings.count(8),
            "7": ratings.count(7),
            "6": ratings.count(6),
            "<6": len([r for r in ratings if r < 6]),
        }

    def _calculate_percentiles(self, ratings: list[int]) -> dict[str, float]:
        """Calculate percentile statistics for ratings.

        Args:
            ratings: List of rating values.

        Returns:
            Dictionary with percentile statistics.
        """
        if not ratings:
            return {"25th": 0, "75th": 0, "90th": 0}

        sorted_ratings = sorted(ratings)
        n = len(sorted_ratings)

        return {
            "25th": round(sorted_ratings[int(0.25 * (n - 1))], 2),
            "75th": round(sorted_ratings[int(0.75 * (n - 1))], 2),
            "90th": round(sorted_ratings[int(0.9 * (n - 1))], 2),
        }

    def _generate_summary(
        self, model_analysis: dict[str, Any], prompt_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate overall summary statistics.

        Args:
            model_analysis: Analysis results for models.
            prompt_analysis: Analysis results for prompts.

        Returns:
            Dictionary with summary statistics.
        """
        if not model_analysis:
            return {"best_model": None, "worst_model": None, "performance_gap": 0}

        # Find best and worst performers
        best_model = max(model_analysis.items(), key=lambda x: x[1]["avg_rating"])
        worst_model = min(model_analysis.items(), key=lambda x: x[1]["avg_rating"])

        # Calculate performance gap
        performance_gap = best_model[1]["avg_rating"] - worst_model[1]["avg_rating"]

        # Calculate overall statistics
        all_ratings = []
        all_latencies = []
        for stats in model_analysis.values():
            all_ratings.extend([stats["avg_rating"]] * stats["total_responses"])
            if stats["avg_latency"] > 0:
                all_latencies.extend([stats["avg_latency"]] * stats["total_responses"])

        return {
            "best_model": {
                "name": best_model[0],
                "rating": best_model[1]["avg_rating"],
                "consistency": best_model[1]["rating_std"],
            },
            "worst_model": {
                "name": worst_model[0],
                "rating": worst_model[1]["avg_rating"],
                "consistency": worst_model[1]["rating_std"],
            },
            "performance_gap": round(performance_gap, 2),
            "overall_avg_rating": round(statistics.mean(all_ratings), 2) if all_ratings else 0,
            "overall_avg_latency": round(statistics.mean(all_latencies), 3)
            if all_latencies
            else 0,
            "total_models": len(model_analysis),
            "total_prompts": len(prompt_analysis),
        }

    def create_enhanced_html_report(self, analysis: dict[str, Any], output_file: str) -> None:
        """Create an enhanced HTML report with interactive visualizations.

        Args:
            analysis: Analysis results from analyze_evaluation_results().
            output_file: Path where to save the HTML report.
        """
        html_content = self._generate_html_header()
        html_content += self._generate_html_stats_section(analysis)
        html_content += self._generate_html_charts_section(analysis)
        html_content += self._generate_html_model_comparison(analysis)
        html_content += self._generate_html_footer()
        html_content += self._generate_html_scripts(analysis)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📊 Enhanced HTML report saved to {output_file}")

    def _generate_html_header(self) -> str:
        """Generate the HTML header with styles."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced LLM Benchmark Report with Evaluations</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 3em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.2em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .section {
            padding: 40px;
            border-bottom: 1px solid #eee;
        }
        .section h2 {
            color: #333;
            margin-bottom: 30px;
            font-weight: 400;
            font-size: 1.8em;
        }
        .chart-container {
            position: relative;
            height: 500px;
            margin: 30px 0;
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .model-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .model-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 4px solid #28a745;
        }
        .model-name {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 5px 0;
        }
        .metric-label {
            color: #666;
        }
        .metric-value {
            font-weight: bold;
            color: #333;
        }
        .rating-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin: 10px 0;
            overflow: hidden;
        }
        .rating-fill {
            height: 100%;
            background: linear-gradient(90deg, #dc3545 0%, #ffc107 50%, #28a745 100%);
            transition: width 0.3s ease;
        }
        .footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
            background: #f8f9fa;
        }
        .rating-distribution {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .rating-item {
            flex: 1;
            text-align: center;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .rating-number {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        .rating-count {
            font-size: 0.8em;
            color: #666;
        }
        .summary-card {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.5em;
        }
        .summary-card p {
            margin: 5px 0;
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Enhanced LLM Benchmark Report</h1>
            <p>with LLM-as-a-Judge Evaluations</p>
        </div>
"""

    def _generate_html_stats_section(self, analysis: dict[str, Any]) -> str:
        """Generate the statistics overview section."""
        summary = analysis["summary"]

        html = f"""
        <div class="summary-card">
            <h3>🏆 Performance Summary</h3>
            <p><strong>Best Model:</strong> {
            summary["best_model"]["name"].replace("blablador:", "")
            if summary["best_model"]
            else "N/A"
        }</p>
            <p><strong>Rating:</strong> {
            summary["best_model"]["rating"] if summary["best_model"] else 0
        }/10</p>
            <p><strong>Performance Gap:</strong> {summary["performance_gap"]} points</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{analysis["total_evaluations"]}</div>
                <div class="stat-label">Total Evaluations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis["models_tested"]}</div>
                <div class="stat-label">Models Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis["prompts_tested"]}</div>
                <div class="stat-label">Prompts Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary["overall_avg_rating"]:.1f}</div>
                <div class="stat-label">Overall Avg Rating</div>
            </div>
        </div>
"""
        return html

    def _generate_html_charts_section(self, analysis: dict[str, Any]) -> str:
        """Generate the charts section."""
        return """
        <div class="section">
            <h2>📊 Model Performance Overview</h2>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>📈 Rating Distribution</h2>
            <div class="chart-container">
                <canvas id="ratingChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>⚡ Latency vs Quality</h2>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
"""

    def _generate_html_model_comparison(self, analysis: dict[str, Any]) -> str:
        """Generate the detailed model comparison section."""
        html = """
        <div class="section">
            <h2>🏆 Detailed Model Analysis</h2>
            <div class="model-comparison">
"""

        # Sort models by average rating
        sorted_models = sorted(
            analysis["model_analysis"].items(), key=lambda x: x[1]["avg_rating"], reverse=True
        )

        for model, stats in sorted_models:
            model_name = model.replace("blablador:", "")
            rating_percentage = (stats["avg_rating"] / 10) * 100

            html += f"""
                <div class="model-card">
                    <div class="model-name">{model_name}</div>

                    <div class="metric">
                        <span class="metric-label">Average Rating</span>
                        <span class="metric-value">{stats["avg_rating"]}/10</span>
                    </div>

                    <div class="rating-bar">
                        <div class="rating-fill" style="width: {rating_percentage}%"></div>
                    </div>

                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value">{stats["success_rate"]}%</span>
                    </div>

                    <div class="metric">
                        <span class="metric-label">Avg Latency</span>
                        <span class="metric-value">{stats["avg_latency"]}s</span>
                    </div>

                    <div class="metric">
                        <span class="metric-label">Rating Range</span>
                        <span class="metric-value">
                            {stats["min_rating"]}-{stats["max_rating"]}
                        </span>
                    </div>

                    <div class="metric">
                        <span class="metric-label">Total Responses</span>
                        <span class="metric-value">{stats["total_responses"]}</span>
                    </div>

                    <div class="rating-distribution">
                        <div class="rating-item">
                            <div class="rating-number">10</div>
                            <div class="rating-count">{stats["rating_distribution"]["10"]}</div>
                        </div>
                        <div class="rating-item">
                            <div class="rating-number">9</div>
                            <div class="rating-count">{stats["rating_distribution"]["9"]}</div>
                        </div>
                        <div class="rating-item">
                            <div class="rating-number">8</div>
                            <div class="rating-count">{stats["rating_distribution"]["8"]}</div>
                        </div>
                        <div class="rating-item">
                            <div class="rating-number">7</div>
                            <div class="rating-count">{stats["rating_distribution"]["7"]}</div>
                        </div>
                        <div class="rating-item">
                            <div class="rating-number">6</div>
                            <div class="rating-count">{stats["rating_distribution"]["6"]}</div>
                        </div>
                        <div class="rating-item">
                            <div class="rating-number"><6</div>
                            <div class="rating-count">{stats["rating_distribution"]["<6"]}</div>
                        </div>
                    </div>
                </div>
"""

        html += """
            </div>
        </div>
"""
        return html

    def _generate_html_footer(self) -> str:
        """Generate the HTML footer."""
        return """
        <div class="footer">
            <p>Generated by HeLLMholtz Benchmark Suite with LLM-as-a-Judge evaluation</p>
        </div>
    </div>
"""

    def _generate_html_scripts(self, analysis: dict[str, Any]) -> str:
        """Generate the JavaScript for interactive charts."""
        # Prepare data for charts
        model_names = [m.replace("blablador:", "") for m in analysis["model_analysis"]]
        avg_ratings = [
            analysis["model_analysis"][m]["avg_rating"] for m in analysis["model_analysis"]
        ]
        latencies = [
            analysis["model_analysis"][m]["avg_latency"] for m in analysis["model_analysis"]
        ]

        # Rating distribution data
        rating_labels = ["10", "9", "8", "7", "6", "<6"]
        rating_datasets = []
        colors = ["#28a745", "#20c997", "#ffc107", "#fd7e14", "#dc3545", "#6c757d"]

        for i, model in enumerate(analysis["model_analysis"].keys()):
            model_name = model.replace("blablador:", "")
            distribution = analysis["model_analysis"][model]["rating_distribution"]
            data = [
                distribution["10"],
                distribution["9"],
                distribution["8"],
                distribution["7"],
                distribution["6"],
                distribution["<6"],
            ]
            rating_datasets.append(
                {
                    "label": model_name,
                    "data": data,
                    "backgroundColor": colors[i % len(colors)],
                    "borderWidth": 1,
                }
            )

        return f"""
    <script>
        // Performance Chart
        const perfCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(perfCtx, {{
            type: 'bar',
            data: {{
                labels: {model_names!r},
                datasets: [{{
                    label: 'Average Rating (/10)',
                    data: {avg_ratings!r},
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 10,
                        title: {{
                            display: true,
                            text: 'Rating'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Model'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Model Performance Ratings'
                    }}
                }}
            }}
        }});

        // Rating Distribution Chart
        const ratingCtx = document.getElementById('ratingChart').getContext('2d');
        new Chart(ratingCtx, {{
            type: 'bar',
            data: {{
                labels: {rating_labels!r},
                datasets: {rating_datasets!r}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Count'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Rating'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Rating Distribution by Model'
                    }}
                }}
            }}
        }});

        // Latency vs Quality Chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        new Chart(latencyCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Models',
                    data: {list(zip(latencies, avg_ratings, strict=False))!r},
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 12
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Average Latency (seconds)'
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        max: 10,
                        title: {{
                            display: true,
                            text: 'Average Rating'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Latency vs Quality Trade-off'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const modelName = {model_names!r}[context.dataIndex];
                                return `${{modelName}}: ${{context.parsed.x.toFixed(3)}}s, ` +
                                    `Rating: ${{context.parsed.y}}/10`;
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    def print_analysis_summary(self, analysis: dict[str, Any]) -> None:
        """Print a comprehensive analysis summary to console.

        Args:
            analysis: Analysis results from analyze_evaluation_results().
        """
        summary = analysis["summary"]

        print("🔍 EVALUATION ANALYSIS RESULTS")
        print("=" * 50)

        print(f"📊 Total Evaluations: {analysis['total_evaluations']}")
        print(f"🤖 Models Tested: {analysis['models_tested']}")
        print(f"❓ Prompts Evaluated: {analysis['prompts_tested']}")
        print(f"📈 Overall Avg Rating: {summary['overall_avg_rating']:.2f}/10")
        print(f"⚡ Overall Avg Latency: {summary['overall_avg_latency']:.3f}s")

        print("\n🏆 MODEL PERFORMANCE RANKING:")
        print("-" * 30)

        for model, stats in sorted(
            analysis["model_analysis"].items(), key=lambda x: x[1]["avg_rating"], reverse=True
        ):
            model_name = model.replace("blablador:", "")
            print(f"🥇 {model_name}")
            print(f"   ⭐ Rating: {stats['avg_rating']}/10 (σ={stats['rating_std']:.2f})")
            print(f"   📈 Success: {stats['success_rate']}%")
            print(f"   ⚡ Latency: {stats['avg_latency']}s")
            print(f"   📊 Ratings: {stats['rating_distribution']}")
            print(
                f"   📝 Responses: {stats['total_responses']} across "
                f"{stats['total_prompts']} prompts"
            )
            print()

        print("🎯 CRITICAL ANALYSIS:")
        print("-" * 20)

        if summary["best_model"] and summary["worst_model"]:
            print(f"🏆 BEST PERFORMER: {summary['best_model']['name'].replace('blablador:', '')}")
            print(f"   Rating: {summary['best_model']['rating']}/10")
            print(f"   Consistency: ±{summary['best_model']['consistency']:.2f} rating variation")

            print(
                f"\n💩 WORST PERFORMER: {summary['worst_model']['name'].replace('blablador:', '')}"
            )
            print(f"   Rating: {summary['worst_model']['rating']}/10")
            print(f"   Consistency: ±{summary['worst_model']['consistency']:.2f} rating variation")

            # Performance gap analysis
            gap_description = (
                "significant"
                if summary["performance_gap"] > 2
                else "moderate"
                if summary["performance_gap"] > 1
                else "minimal"
            )
            print(
                f"\n📈 PERFORMANCE GAP: {summary['performance_gap']:.1f} points "
                f"between best and worst"
            )
            print(f"   This indicates {gap_description} quality differences")

        print("\n💡 RECOMMENDATIONS:")
        print("-" * 15)
        if summary["performance_gap"] > 2:
            print("• Consider focusing development efforts on top-performing models")
            print("• Investigate why lower-performing models struggle")
        elif summary["performance_gap"] > 1:
            print("• Moderate differences suggest room for optimization")
            print("• Consider fine-tuning lower-performing models")
        else:
            print("• All models perform similarly - focus on other factors like latency")

        # Latency analysis
        if analysis["model_analysis"]:
            fastest_model = min(
                analysis["model_analysis"].items(), key=lambda x: x[1]["avg_latency"]
            )
            slowest_model = max(
                analysis["model_analysis"].items(), key=lambda x: x[1]["avg_latency"]
            )

            latency_ratio = slowest_model[1]["avg_latency"] / fastest_model[1]["avg_latency"]
            if latency_ratio > 2:
                print(
                    "• Significant latency differences - consider model "
                    "selection based on speed requirements"
                )


def analyze_evaluations_cli(results_file: str, output_file: str | None = None) -> dict[str, Any]:
    """Convenience function for CLI usage.

    Args:
        results_file: Path to evaluation results file.
        output_file: Optional path for HTML report output.

    Returns:
        Analysis results dictionary.
    """
    analyzer = EvaluationAnalyzer()
    analysis = analyzer.analyze_evaluation_results(results_file)

    # Print summary to console
    analyzer.print_analysis_summary(analysis)

    # Generate HTML report if requested
    if output_file:
        analyzer.create_enhanced_html_report(analysis, output_file)

    return analysis
