"""Tests for hellmholtz.cli.benchmark implementation functions."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.cli.benchmark import _bench_impl, _chart_impl, _analyze_impl, _report_impl


def _results_file(tmp_path: Path, data: list | None = None) -> Path:
    if data is None:
        data = [
            {
                "model": "openai:gpt-4o",
                "prompt_id": "p1",
                "temperature": 0.7,
                "success": True,
                "latency_seconds": 1.5,
                "response_text": "ok",
                "error_message": None,
                "timestamp": "2024-01-01T00:00:00",
                "input_tokens": 10,
                "output_tokens": 20,
            }
        ]
    f = tmp_path / "results.json"
    f.write_text(json.dumps(data))
    return f


# ---- _bench_impl ----


class TestBenchImpl:
    @patch("hellmholtz.benchmark.runner.save_results")
    @patch("hellmholtz.benchmark.run_benchmarks", return_value=[])
    @patch("hellmholtz.cli.benchmark.parse_temperatures", return_value=[0.1])
    @patch("hellmholtz.cli.benchmark.parse_models", return_value=["openai:gpt-4o"])
    @patch("hellmholtz.benchmark.prompts.get_prompts_by_category", return_value=[{"id": "p1"}])
    def test_basic_run(self, mock_prompts, mock_models, mock_temps, mock_bench, mock_save):
        _bench_impl(
            models="openai:gpt-4o",
            prompts_file=None,
            prompts_category=None,
            all_prompts=False,
            temperatures="0.1",
            max_tokens=None,
            replications=3,
            evaluate_with=None,
            system_prompt=None,
        )
        mock_bench.assert_called_once()
        mock_save.assert_called_once()

    @patch("hellmholtz.benchmark.runner.save_results")
    @patch("hellmholtz.benchmark.run_benchmarks", return_value=[])
    @patch("hellmholtz.cli.benchmark.parse_temperatures", return_value=[0.1])
    @patch("hellmholtz.cli.benchmark.parse_models", return_value=["openai:gpt-4o"])
    @patch("hellmholtz.cli.common.load_prompts_from_file")
    def test_with_prompts_file(self, mock_load, mock_models, mock_temps, mock_bench, mock_save, tmp_path):
        pf = tmp_path / "prompts.json"
        pf.write_text("[]")
        mock_load.return_value = []

        _bench_impl(
            models="openai:gpt-4o",
            prompts_file=pf,
            prompts_category=None,
            all_prompts=False,
            temperatures="0.1",
            max_tokens=None,
            replications=1,
            evaluate_with=None,
            system_prompt=None,
        )
        mock_load.assert_called_once_with(pf)

    @patch("hellmholtz.benchmark.runner.save_results")
    @patch("hellmholtz.benchmark.run_benchmarks", return_value=[])
    @patch("hellmholtz.cli.benchmark.parse_temperatures", return_value=[0.1])
    @patch("hellmholtz.cli.benchmark.parse_models", return_value=["openai:gpt-4o"])
    @patch("hellmholtz.benchmark.prompts.get_prompts_by_category", return_value=[{"id": "p1"}])
    def test_with_prompts_category(self, mock_prompts, mock_models, mock_temps, mock_bench, mock_save):
        _bench_impl(
            models="openai:gpt-4o",
            prompts_file=None,
            prompts_category="reasoning",
            all_prompts=False,
            temperatures="0.1",
            max_tokens=None,
            replications=1,
            evaluate_with=None,
            system_prompt=None,
        )
        mock_prompts.assert_called_with("reasoning")

    @patch("hellmholtz.benchmark.runner.save_results")
    @patch("hellmholtz.benchmark.run_benchmarks", return_value=[])
    @patch("hellmholtz.cli.benchmark.parse_temperatures", return_value=[0.1])
    @patch("hellmholtz.cli.benchmark.parse_models", return_value=["openai:gpt-4o"])
    @patch("hellmholtz.benchmark.prompts.get_all_prompts", return_value=[{"id": "p1"}])
    def test_with_all_prompts(self, mock_all, mock_models, mock_temps, mock_bench, mock_save):
        _bench_impl(
            models="openai:gpt-4o",
            prompts_file=None,
            prompts_category=None,
            all_prompts=True,
            temperatures="0.1",
            max_tokens=None,
            replications=1,
            evaluate_with=None,
            system_prompt=None,
        )
        mock_all.assert_called_once()

    @patch("hellmholtz.benchmark.runner.save_results")
    @patch("hellmholtz.benchmark.evaluator.evaluate_responses", return_value=[])
    @patch("hellmholtz.client.check_model_availability", return_value=True)
    @patch("hellmholtz.benchmark.run_benchmarks", return_value=[])
    @patch("hellmholtz.cli.benchmark.parse_temperatures", return_value=[0.1])
    @patch("hellmholtz.cli.benchmark.parse_models", return_value=["openai:gpt-4o"])
    @patch("hellmholtz.benchmark.prompts.get_prompts_by_category", return_value=[{"id": "p1"}])
    def test_evaluate_with_available_judge(
        self, mock_prompts, mock_models, mock_temps, mock_bench, mock_check, mock_eval, mock_save
    ):
        _bench_impl(
            models="openai:gpt-4o",
            prompts_file=None,
            prompts_category=None,
            all_prompts=False,
            temperatures="0.1",
            max_tokens=None,
            replications=1,
            evaluate_with="openai:gpt-4o",
            system_prompt=None,
        )
        mock_eval.assert_called_once()
        mock_save.assert_called_once()

    @patch("hellmholtz.benchmark.runner.save_results")
    @patch("hellmholtz.client.check_model_availability", return_value=False)
    @patch("hellmholtz.benchmark.run_benchmarks", return_value=[])
    @patch("hellmholtz.cli.benchmark.parse_temperatures", return_value=[0.1])
    @patch("hellmholtz.cli.benchmark.parse_models", return_value=["openai:gpt-4o"])
    @patch("hellmholtz.benchmark.prompts.get_prompts_by_category", return_value=[{"id": "p1"}])
    def test_evaluate_with_unavailable_judge(
        self, mock_prompts, mock_models, mock_temps, mock_bench, mock_check, mock_save
    ):
        _bench_impl(
            models="openai:gpt-4o",
            prompts_file=None,
            prompts_category=None,
            all_prompts=False,
            temperatures="0.1",
            max_tokens=None,
            replications=1,
            evaluate_with="nonexistent:model",
            system_prompt=None,
        )
        mock_save.assert_called_once()


# ---- _report_impl ----


class TestReportImpl:
    @patch("hellmholtz.cli.benchmark.save_report_to_file")
    @patch("hellmholtz.reporting.generate_markdown_report", return_value="# Report")
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_markdown_format(self, mock_load, mock_gen, mock_save, tmp_path):
        rf = _results_file(tmp_path)
        _report_impl(rf, "markdown", None)
        mock_gen.assert_called_once()
        mock_save.assert_called_once()

    @patch("hellmholtz.cli.benchmark.save_report_to_file")
    @patch("hellmholtz.reporting.generate_html_report", return_value="<html></html>")
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_html_format(self, mock_load, mock_gen, mock_save, tmp_path):
        rf = _results_file(tmp_path)
        _report_impl(rf, "html", None)
        mock_gen.assert_called_once()
        mock_save.assert_called_once()

    @patch("hellmholtz.cli.benchmark.save_report_to_file")
    @patch("hellmholtz.reporting.generate_html_report_simple", return_value="<html></html>")
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_html_simple_format(self, mock_load, mock_gen, mock_save, tmp_path):
        rf = _results_file(tmp_path)
        _report_impl(rf, "html-simple", None)
        mock_gen.assert_called_once()

    @patch("hellmholtz.cli.benchmark.save_report_to_file")
    @patch("hellmholtz.reporting.generate_html_report_detailed", return_value="<html></html>")
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_html_detailed_format(self, mock_load, mock_gen, mock_save, tmp_path):
        rf = _results_file(tmp_path)
        _report_impl(rf, "html-detailed", None)
        mock_gen.assert_called_once()

    @patch("hellmholtz.cli.benchmark.save_report_to_file")
    @patch("hellmholtz.reporting.generate_html_report_full", return_value="<html></html>")
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_html_full_format(self, mock_load, mock_gen, mock_save, tmp_path):
        rf = _results_file(tmp_path)
        _report_impl(rf, "html-full", None)
        mock_gen.assert_called_once()

    @patch("hellmholtz.cli.benchmark.save_report_to_file")
    @patch("hellmholtz.reporting.generate_markdown_report", return_value="# Report")
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_custom_output(self, mock_load, mock_gen, mock_save, tmp_path):
        rf = _results_file(tmp_path)
        custom = tmp_path / "custom_report.md"
        _report_impl(rf, "markdown", custom)
        mock_save.assert_called_once_with("# Report", custom)


# ---- _chart_impl ----


class TestChartImpl:
    @patch("hellmholtz.reporting.chart.generate_performance_chart")
    def test_success(self, mock_chart, tmp_path):
        rf = _results_file(tmp_path)
        out = tmp_path / "chart.png"
        _chart_impl(rf, out)
        mock_chart.assert_called_once()

    @patch(
        "hellmholtz.reporting.chart.generate_performance_chart",
        side_effect=ImportError("no module"),
    )
    @patch("hellmholtz.cli.benchmark.handle_error")
    def test_matplotlib_import_error(self, mock_err, mock_chart, tmp_path):
        rf = _results_file(tmp_path)
        out = tmp_path / "chart.png"
        _chart_impl(rf, out)
        mock_err.assert_called_once()

    @patch("hellmholtz.reporting.chart.generate_performance_chart")
    @patch("hellmholtz.cli.benchmark.generate_output_path", return_value=Path("/tmp/default.png"))
    @patch("hellmholtz.reporting.load_results", return_value=[])
    def test_auto_output_path(self, mock_load, mock_path, mock_chart, tmp_path):
        rf = _results_file(tmp_path)
        _chart_impl(rf, None)
        mock_path.assert_called_once()


# ---- _analyze_impl ----


class TestAnalyzeImpl:
    @patch("hellmholtz.evaluation_analysis.analyze_evaluations_cli")
    def test_without_html_report(self, mock_analyze, tmp_path):
        rf = _results_file(tmp_path)
        _analyze_impl(rf, None)
        mock_analyze.assert_called_once_with(str(rf), None)

    @patch("hellmholtz.evaluation_analysis.analyze_evaluations_cli")
    def test_with_html_report(self, mock_analyze, tmp_path):
        rf = _results_file(tmp_path)
        html = tmp_path / "analysis.html"
        _analyze_impl(rf, html)
        mock_analyze.assert_called_once_with(str(rf), str(html))
