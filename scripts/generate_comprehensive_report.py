#!/usr/bin/env python3
"""
Comprehensive Weekly Benchmark Report Generator

Generates reports combining benchmark results (which models perform how on which
task categories) with model availability status data. Used by the weekly
benchmark GitHub Actions workflow.

Input benchmark file may be:
  * a flat JSON array of result records (format written by ``hellm bench``),
  * a JSON Lines file (``*.partial.jsonl`` incremental output), or
  * a legacy dict with a top-level ``{"results": {model: {...}}}`` mapping.
"""

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml

# Optional: use the prompt registry for authoritative task categories.
get_prompt_by_id: Callable[..., Any] | None = None
try:  # pragma: no cover - import depends on install mode
    from hellmholtz.benchmark.prompts import get_prompt_by_id as _registry_get_prompt

    get_prompt_by_id = _registry_get_prompt
except ImportError:  # pragma: no cover
    pass

_TRAILING_NUM = re.compile(r"_(\d{3,})$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_benchmark_records(results_file: str) -> list[dict[str, Any]]:  # noqa: C901
    """Load benchmark result records from JSON array, JSONL, or legacy dict file."""
    path = Path(results_file)
    try:
        text = path.read_text()
    except Exception as e:  # noqa: BLE001
        print(f"Error reading {results_file}: {e}", file=sys.stderr)
        return []

    records: list[dict[str, Any]] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None  # fall through to JSONL parsing

    if isinstance(data, list):
        records = [r for r in data if isinstance(r, dict)]
    elif isinstance(data, dict) and isinstance(data.get("results"), dict):
        # Legacy format: {"results": {model: {"prompt_results": [...]}}}
        for model_name, model_data in data["results"].items():
            for pr in model_data.get("prompt_results", []) or []:
                if isinstance(pr, dict):
                    records.append({**pr, "model": pr.get("model", model_name)})
    else:
        # JSONL: one record per line (incremental partial results)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)

    records = [r for r in records if r.get("prompt_id") and r.get("model")]
    if not records:
        print(f"No benchmark records found in {results_file}", file=sys.stderr)
    return records


def load_model_status() -> dict[str, Any]:
    """Load model status from YAML file."""
    try:
        with open("models_status.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model status: {e}", file=sys.stderr)
        return {}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def task_of(record: dict[str, Any]) -> str:
    """Determine the task category for a result record."""
    prompt_id = str(record["prompt_id"])
    if get_prompt_by_id is not None:
        try:
            prompt = get_prompt_by_id(prompt_id)
            if prompt is not None and prompt.category:
                return str(prompt.category)
        except Exception:  # noqa: BLE001
            pass
    # Fallback: strip trailing "_001"-style numbering
    return _TRAILING_NUM.sub("", prompt_id) or "other"


def aggregate_models(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-model aggregate metrics."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_model[str(r["model"])].append(r)

    out: dict[str, dict[str, Any]] = {}
    for model, rs in by_model.items():
        succ = [r for r in rs if r.get("success")]
        lat = [float(r["latency_seconds"]) for r in succ if r.get("latency_seconds") is not None]
        tps = [
            float(r["tokens_per_sec"])
            for r in succ
            if r.get("tokens_per_sec") is not None and float(r["tokens_per_sec"]) > 0
        ]
        ratings = [
            float(r["rating"])
            for r in rs
            if r.get("rating") is not None and float(r["rating"]) > 0
        ]
        tasks = {task_of(r) for r in rs}
        out[model] = {
            "requests": len(rs),
            "successes": len(succ),
            "success_rate": len(succ) / len(rs) if rs else 0.0,
            "avg_latency": sum(lat) / len(lat) if lat else 0.0,
            "avg_tokens_per_sec": sum(tps) / len(tps) if tps else 0.0,
            "avg_rating": sum(ratings) / len(ratings) if ratings else None,
            "tasks": sorted(tasks),
            "prompts": len({str(r["prompt_id"]) for r in rs}),
        }
    return out


def aggregate_tasks(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """model -> task -> metrics matrix."""
    bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        bucket[(str(r["model"]), task_of(r))].append(r)

    matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (model, task), rs in bucket.items():
        succ = [r for r in rs if r.get("success")]
        lat = [float(r["latency_seconds"]) for r in succ if r.get("latency_seconds") is not None]
        ratings = [
            float(r["rating"])
            for r in rs
            if r.get("rating") is not None and float(r["rating"]) > 0
        ]
        matrix[model][task] = {
            "requests": len(rs),
            "success_rate": len(succ) / len(rs) if rs else 0.0,
            "avg_latency": sum(lat) / len(lat) if lat else 0.0,
            "avg_rating": sum(ratings) / len(ratings) if ratings else None,
        }
    return dict(matrix)


def all_tasks(records: list[dict[str, Any]]) -> list[str]:
    return sorted({task_of(r) for r in records})


# ---------------------------------------------------------------------------
# Report pieces
# ---------------------------------------------------------------------------


def _model_sections(model_status: dict[str, Any]) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Group models from models_status.yaml by category."""
    categories: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, info in (model_status.get("models") or {}).items():
        if not isinstance(info, dict):
            info = {}
        categories.setdefault(str(info.get("category", "other")), []).append((name, info))
    return categories


def _availability_summary(model_status: dict[str, Any]) -> tuple[int, int, int]:
    models = model_status.get("models") or {}
    total = len(models)
    available = sum(1 for m in models.values() if isinstance(m, dict) and m.get("available"))
    tested = sum(
        1 for m in models.values() if isinstance(m, dict) and m.get("latency") is not None
    )
    return total, available, tested


def _cell(cell_data: dict[str, Any] | None, kind: str) -> str:
    if not cell_data:
        return "-"
    if kind == "rate":
        return f"{cell_data['success_rate']:.0%}"
    if kind == "latency":
        return f"{cell_data['avg_latency']:.2f}s" if cell_data["avg_latency"] else "-"
    if kind == "rating":
        r = cell_data.get("avg_rating")
        return f"{r:.1f}" if r is not None else "-"
    return "-"


def build_findings(
    model_status: dict[str, Any],
    model_stats: dict[str, dict[str, Any]],
    task_matrix: dict[str, dict[str, dict[str, Any]]],
    tasks: list[str],
) -> list[str]:
    """Human-readable takeaways: availability + best/worst per task."""
    findings: list[str] = []
    total, available, _ = _availability_summary(model_status)
    if total:
        findings.append(
            f"**Availability:** {available}/{total} registered models are currently available."
        )

    if not model_stats:
        findings.append(
            "**No benchmark data** in this run — check API access or the results file."
        )
        return findings

    for model, st in model_stats.items():
        if st["success_rate"] == 0:
            findings.append(
                f"**{model}: all {st['requests']} requests failed**"
                " — likely unavailable or misnamed."
            )

    for task in tasks:
        scored = [
            (m, cm)
            for m, cells in task_matrix.items()
            if (cm := cells.get(task)) and cm["requests"] > 0
        ]
        if not scored:
            continue
        best = max(scored, key=lambda x: (x[1]["success_rate"], -x[1]["avg_latency"]))
        fastest = min(
            [x for x in scored if x[1]["avg_latency"] > 0],
            key=lambda x: x[1]["avg_latency"],
            default=None,
        )
        top_rate = max(cm["success_rate"] for _, cm in scored)
        tied = [m for m, cm in scored if cm["success_rate"] == top_rate]
        if len(tied) > 1:
            # No reliability signal — report the latency winner instead.
            line = (
                f"**{task.title()}:** {len(tied)}/{len(scored)} models tied at "
                f"{top_rate:.0%} success — fastest `{best[0]}` "
                f"({best[1]['avg_latency']:.2f}s avg)"
            )
        else:
            line = (
                f"**{task.title()}:** most reliable `{best[0]}` "
                f"({best[1]['success_rate']:.0%}, {best[1]['avg_latency']:.2f}s avg)"
            )
            if fastest and fastest[0] != best[0]:
                line += f"; fastest `{fastest[0]}` ({fastest[1]['avg_latency']:.2f}s)"
        findings.append(line)
    return findings


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_comprehensive_markdown_report(  # noqa: C901
    records: list[dict[str, Any]], model_status: dict[str, Any], results_file: str
) -> str:
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    model_stats = aggregate_models(records)
    task_matrix = aggregate_tasks(records)
    tasks = all_tasks(records)
    total_models, available_models, tested_models = _availability_summary(model_status)

    lines = [
        f"# Weekly Benchmark Report - {report_date}",
        "",
        "## 📊 Executive Summary",
        "",
        f"**Report Generated:** {report_date}",
        f"**Benchmark Results:** `{Path(results_file).name}`",
        f"**Requests Recorded:** {len(records)}",
        f"**Models Benchmarked:** {len(model_stats)}",
        f"**Task Categories:** {', '.join(t.title() for t in tasks) if tasks else 'none'}",
        "",
        "## 🔍 Model Availability Status",
        "",
        f"- **Total Registered Models:** {total_models}",
        f"- **Available:** {available_models}"
        + (f" ({available_models / total_models * 100:.1f}%)" if total_models else ""),
        f"- **Latency-Tested:** {tested_models}",
        f"- **Last Updated:** {model_status.get('# Last updated', 'Unknown')}",
        "",
    ]

    # Availability by category
    categories = _model_sections(model_status)
    if categories:
        lines.extend(["### Available Models by Category", ""])
        for cat, model_list in sorted(categories.items()):
            lines.append(f"#### {cat.replace('_', ' ').title()} Models")
            for name, info in sorted(
                model_list, key=lambda x: (not x[1].get("available", False), x[0])
            ):
                status = "✅" if info.get("available", False) else "❌"
                latency = f" — {info['latency']}s" if info.get("latency") is not None else ""
                lines.append(f"- {status} `{name}`{latency}")
            lines.append("")

    # Overall performance
    lines.extend(
        [
            "## 📈 Overall Performance by Model",
            "",
            "| Model | Success Rate | Avg Latency | Tokens/s | Prompts | Requests | Ratings |",
            "|-------|-------------|-------------|----------|---------|----------|---------|",
        ]
    )
    for model in sorted(model_stats, key=lambda m: -model_stats[m]["success_rate"]):
        st = model_stats[model]
        tps = f"{st['avg_tokens_per_sec']:.1f}" if st["avg_tokens_per_sec"] else "-"
        rating = f"{st['avg_rating']:.1f}" if st["avg_rating"] is not None else "-"
        lines.append(
            f"| `{model}` | {st['success_rate']:.1%} | {st['avg_latency']:.2f}s "
            f"| {tps} | {st['prompts']} | {st['requests']} | {rating} |"
        )
    lines.append("")

    # Per-task matrices
    if tasks and task_matrix:
        for kind, title, fmt in (
            ("rate", "Success Rate by Task", "🟢"),
            ("latency", "Avg Latency by Task", "⚡"),
        ):
            lines.extend(
                [
                    f"## {fmt} {title}",
                    "",
                    "| Model | " + " | ".join(t.title() for t in tasks) + " |",
                    "|-------|" + "|".join(["------"] * len(tasks)) + "|",
                ]
            )
            for model in sorted(task_matrix):
                row = [f"`{model}`"]
                for t in tasks:
                    cell = task_matrix[model].get(t)
                    val = _cell(cell, kind)
                    if kind == "rate" and cell:
                        val = f"{val} ({cell['requests']})"
                    row.append(val)
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Findings / recommendations
    lines.extend(["## 📋 Findings & Recommendations", ""])
    for f in build_findings(model_status, model_stats, task_matrix, tasks):
        lines.append(f"- {f}")
    lines.append("")

    lines.extend(
        [
            "## 🔗 Links",
            f"- Source results: `{Path(results_file).name}`"
            " (published as [weekly_benchmark_results.json](weekly_benchmark_results.json))",
            "- [Model Status YAML](../models_status.yaml)",
            "- [HTML Report](weekly_benchmark_comprehensive.html)",
            "- [Performance Chart](weekly_benchmark_chart.png)",
            "",
            "---",
            "*This report is automatically generated weekly by GitHub Actions.*",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def generate_comprehensive_html_report(  # noqa: C901
    records: list[dict[str, Any]], model_status: dict[str, Any], results_file: str
) -> str:
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    model_stats = aggregate_models(records)
    task_matrix = aggregate_tasks(records)
    tasks = all_tasks(records)
    total_models, available_models, tested_models = _availability_summary(model_status)
    categories = _model_sections(model_status)

    def esc(s: Any) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html = [
        "<!DOCTYPE html><html><head>",
        f"<title>Weekly Benchmark Report - {report_date}</title>",
        "<style>",
        "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
        "margin:40px;line-height:1.6;}",
        ".header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;"
        "padding:30px;border-radius:10px;margin-bottom:30px;}",
        ".summary{background:#f8f9fa;padding:20px;border-radius:8px;margin:20px 0;}",
        ".available{color:#28a745;} .unavailable{color:#dc3545;}",
        ".metric{font-size:2em;font-weight:bold;margin:10px 0;}",
        "table{width:100%;border-collapse:collapse;margin:20px 0;background:white;}",
        "th,td{border:1px solid #ddd;padding:10px;text-align:left;font-size:0.95em;}",
        "th{background:#f8f9fa;font-weight:600;}",
        "td.good{background:#e6f6e6;} td.bad{background:#fdecea;}",
        ".recommendations{background:#fff3cd;border:1px solid #ffeaa7;"
        "padding:20px;border-radius:5px;margin:20px 0;}",
        ".footer{margin-top:40px;padding-top:20px;border-top:1px solid #e9ecef;"
        "color:#6c757d;font-size:0.9em;}",
        "</style></head><body>",
        '<div class="header"><h1>📊 Weekly Benchmark Report</h1>',
        f"<p><strong>Generated:</strong> {report_date}</p>",
        f"<p><strong>Requests:</strong> {len(records)} | "
        f"<strong>Models Benchmarked:</strong> {len(model_stats)}</p></div>",
        '<div class="summary"><h2>🔍 Model Availability</h2><ul>',
        f"<li><strong>Total Models:</strong> {total_models}</li>",
        f"<li><strong>Available:</strong> {available_models}</li>",
        f"<li><strong>Latency-Tested:</strong> {tested_models}</li></ul>",
    ]
    for cat, model_list in sorted(categories.items()):
        html.append(f"<h3>{esc(cat.replace('_', ' ').title())} Models</h3><ul>")
        for name, info in sorted(
            model_list, key=lambda x: (not x[1].get("available", False), x[0])
        ):
            css = "available" if info.get("available") else "unavailable"
            icon = "✅" if info.get("available") else "❌"
            latency = f" ({info['latency']}s)" if info.get("latency") is not None else ""
            html.append(f'<li class="{css}">{icon} {esc(name)}{latency}</li>')
        html.append("</ul>")
    html.append("</div>")

    # Overall performance table
    html.append('<div class="summary"><h2>📈 Overall Performance by Model</h2>')
    html.append(
        "<table><thead><tr><th>Model</th><th>Success Rate</th><th>Avg Latency</th>"
        "<th>Tokens/s</th><th>Prompts</th><th>Requests</th><th>Avg Rating</th></tr></thead><tbody>"
    )
    for model in sorted(model_stats, key=lambda m: -model_stats[m]["success_rate"]):
        st = model_stats[model]
        cls = (
            ' class="good"'
            if st["success_rate"] >= 0.95
            else (' class="bad"' if st["success_rate"] < 0.8 else "")
        )
        rating = f"{st['avg_rating']:.1f}" if st["avg_rating"] is not None else "-"
        tps = f"{st['avg_tokens_per_sec']:.1f}" if st["avg_tokens_per_sec"] else "-"
        html.append(
            f"<tr><td>{esc(model)}</td><td{cls}>{st['success_rate']:.1%}</td>"
            f"<td>{st['avg_latency']:.2f}s</td><td>{tps}</td><td>{st['prompts']}</td>"
            f"<td>{st['requests']}</td><td>{rating}</td></tr>"
        )
    html.append("</tbody></table></div>")

    # Per-task matrices
    for kind, title, icon in (
        ("rate", "Success Rate by Task", "🟢"),
        ("latency", "Avg Latency by Task", "⚡"),
    ):
        if not tasks or not task_matrix:
            continue
        html.append(f'<div class="summary"><h2>{icon} {title}</h2>')
        html.append(
            "<table><thead><tr><th>Model</th>"
            + "".join(f"<th>{esc(t.title())}</th>" for t in tasks)
            + "</tr></thead><tbody>"
        )
        for model in sorted(task_matrix):
            html.append(f"<tr><td>{esc(model)}</td>")
            for t in tasks:
                cell = task_matrix[model].get(t)
                val = _cell(cell, kind)
                cls = ""
                if kind == "rate" and cell:
                    val = f"{val} ({cell['requests']})"
                    cls = (
                        ' class="good"'
                        if cell["success_rate"] >= 0.95
                        else (' class="bad"' if cell["success_rate"] < 0.8 else "")
                    )
                html.append(f"<td{cls}>{esc(val)}</td>")
            html.append("</tr>")
        html.append("</tbody></table></div>")

    # Findings
    html.append('<div class="recommendations"><h2>💡 Findings &amp; Recommendations</h2><ul>')
    for f in build_findings(model_status, model_stats, task_matrix, tasks):
        # findings use **bold** markdown; convert minimally
        safe = esc(f).replace("**", "<strong>", 1)
        parts = safe.split("**")
        if len(parts) >= 3:
            safe = parts[0] + "<strong>" + parts[1] + "</strong>" + "".join(parts[2:])
        html.append(f"<li>{safe}</li>")
    html.append("</ul></div>")

    html.append(
        f'<div class="footer"><p><strong>Links:</strong> '
        f"Source results <code>{esc(Path(results_file).name)}</code> "
        f'(<a href="weekly_benchmark_results.json">published JSON</a>) | '
        f'<a href="../models_status.yaml">Model Status YAML</a> | '
        f'<a href="weekly_benchmark_chart.png">Performance Chart</a></p>'
        f"<p><em>Automatically generated weekly by GitHub Actions.</em></p></div>"
        f"</body></html>"
    )
    return "\n".join(html)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main function to generate comprehensive reports."""
    if len(sys.argv) != 2:
        print("Usage: python generate_comprehensive_report.py <results_file>", file=sys.stderr)
        sys.exit(1)

    results_file = sys.argv[1]

    records = load_benchmark_records(results_file)
    model_status = load_model_status()

    markdown_report = generate_comprehensive_markdown_report(records, model_status, results_file)
    html_report = generate_comprehensive_html_report(records, model_status, results_file)

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weekly_benchmark_comprehensive.md").write_text(markdown_report)
    (out_dir / "weekly_benchmark_comprehensive.html").write_text(html_report)

    print(
        f"✅ Comprehensive reports generated successfully "
        f"({len(records)} records, {len(aggregate_models(records))} models)"
    )


if __name__ == "__main__":
    main()
