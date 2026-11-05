#!/usr/bin/env python
"""Build the benchmark dashboard, README leaderboard and history from a results file.

Outputs
-------
docs/benchmark/index.html   self-contained interactive dashboard (served by GitHub Pages)
docs/benchmark/data.json    the same data for other tools
reports/leaderboard.svg     static bar chart (renders on GitHub, light + dark)
reports/leaderboard.md      markdown leaderboard
reports/history.json        rolling per-run scores (trend lines, "new model" badges)
README.md                   section between the LEADERBOARD markers is rewritten

Usage:
    python -P scripts/build_dashboard.py results/benchmark_X.judged.json
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import html
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hellmholtz.benchmark.prompts import get_all_prompts  # noqa: E402
from hellmholtz.benchmark.scoring import leaderboard, model_key  # noqa: E402

HISTORY_LIMIT = 60
README_START, README_END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
FAMILIES = [
    ("Qwen", r"qwen|hyperqwen|swift"),
    ("DeepSeek", r"deepseek"),
    ("Mistral", r"mistral|european"),
    ("Gemma", r"gemma"),
    ("Kimi", r"kimi"),
    ("GLM", r"glm"),
    ("MiMo", r"mimo"),
    ("Apertus", r"apertus"),
    ("GPT-OSS", r"gpt-oss"),
    ("MiniMax", r"minimax"),
    ("Other", r".*"),
]


def family(name: str) -> str:
    low = name.lower()
    return next(label for label, pat in FAMILIES if re.search(pat, low))


def display_name(name: str) -> str:
    """Prefer the real model name when the registry name is a marketing tagline."""
    try:
        from hellmholtz.providers.models.blablador_models import KNOWN_MODELS

        for m in KNOWN_MODELS:
            if m.name == name and " - " in m.api_id:
                parts = m.api_id.split(" - ")
                if len(parts) >= 3 and name.count(" ") >= 3 and len(parts[2]) <= 30:
                    return str(parts[2])
    except Exception:  # registry is cosmetic only
        pass
    return name


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return list(json.loads(text))
    records = []
    for line in text.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # blank or torn line from an interrupted run
    return records


KEEP = (
    "model",
    "prompt_id",
    "category",
    "success",
    "final_score",
    "auto_score",
    "judge_score",
    "judge_spread",
    "latency_seconds",
    "tokens_per_sec",
    "output_tokens",
    "finish_reason",
    "ttft_seconds",
    "reasoning_tokens",
    "timestamp",
)


def compact_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Slim a result for the rolling store: scores and diagnostics, no response text."""
    out = {k: rec.get(k) for k in KEEP}
    out["check_results"] = [
        {"name": c["name"], "passed": c["passed"]} for c in (rec.get("check_results") or [])
    ]
    out["critique"] = (rec.get("critique") or "")[:220]
    out["error_message"] = (rec.get("error_message") or "")[:120] or None
    out["judges"] = [d.get("judge") for d in (rec.get("judge_details") or []) if d.get("judge")]
    return out


def merge_records(
    base: list[dict[str, Any]], new: list[dict[str, Any]], keep_models: set[str] | None
) -> list[dict[str, Any]]:
    """New results replace the stored ones per (model, prompt); drop retired models."""
    refreshed = {(model_key(r), r.get("prompt_id")) for r in new}
    merged = [r for r in base if (model_key(r), r.get("prompt_id")) not in refreshed] + new
    if keep_models is not None:
        merged = [r for r in merged if model_key(r) in keep_models]
    return merged


def build_data(
    records: list[dict[str, Any]], history: list[dict[str, Any]], run_id: str, date: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompts = {p.id: p for p in get_all_prompts()}
    cats = {pid: p.category for pid, p in prompts.items()}
    rows = leaderboard(records, cats)
    categories = sorted({c for r in rows for c in r["categories"]})

    # per-model per-prompt detail (mean over replications, details from the best-scoring run)
    detail: dict[str, dict[str, Any]] = {}
    for rec in records:
        pid = rec.get("prompt_id")
        if pid not in prompts:
            continue
        score = rec.get("final_score")
        if score is None:
            score = 0.0 if not rec.get("success") else None
        if score is None:
            continue
        cell = detail.setdefault(model_key(rec), {}).setdefault(pid, {"s": [], "rec": None})
        cell["s"].append(float(score))
        if cell["rec"] is None or float(score) >= cell["rec"][0]:
            cell["rec"] = (float(score), rec)
    compact: dict[str, dict[str, Any]] = {}
    for model, cells in detail.items():
        compact[model] = {}
        for pid, cell in cells.items():
            rec = cell["rec"][1]
            failed = [c["name"] for c in (rec.get("check_results") or []) if not c.get("passed")]
            compact[model][pid] = {
                "s": round(sum(cell["s"]) / len(cell["s"]), 1),
                "a": None if rec.get("auto_score") is None else round(100 * rec["auto_score"]),
                "j": None if rec.get("judge_score") is None else round(100 * rec["judge_score"]),
                "f": failed,
                "c": (rec.get("critique") or "")[:220],
                "l": None
                if rec.get("latency_seconds") is None
                else round(rec["latency_seconds"], 1),
                "e": (rec.get("error_message") or "")[:120] if not rec.get("success") else "",
            }

    prompt_stats = {}
    for pid, p in prompts.items():
        vals = [compact[m][pid]["s"] for m in compact if pid in compact[m]]
        prompt_stats[pid] = {
            "category": p.category,
            "description": p.description,
            "difficulty": p.difficulty,
            "mean": round(sum(vals) / len(vals), 1) if vals else None,
            "solved": round(100 * sum(v >= 90 for v in vals) / len(vals)) if vals else None,
            "n": len(vals),
        }

    previous = [h for h in history if h["date"] != date]
    seen_before = {m for h in previous for m in h["scores"]}
    for row in rows:
        row["display"] = display_name(row["model"])
        row["family"] = family(row["model"])
        row["new"] = bool(previous) and row["model"] not in seen_before
        last = previous[-1]["scores"].get(row["model"]) if previous else None
        row["delta"] = None if last is None else round(row["score"] - last["score"], 1)

    entry = {
        "date": date,
        "run": run_id,
        "scores": {r["model"]: {"score": r["score"], "categories": r["categories"]} for r in rows},
    }
    history = (previous + [entry])[-HISTORY_LIMIT:]

    judge_ok = [r for r in records if r.get("judge_score") is not None]
    judges = sorted(
        {
            j if isinstance(j, str) else j.get("judge", "")
            for r in judge_ok
            for j in (r.get("judges") or r.get("judge_details") or [])
        }
    )
    data = {
        "generated": date,
        "run": run_id,
        "judges": [j.removeprefix("blablador:") for j in judges if j],
        "n_prompts": len(prompts),
        "n_runs": len(records),
        "n_judged": len(judge_ok),
        "categories": categories,
        "rows": rows,
        "prompts": prompt_stats,
        "detail": compact,
        "history": [
            {"date": h["date"], "scores": {m: v["score"] for m, v in h["scores"].items()}}
            for h in history
        ],
    }
    return data, history


# --------------------------------------------------------------------- markdown / svg
def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))


def render_markdown(data: dict[str, Any], top: int = 20) -> str:
    lines = [
        f"_Updated {data['generated']} · {len(data['rows'])} models · {data['n_prompts']} prompts · "
        f"judges: {', '.join(data['judges']) or 'checks only'}_",
        "",
        "| # | Model | Score | "
        + " | ".join(c.capitalize() for c in data["categories"])
        + " | Reliab. | Latency |",
        "|--:|:--|--:|" + "--:|" * len(data["categories"]) + "--:|--:|",
    ]
    for r in data["rows"][:top]:
        cells = " | ".join(
            f"{r['categories'][c]:.0f}" if c in r["categories"] else "–"
            for c in data["categories"]
        )
        badge = " 🆕" if r["new"] else ""
        delta = ""
        if r["delta"]:
            delta = f" ({'▲' if r['delta'] > 0 else '▼'}{abs(r['delta']):.1f})"
        lat = f"{r['median_latency_s']:.1f}s" if r["median_latency_s"] is not None else "–"
        lines.append(
            f"| {medal(r['rank'])} | `{r['display']}`{badge} | **{r['score']:.1f}**{delta} | {cells} | "
            f"{r['reliability']:.0f}% | {lat} |"
        )
    if len(data["rows"]) > top:
        lines.append(
            f"\n_Showing top {top} of {len(data['rows'])}. See the full dashboard for all models._"
        )
    return "\n".join(lines) + "\n"


def render_svg(data: dict[str, Any], top: int = 15) -> str:
    rows = data["rows"][:top]
    w, row_h, left, right, head = 860, 28, 250, 70, 56
    h = head + row_h * len(rows) + 30
    bar_w = w - left - right
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        'role="img" aria-label="Benchmark leaderboard: overall score by model">',
        "<style>"
        ".bg{fill:#fff}.t{fill:#1f2328;font:600 15px system-ui,sans-serif}"
        ".s{fill:#59636e;font:12px system-ui,sans-serif}.l{fill:#1f2328;font:13px system-ui,sans-serif}"
        ".v{fill:#1f2328;font:600 13px system-ui,sans-serif}.b{fill:#2f6fdb}.n{fill:#d97706}"
        ".g{stroke:#d1d9e0;stroke-width:1}.ci{stroke:#1f2328;stroke-opacity:.55;stroke-width:1.5}"
        "@media (prefers-color-scheme:dark){.bg{fill:#0d1117}.t,.l,.v{fill:#e6edf3}.s{fill:#9198a1}"
        ".b{fill:#5b9cf6}.n{fill:#f0a43a}.g{stroke:#30363d}.ci{stroke:#e6edf3}}"
        "</style>",
        f'<rect class="bg" width="{w}" height="{h}" rx="8"/>',
        f'<text class="t" x="16" y="26">Overall benchmark score (0–100)</text>',
        f'<text class="s" x="16" y="44">{html.escape(data["generated"])} · {data["n_prompts"]} prompts · '
        f"checks + LLM-judge panel · bars show mean ± 95% CI</text>",
    ]
    for tick in (0, 25, 50, 75, 100):
        x = left + bar_w * tick / 100
        out.append(f'<line class="g" x1="{x:.1f}" y1="{head - 4}" x2="{x:.1f}" y2="{h - 26}"/>')
        out.append(f'<text class="s" x="{x:.1f}" y="{h - 10}" text-anchor="middle">{tick}</text>')
    for i, r in enumerate(rows):
        y = head + i * row_h
        bw = bar_w * r["score"] / 100
        cls = "n" if r["new"] else "b"
        label = html.escape(r["display"][:34])
        out.append(
            f'<text class="l" x="{left - 8}" y="{y + 17}" text-anchor="end">{r["rank"]}. {label}</text>'
        )
        out.append(
            f'<rect class="{cls}" x="{left}" y="{y + 4}" width="{bw:.1f}" height="{row_h - 10}" rx="3"/>'
        )
        lo = max(0, r["score"] - r["ci95"]) / 100 * bar_w
        hi = min(100, r["score"] + r["ci95"]) / 100 * bar_w
        if r["ci95"] > 0:
            ym = y + row_h / 2 - 1
            out.append(
                f'<line class="ci" x1="{left + lo:.1f}" y1="{ym}" x2="{left + hi:.1f}" y2="{ym}"/>'
            )
        out.append(f'<text class="v" x="{left + bw + 8:.1f}" y="{y + 17}">{r["score"]:.1f}</text>')
    out.append(f'<rect class="n" x="{w - 160}" y="16" width="10" height="10" rx="2"/>')
    out.append(f'<text class="s" x="{w - 144}" y="25">new this run</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def update_readme(readme: Path, markdown: str, svg_rel: str, dash_url: str) -> bool:
    text = readme.read_text(encoding="utf-8")
    block = (
        f"{README_START}\n"
        f"[![Benchmark leaderboard]({svg_rel})]({dash_url})\n\n"
        f"{markdown}\n"
        f"[Open the interactive dashboard →]({dash_url})\n"
        f"{README_END}"
    )
    if README_START in text and README_END in text:
        new = re.sub(
            re.escape(README_START) + r".*?" + re.escape(README_END),
            lambda _m: block,
            text,
            flags=re.DOTALL,
        )
    else:
        heading = "## Model Leaderboard\n\n"
        anchor = "## System-One (Laya) CLI"
        new = (
            text.replace(anchor, heading + block + "\n\n" + anchor, 1)
            if anchor in text
            else (text + "\n" + heading + block + "\n")
        )
    if new != text:
        readme.write_text(new, encoding="utf-8")
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("--history", type=Path, default=Path("reports/history.json"))
    ap.add_argument(
        "--store",
        type=Path,
        default=Path("reports/results_latest.json"),
        help="rolling compact results; new results are merged into it per (model, prompt)",
    )
    ap.add_argument("--status", type=Path, default=Path("models_status.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/benchmark"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d%H%M"))
    ap.add_argument("--date", default=datetime.now(UTC).strftime("%Y-%m-%d"))
    ap.add_argument(
        "--dashboard-url", default="https://jonasheinickebio.github.io/HeLLMholtz/benchmark/"
    )
    args = ap.parse_args(argv)

    records = load_records(args.results)
    if not records:
        print("no records", file=sys.stderr)
        return 1
    keep_models = None
    if args.status.exists():
        import yaml

        sys.path.insert(0, str(Path(__file__).parent))
        from select_models import chat_models

        keep_models = chat_models(yaml.safe_load(args.status.read_text()) or {})
    base = json.loads(args.store.read_text()) if args.store.exists() else []
    records = merge_records(base, [compact_record(r) for r in records], keep_models)
    args.store.parent.mkdir(parents=True, exist_ok=True)
    args.store.write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    history = json.loads(args.history.read_text()) if args.history.exists() else []
    data, history = build_data(records, history, args.run_id, args.date)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    (args.out_dir / "index.html").write_text(template.replace("__DATA__", blob), encoding="utf-8")
    (args.out_dir / "data.json").write_text(
        json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / ".nojekyll").write_text("")
    args.history.write_text(json.dumps(history, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown = render_markdown(data)
    (args.reports_dir / "leaderboard.md").write_text(
        "# Benchmark leaderboard\n\n" + markdown, encoding="utf-8"
    )
    (args.reports_dir / "leaderboard.svg").write_text(render_svg(data), encoding="utf-8")
    changed = args.readme.exists() and update_readme(
        args.readme, markdown, "reports/leaderboard.svg", args.dashboard_url
    )
    print(
        f"dashboard: {len(data['rows'])} models, {data['n_judged']} judged, "
        f"history={len(history)} runs, README {'updated' if changed else 'unchanged'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
