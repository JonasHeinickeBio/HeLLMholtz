#!/usr/bin/env python3
"""Generate the model and performance pages of the docs site and the GitHub wiki.

Reads the dashboard data (``docs/benchmark/data.json``, written by
``build_dashboard.py``) plus ``models_status.yaml`` and writes

* ``docs/performance.md`` and the generated block of ``docs/models.md`` (MkDocs site),
* ``wiki/Home.md``, ``Models.md``, ``Performance.md`` and ``_Sidebar.md`` (GitHub wiki).

Everything between the ``MODELS`` markers and all of ``performance.md`` / ``wiki/`` is
generated: edit this script, not the output.

Usage: python scripts/build_docs.py
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_URL = "https://jonasheinickebio.github.io/HeLLMholtz/benchmark/"
DOCS_URL = "https://jonasheinickebio.github.io/HeLLMholtz/"
REPO_URL = "https://github.com/JonasHeinickeBio/HeLLMholtz"
START, END = "<!-- MODELS:START -->", "<!-- MODELS:END -->"
MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def load_json(path: Path) -> dict[str, Any]:
    import json

    return dict(json.loads(path.read_text(encoding="utf-8")))


def num(value: Any, digits: int = 1, suffix: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{suffix}"


def speed_cell(row: dict[str, Any]) -> str:
    ttft, tps = row.get("ttft_s"), row.get("decode_tps")
    if ttft is None and tps is None:
        return "–"
    return f"{num(ttft, 2, ' s')} / {num(tps, 0, ' tok/s')}"


def cap(name: str) -> str:
    return {"longcontext": "Long context"}.get(name, name[:1].upper() + name[1:])


def ctx(tokens: int | None) -> str:
    if not tokens:
        return "–"
    return f"{tokens // 1_000_000}M" if tokens >= 1_000_000 else f"{round(tokens / 1024)}k"


def name_cell(row: dict[str, Any]) -> str:
    return f"`{row['display']}`"


def table(headers: list[str], rows: list[list[str]], align: str = "") -> str:
    aligns = align or "l" + "r" * (len(headers) - 1)
    sep = ["--:" if a == "r" else ":--" for a in aligns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- sections


def headline(data: dict[str, Any]) -> str:
    rows = data["rows"]
    top = rows[0]
    near = [r for r in rows if r["score"] >= top["score"] - 5]
    with_speed = [r for r in near if r.get("speed_score") is not None]
    fast = max(with_speed, key=lambda r: r["speed_score"]) if with_speed else None
    best = max(
        rows, key=lambda r: r["score"] / 100 * 0.7 + (r.get("speed_score") or 0) / 100 * 0.3
    )
    parts = [
        f"_Updated {data['generated']} · {len(rows)} models · {data['n_prompts']} prompts · "
        f"judges: {', '.join(data.get('judges', [])) or 'checks only'}_",
        "",
        f"- **Best overall:** {name_cell(top)} with **{num(top['score'])}** points",
    ]
    if fast:
        parts.append(
            f"- **Fastest of the top group (within 5 points of #1):** {name_cell(fast)} - "
            f"{num(fast.get('ttft_s'), 2, ' s')} to first token, "
            f"{num(fast.get('decode_tps'), 0, ' tok/s')}"
        )
    parts.append(
        f"- **Best quality/speed balance (70 % quality, 30 % speed):** {name_cell(best)} - "
        f"{num(best['score'])} points, speed {num(best.get('speed_score'), 0)}"
    )
    return "\n".join(parts)


def leaderboard(data: dict[str, Any], top: int = 15) -> str:
    rows = []
    for r in data["rows"][:top]:
        rows.append(
            [
                MEDALS.get(r["rank"], str(r["rank"])),
                name_cell(r),
                r["family"],
                f"**{num(r['score'])}**",
                num(r.get("speed_score"), 0),
                num(r.get("ttft_s"), 2),
                num(r.get("decode_tps"), 0),
                num(r["reliability"], 0, "%"),
                str(r.get("prompts", "")),
            ]
        )
    out = table(
        ["#", "Model", "Family", "Score", "Speed", "TTFT s", "Tok/s", "Reliab.", "Prompts"],
        rows,
        "ccl" + "r" * 6,
    )
    if len(data["rows"]) > top:
        out += f"\n\n_Top {top} of {len(data['rows'])}; the full ranking is in the dashboard._"
    return out


def best_per_category(data: dict[str, Any]) -> str:
    rows = []
    for cat in data["categories"]:
        ranked = sorted(
            (r for r in data["rows"] if r["categories"].get(cat) is not None),
            key=lambda r: (-r["categories"][cat], r["rank"]),
        )[:3]
        rows.append(
            [cap(cat)]
            + [f"`{r['display']}` ({r['categories'][cat]:.0f})" for r in ranked]
            + [""] * (3 - len(ranked))
        )
    return table(["Category", "1st", "2nd", "3rd"], rows, "llll")


def speed_ranking(data: dict[str, Any], top: int = 12) -> str:
    fast = sorted(
        (r for r in data["rows"] if r.get("speed_score") is not None),
        key=lambda r: -r["speed_score"],
    )[:top]
    if not fast:
        return "_No speed probe data yet._"
    rows = [
        [
            str(i),
            name_cell(r),
            num(r["speed_score"], 0),
            num(r.get("ttft_s"), 2),
            num(r.get("decode_tps"), 0),
            num(r.get("median_latency_s"), 1),
            num(r["score"]),
        ]
        for i, r in enumerate(fast, 1)
    ]
    return table(
        ["#", "Model", "Speed", "TTFT s", "Tok/s", "Latency s", "Score"],
        rows,
        "cl" + "r" * 5,
    )


def picks(data: dict[str, Any]) -> str:
    """Which model for which job, from quality and speed together."""
    rows = data["rows"]

    def pick(pool: list[dict[str, Any]], key: Any) -> dict[str, Any] | None:
        return max(pool, key=key) if pool else None

    fast_pool = [r for r in rows if (r.get("decode_tps") or 0) >= 150]
    mid_pool = [r for r in rows if 60 <= (r.get("decode_tps") or 0) < 150]
    cat = data["categories"]
    items: list[tuple[str, dict[str, Any] | None]] = [
        ("Highest quality", rows[0] if rows else None),
        ("Best fast model (≥ 150 tok/s)", pick(fast_pool, lambda r: r["score"])),
        ("Best mid-speed model (60-150 tok/s)", pick(mid_pool, lambda r: r["score"])),
    ]
    for c, label in (
        ("coding", "Coding"),
        ("math", "Math"),
        ("reasoning", "Reasoning"),
        ("medical", "Medical"),
        ("multilingual", "Multilingual"),
        ("longcontext", "Long context"),
    ):
        if c in cat:
            tested = [r for r in rows if r["categories"].get(c) is not None]
            items.append((label, pick(tested, lambda r, c=c: r["categories"][c])))
    out = []
    for label, r in items:
        if r is None:
            continue
        cat_note = ""
        if label in {"Coding", "Math", "Reasoning", "Medical", "Multilingual", "Long context"}:
            key = {"Long context": "longcontext"}.get(label, label.lower())
            cat_note = f", {label.lower()} {r['categories'].get(key, 0):.0f}"
        out.append(
            [
                label,
                name_cell(r),
                f"{num(r['score'])}{cat_note}",
                speed_cell(r),
            ]
        )
    return table(["Use case", "Model", "Score", "TTFT / throughput"], out, "lllr")


def reasoning_behaviour(data: dict[str, Any]) -> str:
    rows = [
        r
        for r in data["rows"]
        if r.get("reasoning_share") is not None or (r.get("truncated_pct") or 0) > 0
    ]
    if not rows:
        return "_No thinking or cut-off data in this run._"
    rows.sort(key=lambda r: (-(r.get("reasoning_share") or 0), -(r.get("truncated_pct") or 0)))
    if not any(r.get("reasoning_share") is not None for r in rows):
        return (
            "_Thinking-token counts are not in this run yet (they are recorded from the next "
            "benchmark on); models that ran into the token limit:_\n\n"
            + table(
                ["Model", "Hit token limit", "Reasoning score"],
                [
                    [
                        name_cell(r),
                        num(r.get("truncated_pct"), 0, "%"),
                        num(r["categories"].get("reasoning"), 0),
                    ]
                    for r in rows[:12]
                ],
                "lrr",
            )
        )
    return table(
        ["Model", "Thinking share of output", "Hit token limit", "Reasoning score"],
        [
            [
                name_cell(r),
                num(r.get("reasoning_share"), 0, "%"),
                num(r.get("truncated_pct"), 0, "%"),
                num(r["categories"].get("reasoning"), 0),
            ]
            for r in rows[:12]
        ],
        "lrrr",
    )


def movers(data: dict[str, Any]) -> str:
    changed = [r for r in data["rows"] if r.get("new") or r.get("delta")]
    if not changed:
        return "_No new models or score changes since the previous run._"
    changed.sort(key=lambda r: -abs(r.get("delta") or 0))
    items = []
    for r in changed[:10]:
        tag = "🆕 new" if r.get("new") else f"{r['delta']:+.1f}"
        items.append(f"- {name_cell(r)} - {tag}, score {num(r['score'])} (#{r['rank']})")
    return "\n".join(items)


def catalog(data: dict[str, Any], status: dict[str, Any]) -> str:
    by_model = {r["model"]: r for r in data["rows"]}
    rows: list[tuple[int, list[str]]] = []
    for name, info in status.items():
        r = by_model.get(name)
        listed = bool(info.get("available"))
        state = "✅ listed" if listed else "⛔ not listed"
        rows.append(
            (
                r["rank"] if r else 10_000,
                [
                    f"`{r['display'] if r else name}`",
                    r["family"] if r else info.get("category", "–"),
                    ctx(info.get("tokens")),
                    state,
                    str(r["rank"]) if r else "–",
                    num(r["score"]) if r else "–",
                    num(r.get("speed_score"), 0) if r else "–",
                    (info.get("description") or "").replace("|", "/")[:70],
                ],
            )
        )
    rows.sort(key=lambda x: (x[0], x[1][0]))
    return table(
        ["Model", "Family", "Context", "API", "Rank", "Score", "Speed", "Description"],
        [r for _, r in rows],
        "llrcrrrl",
    )


def availability(status: dict[str, Any], checked: str | None = None) -> str:
    listed = sum(1 for v in status.values() if v.get("available"))
    when = f"status of {checked}" if checked else "status at the last registry update"
    return (
        f"**{listed} of {len(status)}** known models were listed by the API "
        f"({when}). Models marked ⛔ stay in the registry so they come back "
        f"automatically when the API lists them again."
    )


# ---------------------------------------------------------------- pages

def how(data: dict[str, Any], bench_link: str) -> str:
    only = data.get("n_checks_only")
    mixed = (
        f"{only} of {data['n_prompts']} prompts are scored by their deterministic checks alone; "
        "the others blend the checks (60 %) with a blind LLM-judge panel (40 %). "
        if only is not None
        else "Prompts blend deterministic checks with a blind LLM-judge panel where a rubric "
        "exists. "
    )
    return (
        "Scores are 0-100 per prompt (exact answers, executed unit tests and format rules). "
        f"{mixed}Failed, truncated and empty runs count as 0. Speed comes from a separate "
        "streamed probe per model (time to first token and decode throughput) and never "
        "changes the quality score; a model without a successful probe falls back to the "
        "throughput of its quality runs, so its speed rests on end-to-end latency. "
        f"Details: [Benchmarking]({bench_link}). The interactive version with filters, "
        f"charts and per-prompt drill-down is the [live dashboard]({DASHBOARD_URL})."
    )


def performance_page(data: dict[str, Any], bench_link: str, svg: str | None) -> str:
    parts = [
        "# Model performance",
        "",
        headline(data),
        "",
        how(data, bench_link),
        "",
    ]
    if svg:
        parts += [f"![Leaderboard]({svg})", ""]
    parts += [
        "## Which model for which job",
        "",
        picks(data),
        "",
        "## Leaderboard",
        "",
        leaderboard(data),
        "",
        "## Best per category",
        "",
        best_per_category(data),
        "",
        "## Speed ranking",
        "",
        "Speed score = mean of throughput (10 tok/s = 0, 200 tok/s = 100) and time to first "
        "token (10 s = 0, 0.3 s = 100), both log-scaled.",
        "",
        speed_ranking(data),
        "",
        "## Thinking behaviour",
        "",
        "Share of output tokens spent on reasoning and how often a model ran into the token "
        "limit before finishing.",
        "",
        reasoning_behaviour(data),
        "",
        "## What changed",
        "",
        movers(data),
        "",
    ]
    return "\n".join(parts)


def models_block(data: dict[str, Any], status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "### Model list",
            "",
            availability(status, data.get("status_checked")),
            "",
            f"_Scores from the benchmark of {data['generated']} "
            f"({data['n_prompts']} prompts); sorted by rank, unbenchmarked models last._",
            "",
            catalog(data, status),
            "",
            "> Generated by `scripts/build_docs.py` from `models_status.yaml` and the latest "
            "benchmark; do not edit by hand.",
        ]
    )


def update_models_page(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    wrapped = f"{START}\n{block}\n{END}"
    if START in text and END in text:
        text = re.sub(
            f"{re.escape(START)}.*?{re.escape(END)}", lambda _: wrapped, text, flags=re.S
        )
    else:
        # First run: replace the hand-written list (from its heading to the next "## ").
        m = re.search(r"### Model List.*?(?=\n## )", text, flags=re.S)
        if not m:
            raise SystemExit("models.md: no '### Model List' section and no markers found")
        text = text[: m.start()] + wrapped + "\n" + text[m.end() :]
    path.write_text(text, encoding="utf-8")


def wiki_pages(data: dict[str, Any], status: dict[str, Any]) -> dict[str, str]:
    perf = performance_page(data, f"{DOCS_URL}benchmarking/", None)
    top = data["rows"][:5]
    home = "\n".join(
        [
            "# HeLLMholtz",
            "",
            "Unified LLM access, benchmarking and reporting for the Helmholtz Blablador API.",
            "",
            "## Current state",
            "",
            headline(data),
            "",
            "### Top 5",
            "",
            table(
                ["#", "Model", "Score", "Speed"],
                [
                    [
                        MEDALS.get(r["rank"], str(r["rank"])),
                        name_cell(r),
                        f"**{num(r['score'])}**",
                        num(r.get("speed_score"), 0),
                    ]
                    for r in top
                ],
                "cl" + "rr",
            ),
            "",
            "## Pages",
            "",
            "- [[Performance]] - leaderboard, best model per job, speed ranking",
            "- [[Models]] - every known model with context size, availability and score",
            f"- [Live dashboard]({DASHBOARD_URL}) - interactive charts and drill-down",
            f"- [Documentation]({DOCS_URL}) - installation, usage, configuration",
            f"- [Source]({REPO_URL})",
            "",
            "_This wiki is generated from the benchmark data by `scripts/build_docs.py`; "
            "changes made here are overwritten._",
            "",
        ]
    )
    models = "\n".join(
        [
            "# Models",
            "",
            availability(status, data.get("status_checked")),
            "",
            f"_Scores from the benchmark of {data['generated']}._",
            "",
            catalog(data, status),
            "",
        ]
    )
    sidebar = "\n".join(
        [
            "**HeLLMholtz**",
            "",
            "- [[Home]]",
            "- [[Performance]]",
            "- [[Models]]",
            f"- [Dashboard]({DASHBOARD_URL})",
            f"- [Docs]({DOCS_URL})",
            "",
        ]
    )
    return {"Home.md": home, "Performance.md": perf, "Models.md": models, "_Sidebar.md": sidebar}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "docs/benchmark/data.json")
    ap.add_argument("--status", type=Path, default=ROOT / "models_status.yaml")
    ap.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    ap.add_argument("--wiki-dir", type=Path, default=ROOT / "wiki")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"{args.data} missing: run scripts/build_dashboard.py first", file=sys.stderr)
        return 1
    data = load_json(args.data)
    status_doc = yaml.safe_load(args.status.read_text(encoding="utf-8"))
    status = status_doc["models"]
    checked = status_doc.get("last_checked")
    if isinstance(checked, (int, float)):
        data["status_checked"] = datetime.fromtimestamp(checked, UTC).strftime("%Y-%m-%d")

    svg_src = ROOT / "reports/leaderboard.svg"
    svg_ref = None
    if svg_src.exists():
        (args.docs_dir / "assets").mkdir(exist_ok=True)
        shutil.copyfile(svg_src, args.docs_dir / "assets/leaderboard.svg")
        svg_ref = "assets/leaderboard.svg"

    (args.docs_dir / "performance.md").write_text(
        performance_page(data, "benchmarking.md", svg_ref), encoding="utf-8"
    )
    update_models_page(args.docs_dir / "models.md", models_block(data, status))

    args.wiki_dir.mkdir(exist_ok=True)
    for name, text in wiki_pages(data, status).items():
        (args.wiki_dir / name).write_text(text, encoding="utf-8")
    print(
        f"docs: performance.md, models.md and {len(wiki_pages(data, status))} wiki pages written"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
