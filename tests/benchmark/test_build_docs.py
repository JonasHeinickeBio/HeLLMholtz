"""scripts/build_docs.py: generated model/performance pages and wiki."""

import importlib.util
import json
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_docs.py"
spec = importlib.util.spec_from_file_location("build_docs", SCRIPT)
build_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_docs)


def _row(rank, name, score, **kw):
    base = {
        "model": name,
        "display": name,
        "family": "Other",
        "rank": rank,
        "score": score,
        "categories": {"coding": score, "math": score - 5},
        "reliability": 100.0,
        "median_latency_s": 2.0,
        "ttft_s": 0.5,
        "decode_tps": 100.0,
        "speed_score": 80.0,
        "reasoning_share": None,
        "truncated_pct": 0.0,
        "new": False,
        "delta": None,
    }
    return {**base, **kw}


DATA = {
    "generated": "2026-11-05",
    "n_prompts": 58,
    "judges": ["j1"],
    "categories": ["coding", "math"],
    "rows": [
        _row(1, "big", 98.0),
        _row(2, "fast", 93.0, decode_tps=300.0, speed_score=99.0),
        _row(3, "gone", 50.0, ttft_s=None, decode_tps=None, speed_score=None, new=True),
    ],
}
STATUS = {
    "big": {"available": True, "tokens": 131072, "description": "Big | model"},
    "fast": {"available": True, "tokens": 32768},
    "gone": {"available": False, "tokens": 1048576},
    "untested": {"available": True, "tokens": 8192, "category": "base_model"},
}


def test_pages_contain_models_speed_and_picks():
    page = build_docs.performance_page(DATA, "benchmarking.md", None)
    assert "`big`" in page and "98.0" in page and "Which model for which job" in page
    assert "Best fast model" in page and "`fast`" in page
    assert "No thinking or cut-off data in this run" in page


def test_catalog_marks_availability_and_keeps_unbenchmarked_last():
    table = build_docs.catalog(DATA, STATUS)
    lines = [ln for ln in table.splitlines() if ln.startswith("| `")]
    assert [ln.split("`")[1] for ln in lines] == ["big", "fast", "gone", "untested"]
    assert "⛔ not listed" in lines[2] and "1M" in lines[2] and "Big / model" in lines[0]


def test_models_page_block_is_replaced_in_place(tmp_path):
    page = tmp_path / "models.md"
    page.write_text("# Models\n\n### Model List\n\nold table\n\n## Other Providers\n\nkeep me\n")
    build_docs.update_models_page(page, "NEW BLOCK")
    build_docs.update_models_page(page, "NEWER BLOCK")
    text = page.read_text()
    assert (
        text.count(build_docs.START) == 1 and "NEWER BLOCK" in text and "NEW BLOCK\n" not in text
    )
    assert "old table" not in text and "keep me" in text


def test_main_writes_docs_and_wiki(tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(DATA))
    status = tmp_path / "status.yaml"
    status.write_text(yaml.safe_dump({"models": STATUS}))
    docs, wiki = tmp_path / "docs", tmp_path / "wiki"
    docs.mkdir()
    (docs / "models.md").write_text("### Model List\n\nx\n\n## Other\n")
    import sys

    argv = sys.argv
    sys.argv = [
        "build_docs",
        "--data",
        str(data),
        "--status",
        str(status),
        "--docs-dir",
        str(docs),
        "--wiki-dir",
        str(wiki),
    ]
    try:
        assert build_docs.main() == 0
    finally:
        sys.argv = argv
    assert (docs / "performance.md").exists()
    assert {p.name for p in wiki.iterdir()} == {
        "Home.md",
        "Models.md",
        "Performance.md",
        "_Sidebar.md",
    }
    assert "[[Performance]]" in (wiki / "Home.md").read_text()
