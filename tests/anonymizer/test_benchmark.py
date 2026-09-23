"""Unit tests for benchmarking, accuracy scoring and report writers."""

import json

import pytest

from hellmholtz.anonymizer.anonymizer import Anonymizer
from hellmholtz.anonymizer.benchmark import (
    DEFAULT_REPORT_DIR,
    AccuracyReport,
    BenchmarkRun,
    compare_prompts,
    evaluate_accuracy,
    is_consistent,
    run_benchmark,
    write_reports,
)
from hellmholtz.anonymizer.models import AnonymizationResult, AnonymizedEntity
from tests.anonymizer.fake_model import FakeAnonymizerModel, make_mapping


class TestBenchmarkRun:
    def test_percentiles_and_throughput(self):
        run = BenchmarkRun(
            label="x", total_chars=1000, n_iterations=4, latencies_ms=[10.0, 20.0, 30.0, 40.0]
        )
        assert run.mean_ms == 25.0
        assert run.p50_ms == 25.0
        assert run.p95_ms == 38.5
        assert run.throughput_chars_per_s == pytest.approx(40000.0)

    def test_empty_run_defaults(self):
        run = BenchmarkRun(label="x", total_chars=0, n_iterations=0)
        assert run.mean_ms == 0.0
        assert run.p50_ms == 0.0
        assert run.p95_ms == 0.0
        assert run.throughput_chars_per_s == 0.0
        assert run.pass_rate == 0.0


class TestIsConsistent:
    def test_consistent(self):
        result = AnonymizationResult(
            text="x",
            entities=[
                AnonymizedEntity("PERSON", "John Smith", "A"),
                AnonymizedEntity("PERSON", "john smith", "A"),
            ],
        )
        assert is_consistent(result) is True

    def test_inconsistent(self):
        result = AnonymizationResult(
            text="x",
            entities=[
                AnonymizedEntity("PERSON", "John Smith", "A"),
                AnonymizedEntity("PERSON", "john smith", "B"),
            ],
        )
        assert is_consistent(result) is False


class TestEvaluateAccuracy:
    def test_perfect_detection(self, fake_model, fixture_texts, ground_truth):
        text = fixture_texts["sample_short_en"]
        result = Anonymizer(chat_fn=fake_model).anonymize(text)
        acc = evaluate_accuracy(result, ground_truth["sample_short_en"])
        assert acc.expected_entities == 9
        assert acc.recall == 1.0
        assert acc.leak_rate == 0.0
        assert acc.leaked_entities == []
        assert acc.consistent is True

    def test_missed_value_counts_as_leak(self, fixture_texts, ground_truth):
        gt = ground_truth["sample_short_en"]
        values = [g["value"] for g in gt]
        # The fake model does not know "Berlin" -> it survives the run.
        fake = FakeAnonymizerModel(
            make_mapping([v for v in values if v != "Berlin"]),
            types={g["value"]: g["type"] for g in gt},
        )
        result = Anonymizer(chat_fn=fake).anonymize(fixture_texts["sample_short_en"])
        acc = evaluate_accuracy(result, gt)
        assert acc.recall == pytest.approx(8 / 9)
        assert acc.leak_rate == pytest.approx(1 / 9)
        assert acc.leaked_entities == ["Berlin"]

    def test_empty_ground_truth(self):
        acc = evaluate_accuracy(AnonymizationResult(text="x"), [])
        assert acc.recall == 1.0
        assert acc.leak_rate == 0.0

    def test_report_properties_degenerate(self):
        report = AccuracyReport(expected_entities=0, detected_entities=0, leaked_entities=[], consistent=True)
        assert report.recall == 1.0
        assert report.leak_rate == 0.0


class TestRunBenchmark:
    def test_aggregates_iterations(self, fake_model, fixture_texts):
        runs = run_benchmark(
            {"en": fixture_texts["sample_short_en"]},
            iterations=2,
            chat_fn=fake_model,
        )
        run = runs["en"]
        assert run.n_iterations == 2
        assert len(run.latencies_ms) == 2
        assert run.entity_counts == [9, 9]
        assert run.validation_ok == [True, True]
        assert run.consistent == [True, True]
        assert run.model_reported_ms == [42.0, 42.0]
        assert run.pass_rate == 1.0
        assert run.total_chars == len(fixture_texts["sample_short_en"])

    def test_collect_results(self, fake_model, fixture_texts):
        collected: dict = {}
        run_benchmark(
            {"en": fixture_texts["sample_short_en"]},
            iterations=1,
            collect_results=collected,
            chat_fn=fake_model,
        )
        assert set(collected) == {"en"}
        assert isinstance(collected["en"], AnonymizationResult)


class TestComparePrompts:
    def test_variants_evaluated_side_by_side(self, fake_model, fixture_texts, ground_truth):
        texts = {"en": fixture_texts["sample_short_en"]}
        comp = compare_prompts(
            texts,
            ["default", "strict"],
            iterations=1,
            ground_truth={"en": ground_truth["sample_short_en"]},
            chat_fn=fake_model,
        )
        assert set(comp) == {"default", "strict"}
        for name in ("default", "strict"):
            m = comp[name]
            assert m["n_runs"] == 1
            assert m["footer"] is True
            assert m["mean_recall"] == 1.0
            assert m["mean_leak_rate"] == 0.0
            assert m["validation_pass_rate"] == 1.0
            assert m["per_text"]["en"]["recall"] == 1.0

    def test_accuracy_columns_absent_without_ground_truth(self, fake_model, fixture_texts):
        comp = compare_prompts({"en": fixture_texts["sample_short_en"]}, ["default"], chat_fn=fake_model)
        assert comp["default"]["mean_recall"] is None
        assert "recall" not in comp["default"]["per_text"]["en"]

    def test_unknown_variant_fails_fast(self, fixture_texts):
        with pytest.raises(KeyError):
            compare_prompts({"en": fixture_texts["sample_short_en"]}, ["bogus"])


class TestWriteReports:
    def test_writes_markdown_and_json(self, fake_model, fixture_texts, ground_truth, tmp_path):
        texts = {"en": fixture_texts["sample_short_en"]}
        comp = compare_prompts(
            texts,
            ["default", "footer_off"],
            iterations=1,
            ground_truth={"en": ground_truth["sample_short_en"]},
            chat_fn=fake_model,
        )
        out_dir = tmp_path / "reports"
        md_path, json_path = write_reports(comp, out_dir=out_dir)

        assert md_path.exists() and json_path.exists()
        assert md_path.parent == out_dir

        md = md_path.read_text(encoding="utf-8")
        assert "# Anonymizer prompt comparison" in md
        assert "| default " in md
        assert "| footer_off " in md
        assert "## footer_off" in md

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert set(data) == {"default", "footer_off"}
        assert data["footer_off"]["footer"] is False
        assert data["default"]["footer"] is True

    def test_default_report_dir_constant(self):
        assert DEFAULT_REPORT_DIR == "reports/anonymizer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
