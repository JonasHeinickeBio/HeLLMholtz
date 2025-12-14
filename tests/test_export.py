import json
from hellmholtz.export import select_best_model
from hellmholtz.benchmark import BenchmarkResult

def test_save_results(tmp_path: object) -> None:
    # Create dummy results
    results = [
        BenchmarkResult(
            model="fast-model",
            prompt_id="1",
            latency_seconds=0.1,
            success=True,
            timestamp="2024-01-01",
        ),
        BenchmarkResult(
            model="slow-model",
            prompt_id="1",
            latency_seconds=1.0,
            success=True,
            timestamp="2024-01-01",
        )
    ]

    # Save to file
    filepath = tmp_path / "results.json"
    with open(filepath, "w") as f:
        json.dump([vars(r) for r in results], f)

    # Test latency selection
    best = select_best_model(str(filepath), criterion="latency")
    assert best["model"] == "fast-model"
