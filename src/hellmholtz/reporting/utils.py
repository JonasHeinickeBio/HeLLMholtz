"""
Utility functions for benchmark data handling and export.
"""

import json

from hellmholtz.benchmark import BenchmarkResult


def export_to_csv(results: list[BenchmarkResult], filepath: str) -> None:
    """Export benchmark results to CSV format."""
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model",
            "prompt_id",
            "latency_seconds",
            "success",
            "timestamp",
            "input_tokens",
            "output_tokens",
            "error_message",
            "temperature",
            "max_tokens",
            "run_id",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "model": result.model,
                    "prompt_id": result.prompt_id,
                    "latency_seconds": result.latency_seconds,
                    "success": result.success,
                    "timestamp": result.timestamp,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "error_message": result.error_message,
                    "temperature": result.temperature,
                    "max_tokens": result.max_tokens,
                    "run_id": result.run_id,
                }
            )


def load_results(path: str) -> list[BenchmarkResult]:
    """Load benchmark results from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of results in {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected each result item to be an object in {path}")

    return [BenchmarkResult(**item) for item in data]
