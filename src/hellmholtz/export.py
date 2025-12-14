import glob
import os
from typing import Any, Literal

from hellmholtz.reporting import load_results


def select_best_model(
    results_path: str, criterion: Literal["latency", "success_rate"] = "latency"
) -> dict[str, Any]:
    """Select the best model based on benchmark results."""

    results = load_results(results_path)
    if not results:
        raise ValueError("No results found in file.")

    models = list(set(r.model for r in results))
    best_model = None
    best_score = float("inf") if criterion == "latency" else -1.0

    for model in models:
        model_results = [r for r in results if r.model == model]
        if not model_results:
            continue

        if criterion == "latency":
            # Lower is better
            # Filter for successful runs only for latency calculation
            success_runs = [r for r in model_results if r.success]
            if not success_runs:
                continue
            score = sum(r.latency_seconds for r in success_runs) / len(success_runs)
            if score < best_score:
                best_score = score
                best_model = model
        else:
            # Higher is better
            score = len([r for r in model_results if r.success]) / len(model_results)
            if score > best_score:
                best_score = score
                best_model = model

    if best_model is None:
        # Fallback if no model met criteria (e.g. all failed)
        best_model = models[0]

    return {
        "model": best_model,
        # Could add other derived config here
    }


def get_default_model_config(results_dir: str = "results") -> dict[str, Any]:
    """Get the best model config from the latest benchmark."""

    # Find latest benchmark file
    files = glob.glob(os.path.join(results_dir, "benchmark_*.json"))
    if not files:
        # Fallback default
        return {"model": "openai:gpt-4o"}

    latest_file = max(files, key=os.path.getctime)

    try:
        return select_best_model(latest_file)
    except Exception:
        return {"model": "openai:gpt-4o"}
