from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

from hellmholtz.client import chat_raw


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model: str
    prompt_id: str
    latency_seconds: float
    success: bool
    timestamp: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_message: str | None = None


def run_benchmarks(
    models: Iterable[str],
    prompts: Iterable[str],
    *,
    system_prompt: str | None = None,
    repeat: int = 1,
    results_dir: str = "results",
) -> list[BenchmarkResult]:
    """Run benchmarks across multiple models and prompts."""

    results = []

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    for model in models:
        for i, prompt in enumerate(prompts):
            prompt_id = f"prompt_{i}"

            for _ in range(repeat):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                start_time = time.perf_counter()
                success = False
                error_msg = None
                in_tokens = None
                out_tokens = None

                try:
                    response = chat_raw(model=model, messages=messages)
                    success = True

                    # Try to extract usage
                    if hasattr(response, "usage") and response.usage:
                        in_tokens = response.usage.prompt_tokens
                        out_tokens = response.usage.completion_tokens

                except Exception as e:
                    error_msg = str(e)

                end_time = time.perf_counter()
                latency = end_time - start_time

                result = BenchmarkResult(
                    model=model,
                    prompt_id=prompt_id,
                    latency_seconds=latency,
                    success=success,
                    timestamp=timestamp,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    error_message=error_msg,
                )
                results.append(result)

    # Save results
    save_results(results, results_path, timestamp)

    return results


def run_throughput_benchmark(
    model: str, prompt: str = "Write a long story about a space adventure.", max_tokens: int = 100
) -> dict[str, Any]:
    """Run a throughput benchmark (tokens/sec)."""

    # Note: aisuite doesn't expose streaming token counts easily in a unified way yet
    # for all providers without streaming.
    # We will approximate by generating text and dividing by time.

    start_time = time.perf_counter()
    try:
        response = chat_raw(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens
        )
        end_time = time.perf_counter()

        latency = end_time - start_time

        # Try to get exact token counts
        if hasattr(response, "usage") and response.usage:
            out_tokens = response.usage.completion_tokens
        else:
            # Estimate: 4 chars per token
            content = response.choices[0].message.content
            out_tokens = len(content) / 4

        tokens_per_sec = out_tokens / latency if latency > 0 else 0

        return {
            "model": model,
            "latency": latency,
            "output_tokens": out_tokens,
            "tokens_per_sec": tokens_per_sec,
            "success": True,
        }

    except Exception as e:
        return {"model": model, "success": False, "error": str(e)}


def save_results(results: list[BenchmarkResult], directory: Path, timestamp: str) -> None:
    """Save benchmark results to JSON."""
    filename = f"benchmark_{timestamp.replace(':', '-')}.json"
    filepath = directory / filename

    data = [asdict(r) for r in results]

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
