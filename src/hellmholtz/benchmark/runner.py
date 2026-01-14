from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

from tqdm import tqdm

from hellmholtz.client import chat_raw
from hellmholtz.core.prompts import Prompt

logger = logging.getLogger(__name__)


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
    response_text: str | None = None
    rating: float | None = None
    critique: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    run_id: int | None = None  # For replication tracking


def run_benchmarks(  # noqa: C901
    models: Iterable[str],
    prompts: Iterable[Prompt],
    *,
    system_prompt: str | None = None,
    repeat: int = 1,
    results_dir: str = "results",
    max_concurrent: int = 1,
    temperatures: list[float] | None = None,
    max_tokens: int | None = None,
    replications: int = 1,  # Number of times to run each configuration
) -> list[BenchmarkResult]:
    """Run benchmarks across multiple models and prompts.

    Args:
        models: Model identifiers to benchmark
        prompts: Prompts to test
        system_prompt: Optional system prompt to prepend
        repeat: Number of times to repeat each model-prompt combination
        results_dir: Directory to save results
        max_concurrent: Maximum concurrent requests (for future parallelization)
        temperatures: List of temperature values to test (default: [0.1, 0.7, 1.0])
        max_tokens: Maximum tokens for response
        replications: Number of times to replicate each test for statistical significance
    """

    results = []
    model_list = list(models)
    prompt_list = list(prompts)

    # Set default temperatures if not provided
    if temperatures is None:
        temperatures = [0.1, 0.7, 1.0]

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    total_tests = len(model_list) * len(prompt_list) * len(temperatures) * replications

    logger.info(
        f"Starting benchmarks: {len(model_list)} models × {len(prompt_list)} "
        f"prompts × {len(temperatures)} temperatures × {replications} "
        f"replications = {total_tests} total tests"
    )

    # Track statistics
    stats: dict[str, Any] = {
        "total": total_tests,
        "completed": 0,
        "successful": 0,
        "failed": 0,
        "models_tested": set(),
        "categories_tested": set(),
        "temperatures_tested": set(temperatures),
    }

    with tqdm(
        total=total_tests,
        desc="Benchmarking",
        unit="test",
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    ) as pbar:
        for model_idx, model in enumerate(model_list):
            model_results = 0
            model_successful = 0

            logger.info(f"Testing model {model_idx + 1}/{len(model_list)}: {model}")

            for _prompt_idx, prompt in enumerate(prompt_list):
                prompt_id = prompt.id
                stats["categories_tested"].add(prompt.category)

                for _temp_idx, temperature in enumerate(temperatures):
                    for repl_idx in range(replications):
                        # Update progress description
                        current_test = (
                            f"{model} | {prompt.category}/{prompt_id} | "
                            f"T={temperature} | {repl_idx + 1}/{replications}"
                        )
                        pbar.set_description(f"Benchmarking: {current_test}")

                        # Prepare messages from prompt
                        messages = [msg.model_dump(exclude_none=True) for msg in prompt.messages]

                        # Add system prompt if provided and not already present
                        if system_prompt and not any(msg["role"] == "system" for msg in messages):
                            messages.insert(0, {"role": "system", "content": system_prompt})

                        start_time = time.perf_counter()
                        success = False
                        error_msg = None
                        in_tokens = None
                        out_tokens = None
                        response_text = None

                        try:
                            # Call with temperature and other parameters
                            call_kwargs = {"temperature": temperature}
                            if max_tokens is not None:
                                call_kwargs["max_tokens"] = max_tokens

                            response = chat_raw(model=model, messages=messages, **call_kwargs)
                            success = True

                            if response.choices and len(response.choices) > 0:
                                response_text = response.choices[0].message.content
                            else:
                                error_msg = "No response choices returned"
                                logger.warning(
                                    f"No choices in response for {model}, prompt {prompt_id}"
                                )

                            # Extract token usage information
                            if hasattr(response, "usage") and response.usage:
                                in_tokens = response.usage.prompt_tokens
                                out_tokens = response.usage.completion_tokens
                                logger.debug(
                                    f"Token usage for {model}: "
                                    f"prompt={in_tokens}, completion={out_tokens}, "
                                    f"total={in_tokens + out_tokens}"
                                )
                            else:
                                logger.debug(
                                    f"No token usage information available from {model}"
                                )

                        except Exception as e:
                            success = False
                            error_msg = str(e)
                            logger.exception(
                                f"Benchmark failed for {model}, prompt {prompt_id}, "
                                f"temp {temperature}: {error_msg}"
                            )

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
                            response_text=response_text,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            run_id=repl_idx,
                        )
                        results.append(result)

                        # Update statistics
                        stats["completed"] += 1
                        if success:
                            stats["successful"] += 1
                            model_successful += 1
                        else:
                            stats["failed"] += 1

                        model_results += 1
                        pbar.update(1)

            logger.info(f"Completed testing {model}")
            stats["models_tested"].add(model)

    # Save results
    save_results(results, results_path, timestamp)

    # Print summary
    print_summary_stats(stats, results)

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

        # Extract token counts from response
        in_tokens = None
        out_tokens = None
        tokens_from_usage = False

        if hasattr(response, "usage") and response.usage:
            in_tokens = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
            tokens_from_usage = True
            logger.debug(
                f"Token usage from API for {model}: "
                f"prompt={in_tokens}, completion={out_tokens}"
            )
        else:
            # Estimate tokens: approximately 4 characters per token
            content = response.choices[0].message.content
            out_tokens = len(content) // 4  # Integer division
            in_tokens = len(prompt) // 4  # Integer division
            logger.debug(
                f"Estimated token usage for {model}: "
                f"prompt≈{in_tokens}, completion≈{out_tokens} "
                "(API did not provide usage information)"
            )

        tokens_per_sec = out_tokens / latency if latency > 0 else 0

        return {
            "model": model,
            "latency": latency,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "tokens_from_usage": tokens_from_usage,
            "tokens_per_sec": tokens_per_sec,
            "success": True,
        }

    except Exception as e:
        logger.exception(f"Throughput benchmark failed for {model}: {e}")
        return {"model": model, "success": False, "error": str(e)}


def print_summary_stats(stats: dict[str, Any], results: list[BenchmarkResult]) -> None:
    """Print comprehensive benchmark statistics to console.

    Displays a formatted summary including:
    - Total test counts and success rates
    - Per-model performance metrics (success rate, average latency)
    - Category coverage
    - Overall performance statistics (min/max/avg latency)

    Args:
        stats: Dictionary with aggregate statistics from the benchmark run
        results: List of individual benchmark results for detailed analysis
    """
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Total Tests: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Successful: {stats['successful']} ({stats['successful'] / stats['total'] * 100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed'] / stats['total'] * 100:.1f}%)")

    print(f"\nModels Tested: {len(stats['models_tested'])}")
    for model in sorted(stats["models_tested"]):
        model_results = [r for r in results if r.model == model]
        success_rate = sum(1 for r in model_results if r.success) / len(model_results) * 100
        avg_latency = sum(r.latency_seconds for r in model_results) / len(model_results)
        print(f"  {model}: {success_rate:.1f}% success, {avg_latency:.3f}s avg latency")

    print(f"\nCategories Tested: {len(stats['categories_tested'])}")
    for category in sorted(stats["categories_tested"]):
        # Note: Category breakdown requires prompt metadata, simplified for now
        print(f"  {category}: tested")

    # Calculate overall statistics
    if results:
        latencies = [r.latency_seconds for r in results if r.success]
        if latencies:
            print("\nOverall Performance:")
            print(f"  Average Latency: {sum(latencies) / len(latencies):.3f}s")
            print(f"  Min Latency: {min(latencies):.3f}s")
            print(f"  Max Latency: {max(latencies):.3f}s")


def save_results(results: list[BenchmarkResult], directory: Path, timestamp: str) -> None:
    """Save benchmark results to a timestamped JSON file.

    Creates a JSON file with all benchmark results in the specified directory.
    Each result is converted to a dictionary representation.

    Args:
        results: List of benchmark results to save
        directory: Target directory for the JSON file
        timestamp: ISO format timestamp to use in filename

    Note:
        Colons in the timestamp are replaced with hyphens for filesystem compatibility.
    """
    filename = f"benchmark_{timestamp.replace(':', '-')}.json"
    filepath = directory / filename

    data = [asdict(r) for r in results]

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {filepath}")
