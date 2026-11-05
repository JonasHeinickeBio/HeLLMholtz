from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import re
import threading
import time
from typing import Any

from tqdm import tqdm

from hellmholtz.benchmark.checks import auto_score, run_checks
from hellmholtz.benchmark.scoring import final_score
from hellmholtz.client import chat_raw
from hellmholtz.core.prompts import Prompt

logger = logging.getLogger(__name__)


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient network/server issues)."""
    error_str = str(error).lower()
    # Retry on common transient errors
    retryable_patterns = [
        "502",
        "proxy error",
        "bad gateway",
        "503",
        "service unavailable",
        "504",
        "gateway timeout",
        "500",
        "internal server error",
        "connection",
        "timeout",
        "network",
        "rate limit",
        "429",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def _is_rate_limited(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Retry a function with exponential backoff for retryable errors.

    Rate limiting (HTTP 429) gets a longer schedule (8s, 16s, 32s, 60s, 60s...) and
    two extra attempts: retrying a throttled API after one second just burns the
    retry budget while the limit window is still open.
    """
    last_exception = None
    attempt = 0
    while True:
        try:
            return func()
        except Exception as e:
            last_exception = e
            limited = _is_rate_limited(e)
            budget = max_retries + (2 if limited else 0)
            if attempt >= budget or not _is_retryable_error(e):
                raise e

            if limited:
                delay = min(8.0 * (2**attempt), 60.0)
            else:
                delay = min(base_delay * (backoff_factor**attempt), max_delay)
            jitter = random.uniform(0.1, 1.0) * delay * 0.1  # nosec: non-cryptographic jitter
            total_delay = delay + jitter

            logger.warning(
                f"Attempt {attempt + 1}/{budget + 1} failed with retryable error: "
                f"{str(e)[:200]}. Retrying in {total_delay:.1f}s..."
            )
            time.sleep(total_delay)
            attempt += 1

    raise last_exception  # pragma: no cover


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run, enriched with performance and evaluation metrics."""

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
    tokens_per_sec: float | None = None
    finish_reason: str | None = None  # Completion finish reason (e.g. "stop", "length")
    category: str | None = None
    auto_score: float | None = None  # share of deterministic checks passed, 0-1
    check_results: list[dict[str, Any]] | None = None
    judge_score: float | None = None  # panel-averaged judge score, 0-1
    judge_spread: float | None = None  # max-min across judges (disagreement)
    judge_details: list[dict[str, Any]] | None = None
    final_score: float | None = None  # combined 0-100 score (see benchmark.scoring)
    ttft_seconds: float | None = None  # time to first streamed token (speed probe only)
    reasoning_tokens: int | None = None  # hidden/visible thinking tokens, when detectable


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
    resume_from: Path | None = None,
    _partial_path: Path | None = None,
    _save: bool = True,
    _resume: dict[tuple[str, str, float, int], BenchmarkResult] | None = None,
    _streaks: "_FailureStreaks | None" = None,
) -> list[BenchmarkResult]:
    """Run benchmarks across multiple models and prompts.

    Args:
        models: Model identifiers to benchmark
        prompts: Prompts to test
        system_prompt: Optional system prompt to prepend
        repeat: Number of times to repeat each model-prompt combination
        results_dir: Directory to save results
        max_concurrent: Parallel workers. Work is split into (model, prompt-chunk) units so
            slow models do not serialise the run; requests inside a unit stay sequential
        temperatures: List of temperature values to test (default: [0.1, 0.7, 1.0])
        max_tokens: Maximum tokens for response
        replications: Number of times to replicate each test for statistical significance
        resume_from: Incremental ``.partial.jsonl`` from an interrupted run. Successful runs
            in it are reused instead of re-requested and new results are appended to the
            same file; failed ones are retried.
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

    if resume_from is not None and _resume is None:
        _resume = _load_resume(resume_from)
        _terminate_last_line(resume_from)
        _partial_path = _partial_path or resume_from
        logger.info(f"Resuming from {resume_from}: {len(_resume)} successful runs reused")
    resume = _resume or {}

    if max_concurrent > 1 and (_partial_path is None or resume_from is not None):
        return _run_parallel(
            model_list,
            prompt_list,
            system_prompt=system_prompt,
            results_path=results_path,
            timestamp=timestamp,
            max_concurrent=max_concurrent,
            temperatures=temperatures,
            max_tokens=max_tokens,
            replications=replications,
            partial_path=_partial_path,
            resume=resume,
        )

    # Incremental recovery file: one JSON object per completed test, appended live so
    # an interrupted run keeps everything completed so far.
    partial_path = _partial_path or (
        results_path / f"benchmark_{timestamp.replace(':', '-')}.partial.jsonl"
    )
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
        "skipped_models": set(),
    }

    with tqdm(
        total=total_tests,
        disable=not _save,
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
            streaks = _streaks or _FailureStreaks()
            # Skip a model after this many consecutive API/transport failures.
            # Empty completions are model-quality issues, not API outages, and
            # never count toward this threshold.
            max_consecutive_api_failures = 5

            logger.info(f"Testing model {model_idx + 1}/{len(model_list)}: {model}")

            for _prompt_idx, prompt in enumerate(prompt_list):
                prompt_id = prompt.id
                stats["categories_tested"].add(prompt.category)

                for _temp_idx, temperature in enumerate(temperatures):
                    for repl_idx in range(replications):
                        # Stored successes are kept even after the cutoff; they say nothing
                        # about the API, so they never reset the failure streak.
                        reused = resume.get((model, prompt_id, float(temperature), repl_idx))
                        if reused is not None:
                            reused.category = prompt.category
                            _score_with_checks(reused, prompt)
                            results.append(reused)
                            stats["completed"] += 1
                            stats["successful"] += 1
                            model_successful += 1
                            model_results += 1
                            pbar.update(1)
                            continue

                        # Skip the rest of this model after too many consecutive API failures
                        if streaks.get(model) >= max_consecutive_api_failures:
                            if model not in stats["skipped_models"]:
                                logger.warning(
                                    f"Skipping remaining tests for {model} due to "
                                    f"{streaks.get(model)} consecutive API failures"
                                )
                            stats["skipped_models"].add(model)
                            pbar.update(1)
                            continue

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
                        reasoning_tokens = None
                        response_text = None
                        finish_reason = None
                        transport_failure = False

                        try:
                            # Call with temperature and other parameters with retry logic
                            call_kwargs = {"temperature": temperature}
                            if max_tokens is not None:
                                call_kwargs["max_tokens"] = max_tokens

                            def _make_api_call(
                                m: str = model,
                                msg: list[dict[str, str]] = messages,
                                kw: dict[str, Any] = call_kwargs,
                            ) -> Any:
                                return chat_raw(model=m, messages=msg, **kw)

                            response = _retry_with_backoff(_make_api_call, max_retries=3)
                            success = True

                            if response.choices and len(response.choices) > 0:
                                response_text = response.choices[0].message.content
                                reasoning_tokens = count_reasoning_tokens(response)
                                fr = getattr(response.choices[0], "finish_reason", None)
                                finish_reason = fr if isinstance(fr, str) else None
                                # The call itself can succeed yet still return an
                                # empty completion (e.g. the model exhausts its
                                # token budget on hidden output and is truncated
                                # with finish_reason="length"). Treat these as
                                # failures so successful results never contain an
                                # empty response, and keep the finish reason for
                                # diagnostics.
                                if not (response_text or "").strip():
                                    success = False
                                    error_msg = (
                                        "Model returned an empty response "
                                        f"(finish_reason={finish_reason})"
                                    )
                                    logger.warning(
                                        f"Empty response from {model} for prompt "
                                        f"{prompt_id} (finish_reason={finish_reason})"
                                    )
                                elif finish_reason == "length":
                                    # Cut off at the token limit: an incomplete answer
                                    # scores 0 like any other failed run.
                                    success = False
                                    error_msg = "Response truncated (finish_reason=length)"
                            else:
                                success = False
                                error_msg = "No response choices returned"
                                transport_failure = True
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
                                logger.debug(f"No token usage information available from {model}")

                        except Exception as e:
                            success = False
                            error_msg = str(e)
                            transport_failure = True
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
                            finish_reason=finish_reason,
                            reasoning_tokens=reasoning_tokens,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            run_id=repl_idx,
                            tokens_per_sec=(
                                out_tokens / latency
                                if latency > 0 and out_tokens is not None
                                else None
                            ),
                            category=prompt.category,
                        )
                        _score_with_checks(result, prompt)
                        results.append(result)

                        # Append to the incremental recovery file immediately.
                        try:
                            with _PARTIAL_LOCK, open(partial_path, "a") as pf:
                                pf.write(json.dumps(asdict(result)) + "\n")
                        except OSError as save_exc:  # pragma: no cover - best effort
                            logger.warning(f"Could not write incremental result: {save_exc}")

                        # Update statistics
                        stats["completed"] += 1
                        if success:
                            stats["successful"] += 1
                            model_successful += 1
                            streaks.reset(model)  # Reset on success
                        else:
                            stats["failed"] += 1
                            if transport_failure:
                                streaks.hit(model)
                            else:
                                # Empty completion: the API is up and answered, so
                                # this is a model-quality issue, not an outage.
                                streaks.reset(model)

                        model_results += 1
                        pbar.update(1)

            # If this model was skipped due to consecutive API failures, move on
            # to the next model instead of aborting the whole benchmark.
            if streaks.get(model) >= max_consecutive_api_failures:
                continue

            logger.info(f"Completed testing {model}")
            stats["models_tested"].add(model)

    if _save:
        save_results(results, results_path, timestamp)
        print_summary_stats(stats, results)

    return results


_PARTIAL_LOCK = threading.Lock()


def _terminate_last_line(path: Path) -> None:
    """Make sure appended records start on a fresh line after a killed writer."""
    with open(path, "rb+") as f:
        f.seek(0, 2)
        if f.tell() and (f.seek(-1, 2) is not None) and f.read(1) != b"\n":
            f.write(b"\n")


def _load_resume(path: Path) -> dict[tuple[str, str, float, int], BenchmarkResult]:
    """Successful runs of an interrupted benchmark, keyed by (model, prompt, temp, run)."""
    known = {f.name for f in fields(BenchmarkResult)}
    reusable: dict[tuple[str, str, float, int], BenchmarkResult] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a killed process
        if not data.get("success"):
            continue
        key = (
            data["model"],
            data["prompt_id"],
            float(data.get("temperature") or 0.0),
            int(data.get("run_id") or 0),
        )
        reusable[key] = BenchmarkResult(**{k: v for k, v in data.items() if k in known})
    return reusable


def count_reasoning_tokens(response: Any) -> int | None:
    """Thinking tokens of a completion, or None when the API does not expose them.

    Prefers the provider's own count (``usage.completion_tokens_details``); otherwise
    estimates from a ``reasoning_content`` field or inline ``<think>`` blocks
    (about 4 characters per token).
    """
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    counted = getattr(details, "reasoning_tokens", None)
    if isinstance(counted, int) and counted > 0:
        return counted
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return None
    chars = 0
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str):
            chars += len(value)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        chars += sum(len(m) for m in _THINK_BLOCK.findall(content))
    return max(1, chars // 4) if chars else None


_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


class _FailureStreaks:
    """Consecutive API failures per model, shared by the chunks of one model."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, model: str) -> int:
        with self._lock:
            return self._counts.get(model, 0)

    def hit(self, model: str) -> None:
        with self._lock:
            self._counts[model] = self._counts.get(model, 0) + 1

    def reset(self, model: str) -> None:
        with self._lock:
            self._counts[model] = 0


def _score_with_checks(result: BenchmarkResult, prompt: Prompt) -> None:
    """Run the prompt's deterministic checks and (re)compute auto/final scores.

    An existing ``judge_score`` is kept, so resumed records can be rescored against the
    current prompt definition without losing their verdict. Prompts that need a judge
    (rubric, or no checks at all) stay unscored until one has answered.
    """
    if prompt.checks and result.success:
        checks = run_checks(prompt.checks, result.response_text)
        result.check_results = [c.as_dict() for c in checks]
        result.auto_score = auto_score(checks)
    result.final_score = final_score(
        result.auto_score,
        result.judge_score,
        success=result.success,
        empty=not (result.response_text or "").strip(),
        needs_judge=bool(prompt.rubric) or not prompt.checks,
    )


def _run_parallel(
    model_list: list[str],
    prompt_list: list[Prompt],
    *,
    system_prompt: str | None,
    results_path: Path,
    timestamp: str,
    max_concurrent: int,
    temperatures: list[float],
    max_tokens: int | None,
    replications: int,
    partial_path: Path | None = None,
    resume: dict[tuple[str, str, float, int], BenchmarkResult] | None = None,
) -> list[BenchmarkResult]:
    """Benchmark several models concurrently, sharing one results/partial file."""
    partial_path = partial_path or (
        results_path / f"benchmark_{timestamp.replace(':', '-')}.partial.jsonl"
    )
    logger.info(f"Running {len(model_list)} models with {max_concurrent} parallel workers")

    # Work units are (model, every k-th prompt): a slow reasoning model is spread over
    # several workers instead of becoming a multi-hour sequential tail, while every
    # unit still mixes easy and hard prompts.
    chunks = max(1, min(4, len(prompt_list), max_concurrent))
    units = [(m, prompt_list[i::chunks]) for m in model_list for i in range(chunks)]

    streaks = _FailureStreaks()  # shared by all chunks of a model

    def one(unit: tuple[str, list[Prompt]]) -> list[BenchmarkResult]:
        model, subset = unit
        return run_benchmarks(
            [model],
            subset,
            system_prompt=system_prompt,
            results_dir=str(results_path),
            max_concurrent=1,
            temperatures=temperatures,
            max_tokens=max_tokens,
            replications=replications,
            _partial_path=partial_path,
            _save=False,
            _resume=resume or {},
            _streaks=streaks,
        )

    results: list[BenchmarkResult] = []
    done_models: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        for (model, _subset), batch in zip(units, pool.map(one, units), strict=True):
            done_models[model] = done_models.get(model, 0) + len(batch)
            results.extend(batch)
    for model, n in done_models.items():
        logger.info(f"Finished {model}: {n} runs")

    save_results(results, results_path, timestamp)
    stats: dict[str, Any] = {
        "total": len(results),
        "completed": len(results),
        "successful": sum(r.success for r in results),
        "failed": sum(not r.success for r in results),
        "models_tested": {r.model for r in results},
        "categories_tested": {r.category for r in results if r.category},
        "skipped_models": set(),
    }
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
                f"Token usage from API for {model}: prompt={in_tokens}, completion={out_tokens}"
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

    if stats.get("skipped_models"):
        print(f"Skipped Models: {len(stats['skipped_models'])}")
        for model in sorted(stats["skipped_models"]):
            print(f"  {model}: skipped due to consecutive API failures")

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
