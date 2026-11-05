"""Speed probe: time-to-first-token and decode throughput per model.

The quality prompts measure end-to-end latency, which mixes in answer length and
hidden thinking. The probe sends one short, fixed-length request per model over a
streaming connection so the two speed components can be read apart:

* ``ttft_seconds`` - request start until the first streamed token (queueing plus
  prefill, plus thinking for reasoning models that stream it);
* ``tokens_per_sec`` - tokens after the first one divided by the time they took.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import time
from typing import Any

from hellmholtz.benchmark.runner import BenchmarkResult
from hellmholtz.benchmark.scoring import speed_score  # noqa: F401  (re-exported)
from hellmholtz.client import DEFAULT_REQUEST_TIMEOUT, ClientManager

logger = logging.getLogger(__name__)

SPEED_PROMPT_ID = "speed_probe"
SPEED_CATEGORY = "speed"
PROBE_MAX_TOKENS = 400
PROBE_PROMPT = (
    "Count from 1 to 80 in words separated by commas (one, two, three, ...). "
    "Output only the list, nothing else."
)


def is_speed_record(record: Any) -> bool:
    """True for speed-probe results, which are kept out of the quality score."""
    if isinstance(record, dict):
        return record.get("category") == SPEED_CATEGORY
    return getattr(record, "category", None) == SPEED_CATEGORY


def _delta_text(chunk: Any) -> str:
    """Any visible or reasoning text carried by one streamed chunk."""
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError, TypeError):
        return ""
    parts = []
    for attr in ("content", "reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if isinstance(value, str):
            parts.append(value)
    return "".join(parts)


def _open_stream(model: str, max_tokens: int) -> Any:
    """Open a streamed completion; Blablador goes through its provider's OpenAI client."""
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    extra: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream_options": {"include_usage": True},
        "timeout": DEFAULT_REQUEST_TIMEOUT,
    }
    client, effective_model = ClientManager.get_client(model)
    provider_key, _, name = model.partition(":")
    providers = getattr(client, "providers", None)
    if provider_key == "blablador" and providers is not None:
        if provider_key not in providers:
            from aisuite.provider import ProviderFactory

            providers[provider_key] = ProviderFactory.create_provider(  # type: ignore[attr-defined]
                provider_key, dict(client.provider_configs.get(provider_key, {}))
            )
        return providers[provider_key].stream_chat(name, messages, **extra)
    return client.chat.completions.create(
        model=effective_model, messages=messages, stream=True, **extra
    )


def probe_once(model: str, max_tokens: int = PROBE_MAX_TOKENS) -> BenchmarkResult:
    """One streamed request; failures are returned as unsuccessful results."""
    timestamp = datetime.now().isoformat()
    start = time.perf_counter()
    first: float | None = None
    chars = 0
    out_tokens: int | None = None
    in_tokens: int | None = None
    error: str | None = None
    try:
        stream = _open_stream(model, max_tokens)
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                out_tokens = getattr(usage, "completion_tokens", out_tokens)
                in_tokens = getattr(usage, "prompt_tokens", in_tokens)
            text = _delta_text(chunk)
            if text:
                if first is None:
                    first = time.perf_counter()
                chars += len(text)
    except Exception as exc:
        error = str(exc)[:300]
        logger.warning(f"Speed probe failed for {model}: {error}")
    end = time.perf_counter()

    ok = error is None and first is not None
    tokens = out_tokens if out_tokens else (max(1, chars // 4) if chars else None)
    decode = end - first if first is not None else 0.0
    tps = (tokens - 1) / decode if ok and tokens and tokens > 1 and decode > 0 else None
    return BenchmarkResult(
        model=model,
        prompt_id=SPEED_PROMPT_ID,
        latency_seconds=end - start,
        success=ok,
        timestamp=timestamp,
        input_tokens=in_tokens,
        output_tokens=tokens,
        error_message=error if error else (None if ok else "stream produced no tokens"),
        max_tokens=max_tokens,
        temperature=0.0,
        tokens_per_sec=tps,
        category=SPEED_CATEGORY,
        ttft_seconds=(first - start) if first is not None else None,
        final_score=None,
    )


def probe_models(models: Iterable[str], runs: int = 2, workers: int = 4) -> list[BenchmarkResult]:
    """Probe each model ``runs`` times (models in parallel, runs of one model in series)."""

    def work(model: str) -> list[BenchmarkResult]:
        out = []
        for i in range(runs):
            result = probe_once(model)
            result.run_id = i
            out.append(result)
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [r for batch in pool.map(work, list(models)) for r in batch]
