# HeLLMholtz AI Coding Instructions

## Project Overview
HeLLMholtz is a Python package providing unified LLM access, benchmarking, and reporting using the `aisuite` library. It supports multiple providers (OpenAI, Anthropic, Google, Ollama) plus a custom Helmholtz Blablador provider for internal models.

## Architecture
- **Core Components**: `core/config.py` (env-based settings), `client.py` (lazy singleton ClientManager for aisuite), `cli.py` (Typer-based CLI)
- **Benchmarking**: `benchmark/runner.py` (parallel benchmark execution), `evaluator.py` (LLM-as-a-Judge evaluation), `prompts.py` (categorized prompt sets)
- **Reporting**: `reporting.py` (Markdown/HTML generation), `export.py` (best model selection from results)
- **Integrations**: `integrations/lm_eval.py` (LM Evaluation Harness), `integrations/litellm.py` (proxy server)
- **Providers**: `providers/blablador_provider.py` (custom OpenAI-compatible provider), `blablador_config.py` (model metadata)
- **Data Flow**: Env config → ClientManager → aisuite Client → chat() returns str. Benchmarks collect BenchmarkResult dataclasses, save to JSON in `results/`, reports aggregate metrics.

## Key Patterns
- **Client Usage**: Always use `from hellmholtz.client import chat` for high-level string responses. For full response objects, access ClientManager.get_client() directly.
- **Model Specification**: Use `provider:model` format (e.g., `openai:gpt-4o`, `blablador:GPT-OSS-120b`). Blablador models resolve via `BlabladorModel.api_id` property.
- **Configuration**: Centralized in `Settings` dataclass from env vars. Load via `get_settings()`. Blablador requires `BLABLADOR_API_KEY` and `BLABLADOR_API_BASE`.
- **Benchmarking**: Use `run_benchmarks()` for multi-model/prompt runs. Results saved automatically to `results/benchmark_{timestamp}.json`. Evaluate with LLM judge via `evaluate_responses()`.
- **Custom Provider Registration**: Monkey-patch `ProviderFactory.get_supported_providers` to include 'blablador', inject module into `sys.modules` for importlib compatibility.
- **Error Handling**: Wrap API calls in try-except, log with `logger.error()`, raise `LLMError` for provider-specific issues.
- **Dependencies**: Use Poetry for management. Core: `aisuite`, `pydantic`, `python-dotenv`, `typer`. Optional: `lm-eval` (eval group), `litellm` (proxy group).

## Developer Workflows
- **Setup**: `poetry install` (editable mode), copy `.env.example` to `.env`, set API keys.
- **Run CLI**: `poetry run hellm <command>` (e.g., `poetry run hellm chat --model openai:gpt-4o "test"`).
- **Benchmarking**: `poetry run hellm bench --models openai:gpt-4o,blablador:Ministral-3-14B-Instruct-2512 --prompts-file prompts.txt`.
- **Testing**: `poetry run pytest` (paths: `tests/`, markers: `slow`, `integration`, `network`, `model`).
- **Linting/Formatting**: `poetry run ruff check .` / `poetry run ruff format .`, `poetry run mypy src/`.
- **Pre-commit**: `poetry run pre-commit run --all-files`.
- **Build**: `poetry build` (but typically use editable install).

## Conventions
- **Imports**: Absolute from `hellmholtz.*`, sorted with `isort` (known-first-party: `hellmholtz`).
- **Logging**: Use `logging.getLogger(__name__)` with `logger.info/debug/error()`.
- **CLI**: Typer commands in `cli.py`, configure logging in `main()`.
- **Benchmarks**: Use `tqdm` for progress bars, save results with ISO timestamps.
- **Models**: Blablador models have `id`, `name`, `alias`, `description`; API ID constructed as `{id} - {name} - {description}`.
- **Results**: JSON-serializable `BenchmarkResult` dataclass with `model`, `prompt_id`, `latency_seconds`, `success`, etc.
- **Reports**: Aggregate by model: success rate, avg latency. Use `generate_markdown_report()` for summaries.

## Integration Points
- **aisuite**: Unified client for all providers. Custom providers extend `Provider` class.
- **Blablador API**: OpenAI-compatible, requires specific model ID resolution.
- **LM Eval**: Run via `run_lm_eval()` with model string mapping (e.g., `openai:gpt-4o` → `openai-chat-completions`).
- **LiteLLM Proxy**: Start with `start_proxy()` for model routing.
- **External Config**: `.env` for secrets, `AISUITE_DEFAULT_MODELS` for fallbacks.

## Common Pitfalls
- **Model Resolution**: For Blablador, use model `name` or `alias` in `blablador:{name}`, not raw API ID.
- **Dependencies**: Install optional groups: `poetry install --with eval` for lm-eval.
- **Paths**: Scripts in `scripts/` add `src/` to `sys.path` manually.
- **Async**: No async support; all operations synchronous.
- **Token Counting**: Approximate via `len(content) / 4` if usage not provided.

Reference key files: `src/hellmholtz/client.py`, `src/hellmholtz/benchmark/runner.py`, `src/hellmholtz/providers/blablador_provider.py`, `pyproject.toml`, `docs/configuration.md`.
