# HeLLMholtz

**HeLLMholtz** is a comprehensive Python package and command-line suite for unified LLM access, benchmarking, and reporting. It serves as a thin, reusable wrapper around `aisuite`, providing a consistent interface for various LLM providers including OpenAI, Google, Anthropic, Ollama, and the Helmholtz "Blablador" API.

## Key Features

- **Unified Access**: Interact with multiple LLM providers through a single interface.
- **Centralized Configuration**: Manage settings and API keys via environment variables.
- **Benchmarking**: Run custom benchmarks to measure latency, success rates, and token usage.
- **Throughput Testing**: Measure tokens per second for performance analysis.
- **Reporting**: Generate human-readable summaries of benchmark results.
- **Integrations**:
    - **LM Evaluation Harness**: Run academic benchmarks (e.g., MMLU).
    - **LiteLLM Proxy**: Serve models via an OpenAI-compatible proxy.
- **CLI**: Powerful command-line interface `hellm` for all operations.

## Quick Start

```bash
# Install
pip install -e .

# Chat
hellm chat --model openai:gpt-4o "Hello, world!"

# List Models
hellm models
```
