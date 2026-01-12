# HeLLMholtz

**HeLLMholtz** is a comprehensive Python package and command-line suite for unified LLM access, benchmarking, and reporting. It serves as a thin, reusable wrapper around `aisuite`, providing a consistent interface for various LLM providers including OpenAI, Google, Anthropic, Ollama, and the Helmholtz "Blablador" API.

## Key Features

- **Unified Access**: Interact with multiple LLM providers through a single interface.
- **Centralized Configuration**: Manage settings and API keys via environment variables.
- **Advanced Benchmarking**: Compare model performance across temperatures, replications, and prompt categories.
- **LLM-as-a-Judge Evaluation**: Automated evaluation with comprehensive statistical analysis.
- **Interactive Reports**: HTML reports with Chart.js visualizations and Markdown summaries.
- **Model Monitoring**: Track Blablador model availability and configuration consistency.
- **Throughput Testing**: Measure tokens per second for performance analysis.
- **Flexible Prompts**: Support for both simple text files and structured JSON prompt collections.
- **Integrations**:
    - **LM Evaluation Harness**: Run academic benchmarks (e.g., MMLU, HellaSwag).
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

# Run Benchmarks
hellm bench --models openai:gpt-4o,anthropic:claude-3-haiku --prompts-category reasoning

# Monitor Models
hellm monitor --test-accessibility
```

## Documentation

- **[Installation](installation.md)**: Setup and installation instructions
- **[Configuration](configuration.md)**: Environment variables and settings
- **[Usage](usage.md)**: Command-line interface and Python API
- **[Models](models.md)**: Available models and providers
- **[Monitoring](monitoring.md)**: Model availability and health monitoring
- **[Contributing](https://github.com/JonasHeinickeBio/HeLLMholtz/blob/main/CONTRIBUTING.md)**: Development guidelines
