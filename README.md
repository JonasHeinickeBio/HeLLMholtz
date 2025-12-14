# Helmholtz LLM Suite

A thin, reusable Python package for unified LLM access, benchmarking, and reporting using `aisuite`.

## Features

- **Unified Client**: Single interface for OpenAI, Google, Anthropic, Ollama, and Helmholtz Blablador.
- **Centralized Config**: Environment-based configuration for all your projects.
- **Benchmarking**: Easy-to-use tools for comparing model performance.
- **Reporting**: Generate Markdown reports and export "best model" configurations.

## Installation

Install in editable mode:

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Usage

### Python API

```python
from hellmholtz.client import chat
from hellmholtz.export import get_default_model_config

# Use default best model
cfg = get_default_model_config()
response = chat(cfg["model"], [{"role": "user", "content": "Hello!"}])
print(response)

# Specific model
response = chat("openai:gpt-4o", [{"role": "user", "content": "Hello!"}])
```

### CLI

```bash
# Chat
helmllm chat --model openai:gpt-4o "Hello world"

# Benchmark
helmllm bench --models openai:gpt-4o,ollama:llama3.2 --prompts-file prompts.txt

# Report
helmllm report --results-file results/benchmark_latest.json
```
