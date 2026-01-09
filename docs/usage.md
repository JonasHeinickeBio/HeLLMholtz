# Usage

HeLLMholtz provides both a Python API and a Command-Line Interface (CLI).

## Command-Line Interface (`hellm`)

The `hellm` command is the main entry point.

### Chat

Chat with an LLM directly from the terminal:

```bash
hellm chat --model openai:gpt-4o "Explain quantum computing in one sentence."
```

### List Models

List available models from the Blablador API:

```bash
hellm models
```

### Benchmarking

Run benchmarks across models and prompts. HeLLMholtz supports multiple ways to specify prompts:

#### Using Built-in Prompt Categories

Use predefined prompts organized by category:

```bash
# Use reasoning prompts (default)
hellm bench --models openai:gpt-4o --prompts-category reasoning

# Use all available categories
hellm bench --models openai:gpt-4o --all-prompts

# Available categories: reasoning, coding, creative, knowledge
```

#### Using Custom Prompt Files

Load prompts from external files:

```bash
# Simple text file (one prompt per line)
hellm bench --models openai:gpt-4o --prompts-file prompts.txt

# Structured JSON file with full prompt metadata
hellm bench --models openai:gpt-4o --prompts-file custom_prompts.json
```

**Prompt File Formats:**

- **Text files (.txt)**: Simple format with one prompt per line
- **JSON files (.json)**: Structured format supporting categories, descriptions, and expected outputs

**Example JSON prompts file:**
```json
[
  {
    "id": "custom_reasoning_001",
    "category": "reasoning",
    "messages": [
      {
        "role": "user",
        "content": "If a plane crashes on the border of the US and Canada, where do they bury the survivors?"
      }
    ],
    "description": "Classic lateral thinking puzzle",
    "expected_output": "Answer to riddle"
  }
]
```

### Throughput Benchmarking

Measure the throughput (tokens/sec) of a model:

```bash
hellm bench-throughput --model ollama:llama3.2
```

### Reporting

Generate a Markdown report from benchmark results:

```bash
hellm report --input-file results/benchmark_20241214_120000.json --output-file report.md
```

### LM Evaluation Harness

Run academic benchmarks (requires `[eval]` extra):

```bash
hellm lm-eval --model openai:gpt-4o --tasks mmlu --limit 10
```

### LiteLLM Proxy

Start an OpenAI-compatible proxy server (requires `[proxy]` extra):

```bash
hellm proxy --model ollama:llama3.2 --port 4000
```

## Python API

### Chat

```python
from hellmholtz.client import chat

response = chat("openai:gpt-4o", [{"role": "user", "content": "Hello!"}])
print(response)
```

### Benchmarking

```python
from hellmholtz.benchmark import run_throughput_benchmark

stats = run_throughput_benchmark("ollama:llama3.2")
print(f"Tokens/sec: {stats['tokens_per_sec']}")
```
