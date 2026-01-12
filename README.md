# HeLLMholtz LLM Suite

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/hellmholtz.svg)](https://pypi.org/project/hellmholtz/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/JonasHeinickeBio/HeLLMholtz/actions)

A comprehensive Python package for unified LLM access, benchmarking, evaluation, and reporting. Built on top of `aisuite` with specialized support for Helmholtz Blablador models.

## ✨ Features

- **🔄 Unified Client**: Single interface for OpenAI, Google, Anthropic, Ollama, and Helmholtz Blablador models
- **⚙️ Centralized Configuration**: Environment-based configuration for all your projects
- **📊 Advanced Benchmarking**: Compare model performance across temperatures, replications, and prompt categories
- **🔍 LLM-as-a-Judge Evaluation**: Automated evaluation with comprehensive statistical analysis
- **📈 Interactive Reports**: HTML reports with Chart.js visualizations and Markdown summaries
- **🎯 Flexible Prompt System**: Support for both simple text files and structured JSON prompt collections
- **📊 Model Monitoring**: Track Blablador model availability and configuration consistency
- **🚀 LM Evaluation Harness**: Integration with EleutherAI's comprehensive evaluation suite
- **🌐 LiteLLM Proxy**: Built-in proxy server for model routing and load balancing
- **⚡ Throughput Testing**: Performance benchmarking for high-throughput scenarios
- **🔍 Model Discovery**: Dynamic model listing and availability checking

## 🚀 Installation

### Basic Installation

```bash
pip install hellmholtz
```

### Development Installation

For development with all optional dependencies:

```bash
git clone https://github.com/JonasHeinickeBio/HeLLMholtz.git
cd HeLLMholtz
pip install -e ".[eval,proxy]"
```

### Poetry Installation

```bash
poetry install --with eval,proxy
```

## ⚙️ Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your API keys in `.env`:
```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google
GOOGLE_API_KEY=your_google_key

# Helmholtz Blablador
BLABLADOR_API_KEY=your_blablador_key
BLABLADOR_API_BASE=https://your-blablador-instance.com

# Optional: Default models
AISUITE_DEFAULT_MODELS='{"openai": "gpt-4o", "anthropic": "claude-3-haiku"}'
```

## 📖 Usage

### Python API

#### Basic Chat Interface

```python
from hellmholtz.client import chat

# Simple chat
response = chat("openai:gpt-4o", "Hello, how are you?")
print(response)

# With conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]
response = chat("anthropic:claude-3-sonnet", messages)
```

#### Benchmarking

```python
from hellmholtz.benchmark import run_benchmarks
from hellmholtz.core.prompts import load_prompts

# Load prompts from JSON file
prompts = load_prompts("prompts.json", category="reasoning")

# Run benchmarks
results = run_benchmarks(
    models=["openai:gpt-4o", "anthropic:claude-3-haiku", "blablador:gpt-4o"],
    prompts=prompts,
    temperatures=[0.1, 0.7, 1.0],
    replications=3
)

# Analyze results
from hellmholtz.evaluation_analysis import EvaluationAnalyzer
analyzer = EvaluationAnalyzer()
analysis = analyzer.analyze_evaluation_results("results/benchmark_latest.json")
analyzer.print_analysis_summary(analysis)
```

### Command Line Interface

HeLLMholtz provides a comprehensive CLI for all operations:

#### Chat Interface

```bash
# Simple chat
hellm chat --model openai:gpt-4o "Explain the theory of relativity"

# Interactive mode
hellm chat --model anthropic:claude-3-sonnet --interactive

# With system prompt
hellm chat --model blablador:gpt-4o --system "You are a coding assistant" "Write a Python function to calculate fibonacci numbers"
```

#### Benchmarking

```bash
# Basic benchmark
hellm bench --models openai:gpt-4o,anthropic:claude-3-haiku --prompts-file prompts.txt

# Advanced benchmark with evaluation
hellm bench \
  --models openai:gpt-4o,blablador:gpt-4o \
  --prompts-file prompts.json \
  --prompts-category reasoning \
  --temperatures 0.1,0.7,1.0 \
  --replications 3 \
  --evaluate-with openai:gpt-4o \
  --results-dir results/

# Throughput testing
hellm bench-throughput \
  --model openai:gpt-4o \
  --requests 100 \
  --concurrency 10 \
  --prompt "Write a short story about AI"
```

#### Evaluation and Analysis

```bash
# Analyze benchmark results
hellm analyze results/benchmark_latest.json --html-report analysis_report.html

# Generate reports
hellm report --results-file results/benchmark_latest.json --output report.md
```

#### Model Management

```bash
# List available Blablador models
hellm models

# Monitor model availability and test accessibility
hellm monitor --test-accessibility

# Check model configuration consistency
hellm monitor --check-config
```

#### Advanced Features

```bash
# Run LM Evaluation Harness
hellm lm-eval \
  --model openai:gpt-4o \
  --tasks hellaswag,winogrande \
  --limit 100

# Start LiteLLM proxy server
hellm proxy \
  --config litellm_config.yaml \
  --port 8000
```

## 📁 Project Structure

```
hellmholtz/
├── cli.py                 # Command-line interface
├── client.py              # Unified LLM client
├── monitoring.py          # Model availability monitoring
├── evaluation_analysis.py # Statistical analysis and reporting
├── export.py              # Result export utilities
├── core/
│   ├── config.py          # Configuration management
│   └── prompts.py         # Prompt loading and validation
├── benchmark/
│   ├── runner.py          # Benchmark execution
│   ├── evaluator.py       # LLM-as-a-Judge evaluation
│   └── prompts.py         # Benchmark-specific prompts
├── providers/
│   ├── blablador_provider.py # Custom Blablador provider
│   ├── blablador_config.py   # Blablador model configuration
│   ├── blablador.py          # Blablador utilities
│   └── __init__.py
├── reporting/
│   ├── html.py            # HTML report generation
│   ├── markdown.py        # Markdown report generation
│   ├── stats.py           # Statistical calculations
│   ├── utils.py           # Reporting utilities
│   └── templates/         # HTML templates
└── integrations/
    ├── lm_eval.py         # LM Evaluation Harness integration
    └── litellm.py         # LiteLLM proxy integration
```

## 🎯 Prompt System

HeLLMholtz supports two prompt formats:

### Simple Text Format (`prompts.txt`)

```
What is the capital of France?
Explain quantum computing in simple terms.
Write a Python function to reverse a string.
```

### Structured JSON Format (`prompts.json`)

```json
[
  {
    "id": "capital-france",
    "category": "knowledge",
    "description": "Test basic geographical knowledge",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "expected_output": "Paris"
  },
  {
    "id": "quantum-explanation",
    "category": "reasoning",
    "description": "Test ability to explain complex concepts simply",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
      }
    ]
  }
]
```

## 📊 Evaluation System

The LLM-as-a-Judge evaluation system provides:

- **Automated Scoring**: AI-powered evaluation of response quality
- **Statistical Analysis**: Comprehensive metrics and distributions
- **Model Rankings**: Performance comparisons across all dimensions
- **Interactive Reports**: Web-based visualizations of results
- **Detailed Critiques**: Specific feedback for each response

### Example Analysis Output

```
🔍 EVALUATION ANALYSIS RESULTS
══════════════════════════════════════════════════════════════

📊 OVERVIEW
• Total Evaluations: 150
• Models Tested: 3
• Prompts Tested: 5
• Success Rate: 94.7%

🏆 MODEL RANKINGS
1. openai:gpt-4o        - Avg Score: 8.7/10 (±0.8)
2. anthropic:claude-3-opus - Avg Score: 8.4/10 (±0.9)
3. blablador:gpt-4o     - Avg Score: 7.9/10 (±1.1)

📈 DETAILED METRICS
• Response Quality: 8.3/10 average
• Relevance: 8.6/10 average
• Accuracy: 9.1/10 average
• Creativity: 7.8/10 average
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/JonasHeinickeBio/HeLLMholtz.git
cd HeLLMholtz

# Install with development dependencies
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=hellmholtz --cov-report=html

# Run specific test categories
poetry run pytest -m "slow"        # Slow integration tests
poetry run pytest -m "network"     # Tests requiring network access
poetry run pytest -m "model"       # Tests using actual models
```

### Code Quality

```bash
# Lint code
poetry run ruff check .

# Format code
poetry run ruff format .

# Type checking
poetry run mypy src/

# Security scanning
poetry run bandit -r src/
```

### Building Documentation

```bash
# Generate API documentation
poetry run sphinx-build docs/ docs/_build/

# Serve documentation locally
poetry run sphinx-serve docs/_build/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the full test suite: `poetry run pytest`
5. Ensure code quality: `poetry run ruff check . && poetry run mypy src/`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [aisuite](https://github.com/andrewyng/aisuite) for unified LLM access
- LLM evaluation powered by [EleutherAI's LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Proxy functionality via [LiteLLM](https://github.com/BerriAI/litellm)
- Special thanks to the Helmholtz Association for Blablador model access

## 📞 Support

- 📖 [Documentation](https://hellmholtz.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/JonasHeinickeBio/HeLLMholtz/issues)
- 💬 [Discussions](https://github.com/JonasHeinickeBio/HeLLMholtz/discussions)

---

<p align="center">Made with ❤️ for the scientific computing community</p>
