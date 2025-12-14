# Installation

HeLLMholtz requires Python 3.10 or higher.

## Basic Installation

To install the core package:

```bash
pip install -e .
```

## With Optional Dependencies

HeLLMholtz has optional dependencies for advanced features:

- **`eval`**: For running LM Evaluation Harness benchmarks.
- **`proxy`**: For running the LiteLLM proxy.
- **`dev`**: For development tools (testing, linting).

To install with specific extras:

```bash
# For evaluation support
pip install -e ".[eval]"

# For proxy support
pip install -e ".[proxy]"

# For everything
pip install -e ".[eval,proxy,dev]"
```

## Using Poetry

If you are developing HeLLMholtz, you can use Poetry:

```bash
poetry install --extras "eval proxy"
```
