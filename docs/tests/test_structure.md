# Test Structure - 2026-08-13

**Project**: helmholtz_llm_suite  
**Branch**: `feature/blablador-model-manager`  
**Date**: 2026-08-13

---

## Overview

Tests have been reorganized into a directory structure that mirrors `src/hellmholtz/` for better maintainability and easier navigation.

---

## Directory Structure

```
tests/
├── __init__.py                    # Package marker
├── benchmark/                     # Benchmark and performance tests
│   ├── __init__.py
│   ├── test_benchmark.py
│   ├── test_blablador.py
│   ├── test_blablador_config.py
│   ├── test_blablador_provider.py
│   ├── test_config_converter.py
│   ├── test_evaluation_analysis.py
│   ├── test_export.py
│   ├── test_export_module.py
│   ├── test_exporters.py
│   ├── test_model_manager_core.py
│   ├── test_stats.py
│   └── test_throughput.py
├── cli/                           # CLI command tests
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_cli_benchmark.py
│   ├── test_cli_common.py
│   ├── test_cli_model_manager.py
│   ├── test_cli_models.py
│   └── test_cli_modules.py
├── core/                          # Core module tests
│   ├── __init__.py
│   ├── test_client.py
│   ├── test_config.py
│   ├── test_prompts.py
│   └── test_prompts_core.py
├── integrations/                  # Integration tests
│   ├── __init__.py
│   ├── test_cli_integrations.py
│   └── test_integrations.py
├── providers/                     # Provider tests
│   ├── __init__.py
│   ├── test_litellm.py
│   ├── test_monitoring.py
│   ├── test_monitoring_core.py
│   ├── test_providers.py
│   └── test_token_limits.py
└── reporting/                     # Reporting module tests
    ├── __init__.py
    ├── test_reporting.py
    └── test_reporting_chart.py
```

---

## Mapping to Source Structure

```
src/hellmholtz/
├── benchmark/          →  tests/benchmark/
├── cli/                →  tests/cli/
├── core/               →  tests/core/
├── integrations/       →  tests/integrations/
├── providers/          →  tests/providers/
└── reporting/          →  tests/reporting/
```

---

## Test File Conventions

### Naming
- Test files: `test_*.py`
- Test classes: `Test*` (CamelCase)
- Test methods: `test_*` (snake_case)

### Imports
Tests use absolute imports:

```python
from hellmholtz.providers.blablador import BlabladorAPI
from hellmholtz.cli.models import list_models, check_model
```

---

## Test Categories

### 1. Benchmark Tests (`tests/benchmark/`)

Tests for performance metrics and model evaluation:

| Test File | Coverage |
|-----------|----------|
| `test_benchmark.py` | Benchmark test setup |
| `test_blablador.py` | Blablador provider tests |
| `test_blablador_config.py` | Configuration parsing |
| `test_blablador_provider.py` | Provider integration |
| `test_config_converter.py` | Config conversion logic |
| `test_evaluation_analysis.py` | Evaluation result analysis |
| `test_export.py` | Export functionality |
| `test_export_module.py` | Module export tests |
| `test_exporters.py` | Exporter classes |
| `test_model_manager_core.py` | Model manager core |
| `test_stats.py` | Statistics generation |
| `test_throughput.py` | Throughput measurements |

### 2. CLI Tests (`tests/cli/`)

Tests for command-line interface:

| Test File | Coverage |
|-----------|----------|
| `test_cli.py` | Main CLI commands |
| `test_cli_benchmark.py` | Benchmark subcommands |
| `test_cli_common.py` | Common CLI utilities |
| `test_cli_model_manager.py` | Model manager CLI |
| `test_cli_models.py` | Models subcommand |
| `test_cli_modules.py` | Modules subcommand |

**Commands Tested**:
- `hellm models list`
- `hellm models check`
- `hellm models sync`
- `hellm models setup`
- `hellm chat`
- `hellm benchmark`

### 3. Core Tests (`tests/core/`)

Tests for core module functionality:

| Test File | Coverage |
|-----------|----------|
| `test_client.py` | Client initialization |
| `test_config.py` | Configuration management |
| `test_prompts.py` | Prompt templates |
| `test_prompts_core.py` | Core prompt logic |

**Modules Tested**:
- `client.py` - API client setup
- `config.py` - Configuration loading
- `prompts.py` - Prompt management
- `export.py` - Result exporting

### 4. Integration Tests (`tests/integrations/`)

Tests for external service integrations:

| Test File | Coverage |
|-----------|----------|
| `test_cli_integrations.py` | CLI integration tests |
| `test_integrations.py` | Integration utilities |

**Services Tested**:
- LM-Eval integration
- Custom integration points

### 5. Provider Tests (`tests/providers/`)

Tests for model provider functionality:

| Test File | Coverage |
|-----------|----------|
| `test_litellm.py` | LiteLLM provider |
| `test_monitoring.py` | Monitoring utilities |
| `test_monitoring_core.py` | Monitoring core |
| `test_providers.py` | Provider base class |
| `test_token_limits.py` | Token limit handling |

**Providers Tested**:
- Blablador (primary provider)
- LiteLLM (aggregator)

### 6. Reporting Tests (`tests/reporting/`)

Tests for report generation and visualization:

| Test File | Coverage |
|-----------|----------|
| `test_reporting.py` | Report generation |
| `test_reporting_chart.py` | Chart generation |

**Reports Tested**:
- HTML reports
- Statistics calculations
- Performance charts

---

## Running Tests

### Run All Tests
```bash
poetry run pytest tests/
```

### Run Specific Test File
```bash
poetry run pytest tests/providers/test_providers.py
```

### Run Specific Test Class
```bash
poetry run pytest tests/providers/test_providers.py::TestBlabladorAPI
```

### Run Specific Test Method
```bash
poetry run pytest tests/providers/test_providers.py::TestBlabladorAPI::test_list_models_success
```

### With Coverage
```bash
poetry run pytest tests/ --cov=src/hellmholtz --cov-report=term-missing
```

---

## Adding New Tests

1. **Create test file** in appropriate subfolder
2. **Import modules** using absolute imports
3. **Use test conventions** (TestClassName, test_* methods)
4. **Add to appropriate coverage** (unit, integration, or end-to-end)
5. **Run tests** to verify

### Example

```python
# tests/cli/test_new_command.py
from hellmholtz.cli.new_command import new_command_impl

class TestNewCommand:
    def test_new_command_success(self):
        # Test successful execution
        result = new_command_impl("test")
        assert result is not None
    
    def test_new_command_with_invalid_input(self):
        # Test error handling
        with pytest.raises(ValueError):
            new_command_impl("")
```

---

## Coverage Strategy

| Target Coverage | Action |
|----------------|--------|
| 95%+ | Excellent - minimal gaps |
| 85-95% | Good - minor gaps |
| 70-85% | Needs work - add tests |
| <70% | High priority - add tests |

**Current Overall**: 82%

---

## Related Documentation

- [[docs/progress/test_reorganization_2026-08-13.md]] - Progress report
- [[docs/coverage/coverage_report_2026-08-13.md]] - Coverage details