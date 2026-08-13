# Tests Directory Structure

**Date**: 2026-08-13
**Project**: helmholtz_llm_suite

## Overview

Tests are organized in subfolders that mirror the `src/hellmholtz/` structure.

## Directory Structure

```
tests/
├── __init__.py
├── benchmark/           # Benchmark and performance tests
│   ├── __init__.py
│   ├── test_blablador.py
│   ├── test_blablador_config.py
│   ├── test_blablador_provider.py
│   ├── test_benchmark.py
│   ├── test_config_converter.py
│   ├── test_evaluation_analysis.py
│   ├── test_export.py
│   ├── test_export_module.py
│   ├── test_exporters.py
│   ├── test_model_manager_core.py
│   ├── test_stats.py
│   └── test_throughput.py
├── cli/                 # CLI command tests
│   ├── __init__.py
│   ├── test_cli.py
│   ├── test_cli_benchmark.py
│   ├── test_cli_common.py
│   ├── test_cli_models.py
│   ├── test_cli_model_manager.py
│   └── test_cli_modules.py
├── core/                # Core module tests
│   ├── __init__.py
│   ├── test_client.py
│   ├── test_config.py
│   ├── test_prompts.py
│   └── test_prompts_core.py
├── integrations/        # Integration tests
│   ├── __init__.py
│   ├── test_cli_integrations.py
│   └── test_integrations.py
├── providers/           # Provider tests
│   ├── __init__.py
│   ├── test_litellm.py
│   ├── test_monitoring.py
│   ├── test_monitoring_core.py
│   ├── test_providers.py
│   └── test_token_limits.py
└── reporting/           # Reporting module tests
    ├── __init__.py
    ├── test_reporting.py
    └── test_reporting_chart.py
```

## Test Categories

### Benchmark Tests
Performance and benchmarking tests for the system.

### CLI Tests
Tests for command-line interface commands:
- `test_cli.py` - Generic CLI tests
- `test_cli_benchmark.py` - Benchmark command tests
- `test_cli_common.py` - Common CLI functionality
- `test_cli_models.py` - Models command tests
- `test_cli_model_manager.py` - Model manager tests
- `test_cli_modules.py` - Module management tests

### Core Tests
Tests for core modules:
- `test_client.py` - Client functionality
- `test_config.py` - Configuration management
- `test_prompts.py` - Prompt handling
- `test_prompts_core.py` - Core prompt functionality

### Integration Tests
End-to-end integration tests:
- `test_cli_integrations.py` - CLI integration flows
- `test_integrations.py` - General integration tests

### Provider Tests
Tests for model providers:
- `test_litellm.py` - LiteLLM provider tests
- `test_monitoring.py` - Monitoring functionality
- `test_monitoring_core.py` - Core monitoring tests
- `test_providers.py` - Provider tests
- `test_token_limits.py` - Token limit tests

### Reporting Tests
Tests for reporting functionality:
- `test_reporting.py` - General reporting tests
- `test_reporting_chart.py` - Chart generation tests

## Test Naming Conventions

- Files start with `test_` prefix
- Classes start with `Test` prefix (PascalCase)
- Methods start with `test_` prefix (snake_case)

## Running Tests

```bash
# Run all tests
poetry run pytest tests/

# Run specific test file
poetry run pytest tests/cli/test_cli_models.py

# Run specific test class
poetry run pytest tests/cli/test_cli_models.py::TestModelsList

# Run with coverage
poetry run pytest tests/ --cov=src/hellmholtz --cov-report=term-missing

# Run with verbose output
poetry run pytest tests/ -v

# Run with traceback on
poetry run pytest tests/ --tb=short
```

## Test Dependencies

Tests use:
- `pytest` - Testing framework
- `unittest.mock` - Mocking utilities
- `httpx` - HTTP client for mocking

## CI/CD Integration

Tests are run automatically on:
- Pull requests
- Merge to main branch
- Scheduled runs (daily)
