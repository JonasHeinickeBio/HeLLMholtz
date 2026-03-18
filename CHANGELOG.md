# Changelog

All notable changes to HeLLMholtz will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-18

### Fixed

- Fixed type annotations in benchmark scripts (`generate_comprehensive_report.py`, `generate_chart.py`)
- Fixed blanket noqa comments in monitoring module
- Corrected Poetry build command in GitHub Actions workflow (removed invalid `--no-directory` flag)
- Resolved mypy type checking discrepancies between local and CI runs

## [0.2.0] - 2026-03-18

### Added

- **Unified LLM Client**: Single interface for OpenAI, Google, Anthropic, Ollama, and Helmholtz Blablador models
- **CLI Tool**: Command-line interface `hellm` with subcommands for chat, benchmarking, monitoring, and reporting
- **Benchmarking System**: Compare model performance across temperatures, replications, and prompt categories
- **LLM-as-a-Judge Evaluation**: Automated evaluation with comprehensive statistical analysis
- **Interactive Reports**: HTML reports with Chart.js visualizations and Markdown summaries
- **Model Monitoring**: Track Blablador model availability and configuration consistency
- **Throughput Testing**: Performance benchmarking for high-throughput scenarios
- **LM Evaluation Harness Integration**: Support for running academic benchmarks (HellaSwag, Winogrande, MMLU, etc.)
- **LiteLLM Proxy**: Built-in proxy server for model routing and load balancing
- **Flexible Prompt System**: Support for both simple text files and structured JSON prompt collections
- **Token Usage Tracking**: Automatic tracking of input/output tokens in benchmark results
- **GitHub Actions Workflow**: Automated weekly benchmarking with report generation and updates

### Changed

- Updated pyproject.toml with comprehensive project metadata and documentation
- Improved error handling and timeout configuration across all providers
- Enhanced configuration management with environment-based settings

### Documentation

- Comprehensive README with installation, usage, and feature documentation
- Split documentation into organized sections (Installation, Configuration, Usage, Models, Monitoring)
- Added usage examples for both CLI and Python API
- Documented token limits for all supported models
- Added monitoring documentation with health check examples

### Configuration

- Added BLABLADOR_API_KEY and BLABLADOR_API_BASE environment variables
- Added HELMHOLTZ_TIMEOUT_SECONDS configuration
- Added AISUITE_DEFAULT_MODELS for default model selection
- Updated .env.example with comprehensive configuration options

### Dependencies

- Base dependency on aisuite for unified LLM access
- Optional dependencies:
  - `eval`: lm-eval for academic benchmarking
  - `proxy`: litellm for proxy server functionality
  - `reporting`: matplotlib, scipy, seaborn for visualization

## [0.1.0] - 2026-01-12

### Initial Release

- Basic LLM client functionality
- Simple chat interface
- Minimal benchmarking support
- Basic monitoring capabilities
