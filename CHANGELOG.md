# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive GitHub Actions workflows for CI/CD
  - Testing workflow with multi-OS and multi-Python version support
  - Benchmarking workflow with scheduled and manual triggers
  - Release workflow with PyPI publishing
  - Pre-commit workflow for code quality
  - Code quality workflow with security scanning
- Dependabot configuration for automated dependency updates
- Markdown link checking in CI
- CodeQL security analysis
- Comprehensive workflow documentation in `.github/workflows/README.md`

### Changed
- Updated README.md with workflow status badges
- Enhanced CONTRIBUTING.md with CI/CD workflow information

## [0.1.0]

### Added
- Initial release of HeLLMholtz
- Unified LLM client supporting OpenAI, Anthropic, Google, Ollama, and Helmholtz Blablador
- Comprehensive benchmarking suite with configurable parameters
- LLM-as-a-Judge evaluation system
- HTML and Markdown report generation
- Model monitoring and availability checking
- LM Evaluation Harness integration
- LiteLLM proxy server support
- Throughput testing capabilities
- CLI interface with multiple commands
- Type hints and comprehensive documentation

[Unreleased]: https://github.com/JonasHeinickeBio/HeLLMholtz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JonasHeinickeBio/HeLLMholtz/releases/tag/v0.1.0
