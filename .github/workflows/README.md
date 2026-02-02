# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration, testing, benchmarking, and releases.

## Workflows Overview

### 1. Tests (`test.yml`)

Runs comprehensive testing across multiple Python versions and operating systems.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual trigger via workflow dispatch

**Jobs:**
- **test**: Runs pytest across Python 3.10, 3.11, 3.12 on Ubuntu, Windows, and macOS
  - Excludes network and integration tests by default
  - Generates code coverage reports
  - Uploads coverage to Codecov (requires `CODECOV_TOKEN` secret)
- **lint**: Runs ruff linter and formatter checks, plus mypy type checking
- **security**: Runs bandit security linter and uploads report as artifact

**Required Secrets:**
- `CODECOV_TOKEN` (optional): For uploading coverage reports to Codecov

### 2. Benchmarks (`benchmark.yml`)

Runs model benchmarking tests to compare performance across different LLM providers.

**Triggers:**
- Weekly schedule (Mondays at 00:00 UTC)
- Manual trigger via workflow dispatch with customizable parameters:
  - `models`: Comma-separated list of models to benchmark
  - `prompt_categories`: Categories to test (reasoning, coding, creative, or all)
  - `replications`: Number of replications per prompt

**Jobs:**
- **benchmark**: Runs the main benchmarking suite
  - Creates results and reports directories
  - Executes benchmarks with specified parameters
  - Uploads results as artifacts (90-day retention)
  - Adds summary to GitHub Actions summary page
- **throughput-test**: Tests throughput performance
  - Runs throughput-specific tests
  - Uploads results as artifacts (30-day retention)

**Required Secrets:**
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_API_KEY`: Google API key
- `BLABLADOR_API_KEY`: Helmholtz Blablador API key
- `BLABLADOR_API_BASE`: Helmholtz Blablador API base URL

### 3. Release (`release.yml`)

Automates package building and publishing to PyPI.

**Triggers:**
- Push of version tags matching `v*.*.*` (e.g., `v1.0.0`, `v2.1.3`)
- Manual trigger via workflow dispatch

**Jobs:**
- **build**: Builds the distribution packages
  - Runs tests before building
  - Creates wheel and source distributions
  - Uploads artifacts for other jobs
- **publish-pypi**: Publishes to PyPI
  - Uses trusted publishing (OIDC)
  - Requires PyPI environment configuration
- **github-release**: Creates GitHub release
  - Attaches distribution files
  - Extracts release notes from CHANGELOG.md if available
  - Generates automatic release notes
- **publish-test-pypi**: Publishes to TestPyPI (pre-releases only)
  - Triggers for alpha, beta, or rc tags

**Required Secrets:**
- `PYPI_API_TOKEN`: PyPI API token for publishing
- `TEST_PYPI_API_TOKEN`: TestPyPI API token for pre-release publishing

**Environment Setup:**
1. Create PyPI environment in repository settings
2. Configure trusted publisher on PyPI (recommended) or add API token
3. For TestPyPI, create separate environment and token

### 4. Pre-commit (`pre-commit.yml`)

Runs pre-commit hooks on code changes.

**Triggers:**
- Pull requests
- Push to `main` or `develop` branches

**Jobs:**
- **pre-commit**: Executes all pre-commit hooks
  - Runs ruff, mypy, bandit, and other checks
  - Auto-commits fixes on pull requests (if changes made)

### 5. Code Quality (`code-quality.yml`)

Additional code quality checks and security scanning.

**Triggers:**
- Pull requests
- Push to `main` or `develop` branches

**Jobs:**
- **dependency-review**: Reviews dependency changes in PRs
  - Fails on moderate or higher severity vulnerabilities
- **codeql**: Runs CodeQL security analysis
  - Scans for security vulnerabilities
  - Uploads results to GitHub Security tab
- **check-links**: Validates markdown links
  - Uses configuration from `.github/markdown-link-check-config.json`
- **coverage-report**: Generates coverage reports
  - Posts coverage comment on PRs
  - Shows coverage changes

## Dependabot (`dependabot.yml`)

Automates dependency updates for:
- GitHub Actions (weekly on Mondays)
- Python packages (weekly on Mondays)
- Groups patch updates together
- Adds appropriate labels to PRs

## Setup Instructions

### 1. Repository Secrets

Add the following secrets in repository settings (Settings → Secrets and variables → Actions):

**Required for Testing:**
- `CODECOV_TOKEN` (optional but recommended)

**Required for Benchmarking:**
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `BLABLADOR_API_KEY`
- `BLABLADOR_API_BASE`

**Required for Releases:**
- `PYPI_API_TOKEN`
- `TEST_PYPI_API_TOKEN`

### 2. PyPI Trusted Publishing (Recommended)

Instead of using API tokens, configure trusted publishing:

1. Go to PyPI account settings → Publishing
2. Add a new publisher:
   - Repository: `JonasHeinickeBio/HeLLMholtz`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. Create a `pypi` environment in GitHub repository settings
4. Remove `password` field from publish step and use `id-token: write` permission

### 3. Branch Protection Rules

Recommended branch protection for `main`:

1. Require pull request reviews
2. Require status checks to pass:
   - `test (ubuntu-latest, 3.12)`
   - `lint`
   - `security`
   - `pre-commit`
3. Require branches to be up to date
4. Include administrators

### 4. Codecov Integration

1. Sign up at [codecov.io](https://codecov.io)
2. Enable the repository
3. Copy the upload token
4. Add as `CODECOV_TOKEN` secret

## Usage Examples

### Running Tests Locally

```bash
# Install dependencies
poetry install --with dev

# Run all tests
poetry run pytest tests/ -v

# Run tests excluding network tests
poetry run pytest tests/ -v -m "not network and not integration"

# Run with coverage
poetry run pytest tests/ --cov=hellmholtz --cov-report=term-missing
```

### Running Benchmarks Locally

```bash
# Basic benchmark
poetry run hellm bench --models openai:gpt-4o-mini --replications 3

# Multiple models and categories
poetry run hellm bench \
  --models openai:gpt-4o,anthropic:claude-3-5-sonnet-20241022 \
  --prompt-categories reasoning,coding \
  --replications 5 \
  --save-markdown
```

### Creating a Release

```bash
# Update version in pyproject.toml
poetry version 1.0.0

# Create and push tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# The release workflow will automatically:
# 1. Run tests
# 2. Build package
# 3. Publish to PyPI
# 4. Create GitHub release
```

### Manual Workflow Triggers

You can manually trigger workflows from the Actions tab:

1. Go to Actions → Select workflow
2. Click "Run workflow"
3. Fill in parameters (for benchmark workflow)
4. Click "Run workflow" button

## Workflow Status Badges

Add these badges to your README.md:

```markdown
[![Tests](https://github.com/JonasHeinickeBio/HeLLMholtz/actions/workflows/test.yml/badge.svg)](https://github.com/JonasHeinickeBio/HeLLMholtz/actions/workflows/test.yml)
[![Code Quality](https://github.com/JonasHeinickeBio/HeLLMholtz/actions/workflows/code-quality.yml/badge.svg)](https://github.com/JonasHeinickeBio/HeLLMholtz/actions/workflows/code-quality.yml)
[![codecov](https://codecov.io/gh/JonasHeinickeBio/HeLLMholtz/branch/main/graph/badge.svg)](https://codecov.io/gh/JonasHeinickeBio/HeLLMholtz)
```

## Troubleshooting

### Test Failures

1. Check the workflow logs in the Actions tab
2. Look for failed test names and error messages
3. Run tests locally to reproduce: `poetry run pytest tests/ -v`

### Benchmark Failures

1. Verify API keys are correctly set in secrets
2. Check if models are available
3. Review timeout settings (currently 120 minutes)

### Release Failures

1. Ensure version in `pyproject.toml` matches tag
2. Verify PyPI token is valid
3. Check if version already exists on PyPI

### Coverage Not Uploading

1. Verify `CODECOV_TOKEN` is set correctly
2. Check if repository is enabled on Codecov
3. Review Codecov action logs for errors

## Maintenance

### Updating GitHub Actions

Dependabot automatically creates PRs for GitHub Actions updates. Review and merge them regularly.

### Updating Python Dependencies

Dependabot also handles Python dependency updates. Test thoroughly before merging.

### Workflow Modifications

When modifying workflows:

1. Test locally with [act](https://github.com/nektos/act) if possible
2. Validate YAML syntax
3. Create a feature branch and test via PR
4. Monitor first run carefully

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
