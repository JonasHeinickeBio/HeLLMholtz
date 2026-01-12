# Contributing to HeLLMholtz

We welcome contributions from the community! This document provides guidelines for contributing to the HeLLMholtz project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- Git

### Setup

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/HeLLMholtz.git
   cd HeLLMholtz
   ```

2. **Install dependencies**:
   ```bash
   poetry install --with dev
   ```

3. **Install pre-commit hooks**:
   ```bash
   poetry run pre-commit install
   ```

4. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys for testing
   ```

## Development Workflow

### 1. Choose an Issue

- Check the [issue tracker](https://github.com/JonasHeinickeBio/HeLLMholtz/issues) for open issues
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to indicate you're working on it

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Write clear, concise commit messages
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 4. Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=hellmholtz --cov-report=html

# Run specific tests
poetry run pytest tests/test_client.py

# Run tests matching a pattern
poetry run pytest -k "test_chat"
```

### 5. Code Quality

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

### 6. Update Documentation

- Update README.md if adding new features
- Update docstrings for new functions/classes
- Add examples for new functionality
- Update type hints

### 7. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: add new monitoring functionality

- Add ModelAvailabilityMonitor class
- Add CLI commands for model monitoring
- Add comprehensive tests
- Update documentation"

# For fixes
git commit -m "fix: resolve issue with model listing

- Fix timeout handling in API calls
- Add retry logic for failed requests"
```

### 8. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
```

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all function parameters and return values
- Write docstrings for all public functions, classes, and modules
- Use descriptive variable names
- Keep functions small and focused

### Commit Messages

Follow the [Conventional Commits](https://conventionalcommits.org/) format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Testing

- Write unit tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

### Documentation

- Update README.md for new features
- Add docstrings to all public APIs
- Include usage examples
- Keep documentation up to date

## CI/CD Workflows

The project uses GitHub Actions for continuous integration and deployment. When you create a pull request, the following workflows will automatically run:

### Automated Checks

1. **Tests Workflow** (`test.yml`):
   - Runs pytest across Python 3.10, 3.11, 3.12
   - Tests on Ubuntu, Windows, and macOS
   - Generates code coverage reports
   - Must pass before merging

2. **Code Quality Workflow** (`code-quality.yml`):
   - Dependency review for security vulnerabilities
   - CodeQL security analysis
   - Markdown link checking
   - Coverage reporting with PR comments

3. **Pre-commit Workflow** (`pre-commit.yml`):
   - Runs all pre-commit hooks
   - Auto-fixes formatting issues
   - Commits fixes automatically

4. **Linting and Security**:
   - Ruff linter and formatter checks
   - mypy type checking
   - Bandit security scanning

### Viewing Workflow Results

- Go to the "Actions" tab in the pull request
- Click on a workflow run to see details
- Review any failures and fix issues
- Push new commits to re-trigger checks

### Local Testing Before Push

To avoid CI failures, run these commands locally:

```bash
# Run all pre-commit hooks
poetry run pre-commit run --all-files

# Run tests with the same filters as CI
poetry run pytest tests/ -v -m "not network and not integration"

# Check code coverage
poetry run pytest --cov=hellmholtz --cov-report=term-missing

# Type checking
poetry run mypy src/

# Security check
poetry run bandit -r src/
```

## Pull Request Process

1. **Ensure CI passes**: All tests must pass, code must be formatted and linted
2. **Update documentation**: Ensure docs reflect your changes
3. **Squash commits**: Keep the git history clean
4. **Write a clear PR description**:
   - What changes were made
   - Why they were needed
   - How to test the changes
   - Any breaking changes
5. **Wait for review**: Maintainers will review your PR and provide feedback
6. **Address feedback**: Make requested changes and push new commits

## Areas for Contribution

### High Priority
- Bug fixes
- Performance improvements
- Additional model providers
- Enhanced error handling

### Medium Priority
- New evaluation metrics
- Additional benchmark datasets
- CLI improvements
- Documentation enhancements

### Low Priority
- UI/UX improvements
- Additional integrations
- Research features

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Discord**: Join our community Discord for real-time help

## License

By contributing to HeLLMholtz, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to HeLLMholtz! 🚀
