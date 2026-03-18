# PyPI Publishing Workflow

This document describes the professional publishing workflow for the HeLLMholtz project using GitHub Actions and PyPI's Trusted Publishers (OIDC).

## Overview

The publishing workflow is automated using GitHub Actions with the following features:

- ✅ **Automated Testing**: Runs tests on Python 3.10, 3.11, and 3.12 before publishing
- ✅ **Code Quality Checks**: Linting, formatting, and type checking validation
- ✅ **Zero Credentials**: Uses OIDC Trusted Publishers (no API tokens in repository)
- ✅ **Multi-Environment Support**: Can publish to PyPI or TestPyPI
- ✅ **GitHub Release Integration**: Automatically uploads artifacts to releases

## Setup Instructions

### 1. Prerequisites

- Repository must be public (required for OIDC Trusted Publishers)
- Repository owner account with admin access to PyPI
- GitHub repository with write permissions

### 2. Configure Trusted Publishers on PyPI

The trusted publishers are already configured for this repository. To verify or reconfigure:

1. Go to [PyPI Account Settings - Publishing](https://pypi.org/manage/account/publishing/)
2. Look for **"HeLLMholtz"** project or add a new trusted publisher if needed
3. Configure for your repository:
   - **Project Name**: `hellmholtz`
   - **GitHub Repository**: `JonasHeinickeBio/HeLLMholtz`
   - **Workflow Name**: `publish.yml`
   - **Environment name**: `pypi` (optional but recommended)

### 3. Create GitHub Environments (Recommended)

For enhanced security, create dedicated GitHub environments:

1. Go to **Settings → Environments** in your GitHub repository
2. Create two environments:
   - **Environment name**: `pypi`
     - Deployment branches: Main branch only
     - Protected branch policy: Required
   - **Environment name**: `testpypi`
     - Deployment branches: Any branch
     - Optional protection rules

## Publishing Process

### Automatic Publishing (Recommended)

1. **Create a Release on GitHub**:
   ```bash
   # Create and push a tag
   git tag -a v0.2.1 -m "Release version 0.2.1"
   git push origin v0.2.1
   ```

2. **Create GitHub Release**:
   - Go to [Releases](https://github.com/JonasHeinickeBio/HeLLMholtz/releases)
   - Click "Draft a new release"
   - Select your tag
   - Add release notes
   - Click "Publish release"

3. **Workflow Triggers Automatically**:
   - The `publish.yml` workflow starts
   - Tests run on all Python versions
   - Package is built and published to PyPI
   - Artifacts are attached to the GitHub release

### Manual Publishing to TestPyPI

For testing the publishing process without affecting production:

```bash
# Trigger workflow manually
gh workflow run publish.yml -f target=testpypi
```

Or use the GitHub UI:
1. Go to **Actions → Publish to PyPI**
2. Click **Run workflow**
3. Select `testpypi` as target
4. Verify at [test.pypi.org](https://test.pypi.org/project/hellmholtz/)

## Workflow Details

### Jobs

#### `test`
- Runs on Python 3.10, 3.11, 3.12
- Executes: Ruff linting, MyPy type checking, Pytest coverage
- Fails immediately on any test failure
- **Required before publishing**

#### `build`
- Depends on: `test` job
- Builds wheel and source distributions
- Uploads artifacts for publishing
- Verifies package structure

#### `publish-to-pypi`
- **Triggers**: On GitHub release publication
- **Environment**: `pypi` (with OIDC Trusted Publisher)
- **Output**: Package available at https://pypi.org/project/hellmholtz/
- Uses GitHub token via OIDC for authentication

#### `publish-to-testpypi`
- **Triggers**: Manual workflow dispatch with `testpypi` selected
- **Environment**: `testpypi` (with OIDC Trusted Publisher)
- **Output**: Package available at https://test.pypi.org/project/hellmholtz/
- For testing only

#### `github-release-artifacts`
- **Triggers**: On GitHub release publication
- Uploads built distributions (.whl, .tar.gz) to release page
- Allows users to download artifacts directly from GitHub

## Environment Variables

The workflow uses these environment variables:

```yaml
PYTHON_VERSION: "3.12"  # Primary Python version for building
```

## Updating Package Metadata

To update PyPI project metadata, edit these files:

### `pyproject.toml` - Project Metadata
```toml
[project]
name = "hellmholtz"
version = "0.2.0"  # Update version before release
description = "Helmholtz LLM Suite"
keywords = ["llm", "benchmark", ...]  # Add relevant keywords

[project.urls]
Homepage = "https://github.com/..."
Repository = "https://github.com/..."
Documentation = "https://github.com/..."
Issues = "https://github.com/..."
```

### `README.md` - Project Description
- Used as long description on PyPI
- Keep updated with features, installation, usage examples

### `LICENSE` - License File
- Include LICENSE file in repository
- Reference in `pyproject.toml` as `license = { text = "MIT" }`

## Version Management

### Semantic Versioning

Follow [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

v0.2.1
├── 0: MAJOR version (breaking changes)
├── 2: MINOR version (features, backward compatible)
└── 1: PATCH version (bug fixes)
```

### Update Version

1. Update `version` in `pyproject.toml`:
   ```toml
   [project]
   version = "0.3.0"
   ```

2. Commit and create tag:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.3.0"
   git tag -a v0.3.0 -m "Release version 0.3.0"
   git push origin main v0.3.0
   ```

## Testing Before Release

### Local Build Testing

```bash
# Build locally
poetry build

# Test with TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ hellmholtz
```

### Verify Package Contents

```bash
# List contents of wheel
unzip -l dist/hellmholtz-*.whl

# List contents of sdist
tar -tzf dist/hellmholtz-*.tar.gz
```

### Manual TestPyPI Upload

```bash
# Build
poetry build

# Upload to TestPyPI (requires TestPyPI API token)
poetry publish -r testpypi
```

## Troubleshooting

### Publishing Failed

1. **Check workflow logs**: Go to **Actions → Publish to PyPI → Latest run**
2. **Common issues**:
   - ❌ **Test failures**: Fix failing tests before retrying
   - ❌ **Version conflict**: Version already exists on PyPI, increment version
   - ❌ **Trusted Publisher not configured**: Configure on PyPI account settings
   - ❌ **Artifacts not found**: Build may have failed, check build job logs

### OIDC Trusted Publisher Issues

1. **"Trusted publisher not found"**:
   - Verify GitHub repository path in PyPI settings
   - Verify workflow filename and environment name match
   - Ensure repository is public

2. **"Token generation failed"**:
   - Check GitHub Actions environment permissions
   - Verify `permissions: id-token: write` in workflow

## Post-Release Checklist

- [ ] Verify package on PyPI: https://pypi.org/project/hellmholtz/
- [ ] Test installation: `pip install hellmholtz`
- [ ] Check metadata displays correctly
- [ ] Update project website if needed
- [ ] Announce release (optional)
- [ ] Update changelog/release notes

## Additional References

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions: PyPI Publish](https://github.com/pypa/gh-action-pypi-publish)
- [Poetry Documentation](https://python-poetry.org/docs/pyproject/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

## Quick Commands

```bash
# Create and push release
git tag -a v0.3.0 -m "Release version 0.3.0"
git push origin v0.3.0

# Trigger manual TestPyPI publishing
gh workflow run publish.yml -f target=testpypi

# View workflow status
gh run list --workflow=publish.yml

# Check package on PyPI
python -m pip index versions hellmholtz
```

## Security Best Practices

- ✅ **Never commit API tokens** to the repository
- ✅ **Use OIDC Trusted Publishers** instead of personal API tokens
- ✅ **Require branch protection** for release workflows
- ✅ **Use dedicated GitHub environments** for PyPI access
- ✅ **Review tests** before every release
- ✅ **Sign tags** with GPG for verification (optional)

```bash
# Sign a tag with GPG
git tag -s -a v0.3.0 -m "Release version 0.3.0"

# Verify tag signature
git verify-tag v0.3.0
```
