# Lesson: Continuous Integration with GitHub Actions

## Introduction

**Why CI/CD matters:**
- **Automated quality checks** ensure code quality before merging
- **Consistent environments** - tests run in the same environment every time
- **Early detection** of bugs and code quality issues
- **Team confidence** - everyone knows the code meets standards
- **Prevents broken code** from reaching the main branch

**What we'll cover:**
- Setting up GitHub Actions workflow
- Running linting, formatting, and type checking
- Ensuring consistent import sorting across environments
- Running tests automatically
- Troubleshooting common CI issues

---

## Part 1: Understanding GitHub Actions

### What is GitHub Actions?

GitHub Actions is a CI/CD platform that allows you to automate workflows directly in your GitHub repository. Workflows are defined in YAML files stored in `.github/workflows/`.

### Basic Workflow Structure

A GitHub Actions workflow consists of:
- **Triggers**: When the workflow runs (push, pull request, etc.)
- **Jobs**: Sets of steps that run on the same runner
- **Steps**: Individual commands or actions
- **Runners**: Virtual machines that execute the workflow

---

## Part 2: Setting Up Your First GitHub Actions Workflow

### Step 1: Create Workflow Directory

On GitHub, select Python application workflow

Change Python version to 3.12, actions/setup-python@v5


```yaml
name: Python application

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12.12'
    
    - name: Install uv via pip
      run: pip install uv

    - name: Sync dependencies
      run: uv sync --dev

    - name: Lint with Ruff
      run: uv run ruff check .

    - name: Enforce formatting with Ruff
      run: uv run ruff format --check .

    - name: Type checking
      run: uv run ty check src tests

    - name: Test with pytest
      run: uv run pytest
```

### Step 3: Understanding the Workflow

Let's break down each part:

**Workflow name and triggers:**
```yaml
name: Python application
on: [push, pull_request]
```
- The workflow runs on every push and pull request

**Job definition:**
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
```
- Defines a job named `build` that runs on Ubuntu

**Checkout code:**
```yaml
- uses: actions/checkout@v4
```
- Checks out your repository code

**Set up Python:**
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.12.12'
```
- Installs the specified Python version

**Install uv:**
```yaml
- name: Install uv via pip
  run: pip install uv
```
- Installs uv package manager

**Sync dependencies:**
```yaml
- name: Sync dependencies
  run: uv sync --dev
```
- Installs all project dependencies (including dev dependencies)

**Quality checks:**
```yaml
- name: Lint with Ruff
  run: uv run ruff check .
```
- Runs linting checks

```yaml
- name: Enforce formatting with Ruff
  run: uv run ruff format --check .
```
- Checks that code is properly formatted (doesn't modify, just checks)

```yaml
- name: Type checking
  run: uv run ty check src tests
```
- Runs type checking

**Run tests:**
```yaml
- name: Test with pytest
  run: uv run pytest
```
- Runs all tests

### Step 4: Commit and Push

Commit the workflow file:

```bash
git add .github/workflows/python-app.yml
git commit -m "Add GitHub Actions CI workflow"
git push
```

### Step 5: View Workflow Results

1. Go to your GitHub repository
2. Click on the "Actions" tab
3. You'll see your workflow running
4. Click on a workflow run to see detailed results

---

## Part 3: Ensuring Consistent Import Sorting

### The Problem: Different Sorting Results

A common issue with CI is that import sorting results differ between local and CI environments, even with the same versions of ruff and uv. This happens because:

1. **File processing order** can vary between systems
2. **Default isort settings** may not be explicit enough
3. **Package classification** (first-party vs third-party) can be ambiguous

### Solution: Explicit isort Configuration

Add explicit isort configuration to your `pyproject.toml` to ensure deterministic sorting:

```toml
[tool.ruff.lint.isort]
known-first-party = ["ml_app_wind_draft"]
# Force mlflow to be grouped with the other third-party libraries
known-third-party = ["mlflow"]
# Ensures a consistent category order
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
```

### Understanding the Configuration

**`known-first-party`:**
- Explicitly tells ruff which packages belong to your project
- Ensures imports from `ml_app_wind_draft` are correctly classified as first-party
- Prevents misclassification issues

**`known-third-party`:**
- Explicitly classifies specific packages as third-party
- Useful for packages like `mlflow` that might be ambiguous
- Ensures consistent grouping with other third-party libraries

**`section-order`:**
- Defines the exact order of import sections
- Ensures imports are always sorted in the same order:
  1. Future imports (e.g., `from __future__ import annotations`)
  2. Standard library imports (e.g., `import os`, `from typing import List`)
  3. Third-party imports (e.g., `import pandas`, `from sklearn import ...`)
  4. First-party imports (e.g., `from ml_app_wind_draft import ...`)
  5. Local folder imports (relative imports)

### Step 1: Add Configuration to pyproject.toml

Add the isort configuration section to your `pyproject.toml`:

```toml
[tool.ruff.lint.isort]
known-first-party = ["ml_app_wind_draft"]
# Force mlflow to be grouped with the other third-party libraries
known-third-party = ["mlflow"]
# Ensures a consistent category order
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
```

### Step 2: Apply New Sorting Rules Locally

After adding the configuration, run:

```bash
ruff check --fix .
```

This will apply the new sorting rules to all files.

### Step 3: Commit Changes

Commit both the configuration and the re-sorted files:

```bash
git add pyproject.toml
git add .
git commit -m "Configure ruff isort for consistent import sorting"
git push
```

### Step 4: Verify CI Consistency

After pushing, check that:
1. CI passes with the new configuration
2. Local `ruff check .` matches CI results
3. No import sorting differences between environments

---

## Part 4: Workflow Best Practices

### 1. Pin uv Version (Optional)

For maximum reproducibility, you can pin the uv version:

```yaml
- name: Install uv via pip
  run: pip install uv==0.9.18
```

**Note:** Pinning versions ensures consistency but requires manual updates. Using `pip install uv` gets the latest version automatically.

### 2. Cache Dependencies

Speed up workflows by caching dependencies:

```yaml
- name: Cache uv dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: ${{ runner.os }}-uv-${{ hashFiles('**/uv.lock') }}
    restore-keys: |
      ${{ runner.os }}-uv-
```

### 3. Run Quality Checks in Parallel

You can run multiple checks in parallel using a matrix strategy, but for most projects, sequential execution is simpler and easier to debug.

### 4. Add Status Badges

Add a status badge to your README:

```markdown
![CI](https://github.com/yourusername/yourrepo/workflows/Python%20application/badge.svg)
```

### 5. Fail Fast

The default behavior (failing on first error) is usually best. Each step fails if the command returns a non-zero exit code, stopping the workflow immediately.

---

## Part 5: Troubleshooting Common Issues

### Problem 1: Import Sorting Differences

**Symptoms:**
- `ruff check .` passes locally but fails in CI
- Different import order between local and CI

**Solutions:**
1. Ensure you have the explicit isort configuration (as shown in Part 3)
2. Verify ruff versions match: `uv run ruff --version` (both local and CI)
3. Clear ruff cache: `rm -rf .ruff_cache`
4. Re-apply sorting: `ruff check --fix .`

### Problem 2: Dependency Installation Fails

**Symptoms:**
- `uv sync --dev` fails in CI
- Package not found errors

**Solutions:**
1. Check that `uv.lock` file is committed to the repository
2. Verify all dependencies are specified in `pyproject.toml`
3. Ensure Python version matches: check `requires-python` in `pyproject.toml`

### Problem 3: Tests Pass Locally but Fail in CI

**Symptoms:**
- Tests work on your machine but fail in CI
- Random test failures

**Solutions:**
1. Check for hardcoded paths (use relative paths)
2. Ensure random seeds are set for reproducible tests
3. Check for OS-specific differences (Windows vs Linux vs macOS)
4. Verify environment variables are set in CI if needed

### Problem 4: Type Checking Fails

**Symptoms:**
- `ty check` fails in CI
- Type errors that don't appear locally

**Solutions:**
1. Ensure `pyproject.toml` has correct `[tool.ty.src]` configuration
2. Verify type checking paths match: `ty check src tests`
3. Check for version differences in type stubs

### Problem 5: Workflow Doesn't Run

**Symptoms:**
- Workflow file exists but doesn't trigger
- No "Actions" tab visible

**Solutions:**
1. Verify file is in `.github/workflows/` directory
2. Check YAML syntax is valid (no indentation errors)
3. Ensure workflow file has `.yml` or `.yaml` extension
4. Check that GitHub Actions is enabled for your repository

---

## Part 6: Advanced Configuration

### Running Tests with Coverage

Add coverage reporting:

```yaml
- name: Test with pytest and coverage
  run: uv run pytest --cov=src/ml_app_wind_draft --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Matrix Testing (Multiple Python Versions)

Test against multiple Python versions:

```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12']
steps:
  - name: Set up Python
    uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
```

### Conditional Steps

Run steps conditionally:

```yaml
- name: Deploy to staging
  if: github.ref == 'refs/heads/main'
  run: echo "Deploying..."
```

### Environment Variables

Set environment variables:

```yaml
env:
  PYTHONPATH: src
steps:
  - name: Run tests
    run: uv run pytest
```

---

## Summary

**What we've covered:**

1. **GitHub Actions basics** - understanding workflows, jobs, and steps
2. **Setting up CI** - creating a workflow file for automated quality checks
3. **Import sorting** - configuring ruff isort for consistent results
4. **Best practices** - caching, versioning, and optimization
5. **Troubleshooting** - common issues and solutions
6. **Advanced features** - coverage, matrix testing, conditionals

**Key takeaways:**

- CI/CD automates quality checks and prevents broken code from merging
- Explicit configuration (especially for isort) ensures consistent behavior
- Regular workflow runs catch issues early
- Consistent environments between local and CI reduce debugging time

**Benefits:**

- ✅ Automated quality assurance
- ✅ Consistent code standards
- ✅ Early bug detection
- ✅ Team confidence in code quality
- ✅ Reduced manual review time

---

## Next Steps

1. **Customize your workflow** - Add more checks specific to your project
2. **Set up deployment** - Add deployment steps for staging/production
3. **Add notifications** - Configure Slack/email notifications for failures
4. **Optimize performance** - Add caching and parallel jobs for faster runs
5. **Security scanning** - Add security vulnerability scanning

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ruff isort Settings](https://docs.astral.sh/ruff/settings/#isort)
- [uv Documentation](https://docs.astral.sh/uv/)
- [pytest Documentation](https://docs.pytest.org/)

