# Lesson 2: Python Code Quality

## Introduction

**Why code quality matters:**
- **Automated checks reduce the amount of time the code review is done**
- **You make sure that your code quality is high**
- **It shows to the code reviewers that your code follows the rules of development and is ready for a review**

---

## Part 1: Linting with Ruff

### Step 1: Install Ruff

If not installed, run:
```bash
uv add ruff
```

### Step 2: Check pyproject.toml

Verify that ruff is added to your dependencies.

### Step 3: Run Ruff Check

Run:
```bash
ruff check .
```

**What to expect:**
- It can find A LOT of errors even though we tried to keep our code clean, add typings, remove unused imports, etc.
- However, it covered everything including notebooks

### Step 4: Exclude Notebooks

**Why exclude notebooks?**
- In notebooks, code quality is less important because it does not go to production
- Notebooks are for exploration and experimentation

**To exclude notebooks:**

In `pyproject.toml`, add to the existing `[tool.ruff]` key:
```toml
[tool.ruff]
extend-exclude = [
    "*.ipynb",  # Excludes all Jupyter Notebooks
    "notebooks/",
]
```

Now, we have many fewer errors.

### Step 5: Check Individual Files

Let's run it for one file - utils file in the training pipelines directory.

**Ask Cursor how to run it** (or use IDE integration).

Discuss some errors:
- Import sorting
- Unused imports
- Type hints
- Code style

### Step 6: Auto-fix Errors

Fix the errors. Run:
```bash
ruff check . --fix
```

This also fixes import sorting automatically.

### Step 7: Check Remaining Errors

Check when errors still exist. It might not fix errors that it **CANNOT fix without intervention**. Resolve or ignore these manually.

### Step 8: Run on Entire Codebase

Now, let's run the whole code database:
```bash
ruff check . --fix
```

**Note:** White spaces removed in formatting but `ruff check .` reports it.

---

## Part 2: Formatting with Ruff

### Step 1: Check What Would Be Formatted

First, run:
```bash
ruff format --check .
```

This shows what would be formatted (but not formatted yet).

### Step 2: Format Code

To format, run:
```bash
ruff format .
```

### Step 3: Verify

Then run:
```bash
ruff check .
```

(Should be all fixed)

---

## Part 3: Type Checking with ty

### Step 1: Install ty

Run:
```bash
uv add ty
```

**Note:** `ty` is created by the ruff creators, so it integrates well with ruff.

### Step 2: Run Type Check

Run:
```bash
ty check
```

### Step 3: Configure ty

Add to `pyproject.toml`:
```toml
[tool.ty.src]
include = ["src", "tests", "entrypoint"]
```

### Step 4: Demonstrate Type Errors

If there are not many mistakes, change types in `compute_metrics()` and demonstrate:
- The error
- How to fix it

### Step 5: Handle MLflow Type Errors

For mlflow errors, add:
```toml
[[tool.ty.overrides]]
include = ["**/pipelines/**"]
rules = { "possibly-missing-attribute" = "ignore" }
```

This ignores MLflow's dynamic attribute access that type checkers can't understand.

---

## Part 4: Pre-commit Framework

### Introduction

**Problem:** Now we know how to run Ruff and Ty manually. The problem is: **humans forget.**

**Solution:** `pre-commit` framework solves this by running the same checks automatically before a commit is created.

### Step 1: Install Pre-commit

Let's install pre-commit framework:
```bash
uv add pre-commit
```

### Step 2: Create Pre-commit Config

Create a file named `.pre-commit-config.yaml`

### Step 3: Install Git Hook

Run:
```bash
pre-commit install
```

This installs the Git hook.

### Step 4: Verify Hook Installation

Run:
```bash
ls -l .git/hooks/pre-commit
```

To see if the hook is installed. It's added to the hidden `.git` file (at least for Mac), so we will not see it in the Editor.

### Step 5: Configure Ruff for Pre-commit

Google "pre-commit ruff" and take the file from the git repo of ruff-pre-commit.

**What is nice:** The pre-commit-config is already compatible with `pyproject.toml`, so we can keep our setup and just add vanilla config for pre-commit.

### Step 6: Test Pre-commit

Now, we can run:
```bash
pre-commit run --all-files
```

### Step 7: Demonstrate Auto-fix on Commit

Now, if we change some imports sorting and then commit changes, it will:
- Automatically fix the issues it can fix
- Leave the issues it cannot fix (for manual resolution)

### Step 8: Demonstrate Type Checking

Introduce some typing errors (e.g., numpy and `pd.Series` function returns) and show how it works.

**But if we change typings, it does not work**, because we need to add `ty`.

### Step 9: Add ty to Pre-commit

Google "how to add ty to pre-commit". Add the config.

Now pre-commit will:
- Run ruff (linting and formatting)
- Run ty (type checking)
- All automatically before each commit

---

## Summary

**Tools covered:**
1. **Ruff**: Linting and formatting
2. **ty**: Type checking
3. **pre-commit**: Automated quality checks

**Benefits:**
- Consistent code style
- Catch errors early
- Reduce code review time
- Professional development workflow

