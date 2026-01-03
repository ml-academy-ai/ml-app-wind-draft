# Lesson 1: Project Setup with Kedro

## Step 1: Install uv

Install uv using the standalone installer:
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

## Step 2: Install Kedro

Run:
```bash
uvx kedro new
```

**Project name:** `ml-app-wind-draft` (use only lower letters)

**Select option 1-5** (because we will not use pyspark)

**No example pipeline** - select "No" when asked

## Step 3: Set Up Virtual Environment

Go to the IDE and `cd` to your project directory.

Next, create a new virtual environment inside the `.venv` subdirectory:
```bash
uv venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Step 4: Configure Python Version

Change in `pyproject.toml`:
```toml
requires-python = ">=3.11,<3.13"
```

Then recreate the virtual environment:
```bash
rm -rf .venv
uv venv .venv
```

Run:
```bash
uv sync
```
(This might take longer the first time)

## Step 5: Add Required Packages

Add all the packages we need:
```bash
uv add \
  kedro-datasets[csv,pandas]==9.0.0 \
  mlflow==3.6.0 \
  scikit-learn==1.4.0 \
  catboost==1.2.8 \
  optuna==4.3.0 \
  torch==2.2.0 \
  torchaudio==2.2.0 \
  torchvision==0.17.0 \
  shap<0.50 \
  numpy==1.26.4 \
  pandas==2.2.2 \
  pyarrow==20.0.0 \
  pyyaml==6.0.2 \
  python-dateutil==2.8.2 \
  ydata-profiling==4.16.1 \
  matplotlib==3.8.2 \
  seaborn==0.13.1
```

## Step 6: Configure Interpreter

In your IDE (PyCharm/VSCode), configure the Python interpreter to use `.venv/bin/python`.

## Step 7: Copy Data

Copy your data files to `data/01_raw/` directory.

## Step 8: Create EDA Notebook

Create your first EDA notebook (e.g., `notebooks/01 - EDA.ipynb`).

## Overview of the Package After Creation

### Understanding egg-info

In `src/`, you can find `egg-info` directory.

**What is egg-info?**
- `egg-info` is a directory created by setuptools during package installation
- It contains metadata about your Python package
- You can safely delete it

### What is setuptools?

**Setuptools** is a Python library that helps you package, build, and distribute your Python code. It's the tool that makes it possible to install Python packages using `pip`.

#### What does it do?

Think of setuptools as the "packaging system" for Python. When you want to share your code or make it installable, setuptools helps you:

- **Package your code**: Organizes your Python files into a format that can be easily installed
- **Define dependencies**: Specifies what other packages your project needs (like pandas, numpy, etc.)
- **Make it installable**: Allows others (or yourself) to install your project using `pip install`

#### Why is it in your Kedro project?

When Kedro creates your project, it uses setuptools to:
- Make your project code importable (so you can use `from ml_app_wind_draft.pipelines...`)
- Allow you to install your project in "editable mode" with `pip install -e .`
- Manage all the dependencies listed in your `pyproject.toml` file

**In simple terms**: Setuptools is the behind-the-scenes tool that makes your Python project work like a proper package, so you can import and use your code easily.

