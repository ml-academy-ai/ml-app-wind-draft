# Lesson 0: Directory Setup and UV Installation

## Overview

Before starting any Python project, you need to:
1. Create a project directory
2. Install a Python package manager (`uv`)
3. Verify the installation

This lesson covers the initial setup steps that are prerequisites for all subsequent lessons.

## Step 1: Create Project Directory

First, decide where you want to create your project. Common locations include:
- `~/Projects/` (on macOS/Linux)
- `~/Documents/Projects/` 
- `C:\Users\YourName\Projects\` (on Windows)

Create a new directory for your project:

```bash
mkdir ml-app-wind-draft
cd ml-app-wind-draft
```

**Note:** Use lowercase letters and hyphens for directory names to follow Python naming conventions.

## Step 2: Install uv

`uv` is a fast Python package installer and resolver written in Rust. It's designed to be a drop-in replacement for `pip` and `pip-tools`, but much faster.

### Installation Methods

#### Option 1: Standalone Installer (Recommended)

Visit the official installation page:
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

**For macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**For Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Option 2: Using pip

If you already have Python and pip installed:
```bash
pip install uv
```

#### Option 3: Using Homebrew (macOS)

```bash
brew install uv
```

### Post-Installation

After installation, you may need to:
- Restart your terminal
- Add `uv` to your PATH (usually done automatically)
- Verify installation (see Step 3)

## Step 3: Verify Installation

Check that `uv` is installed correctly:

```bash
uv --version
```

You should see output like:
```
uv 0.x.x
```

Test basic functionality:

```bash
uv --help
```

This should display the help menu with available commands.

## Step 4: Create Virtual Environment

Now that `uv` is installed, create a virtual environment for your project:

```bash
uv venv .venv
```

This creates a virtual environment in the `.venv` directory.

### Activate the Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt, indicating the virtual environment is active.

### Verify Virtual Environment

Check that Python is using the virtual environment:

```bash
which python
```

On macOS/Linux, this should show a path ending in `.venv/bin/python`. On Windows, use `where python`.

## Step 5: Initialize Project with pyproject.toml

Initialize your project with a `pyproject.toml` file:

```bash
uv init
```

This command will:
- Create a `pyproject.toml` file with basic project metadata
- Optionally create a `README.md` file
- Optionally initialize a git repository (if git is available)

The `pyproject.toml` file is the modern standard for Python project configuration and will be used to manage your dependencies with `uv`.

## Step 6: Create Basic Project Structure

Create directories for your EDA work:

```bash
mkdir notebooks
mkdir data
mkdir data/01_raw
```

This creates:
- `notebooks/` - for your Jupyter notebooks
- `data/01_raw/` - for raw data files

## Step 7: Add Data Files

Copy your data files to the `data/01_raw/` directory. This is where all raw, unprocessed data should be stored.

**Example:**
```bash
# If your data files are in another location
cp /path/to/your/data/*.parquet data/01_raw/

# Or copy specific files
cp /path/to/your/data/df_train.parquet data/01_raw/
cp /path/to/your/data/df_test.parquet data/01_raw/
```

**Or manually:**
- Navigate to the `data/01_raw/` folder in your file explorer
- Copy and paste your data files (`.parquet`, `.csv`, etc.) into this directory

**Important:** Make sure your data files are placed in `data/01_raw/` before starting your EDA work, as your notebooks will reference this location.

## Step 8: Install Required Packages

Install all the packages needed for your project:

```bash
uv add \
  jupyter \
  notebook \
  ipykernel \
  kedro-datasets[csv,pandas]==9.0.0 \
  mlflow==3.6.0 \
  scikit-learn==1.4.0 \
  catboost==1.2.8 \
  optuna==4.3.0 \
  torch==2.2.0 \
  torchaudio==2.2.0 \
  torchvision==0.17.0 \
  "shap<0.50" \
  numpy==1.26.4 \
  pandas==2.2.2 \
  pyarrow==20.0.0 \
  pyyaml==6.0.2 \
  python-dateutil==2.8.2 \
  ydata-profiling==4.16.1 \
  matplotlib==3.8.2 \
  seaborn==0.13.1
```

**Note:** This installation may take several minutes, especially for PyTorch packages.

## What's Next?

Once you have:
- ✅ A project directory created
- ✅ `uv` installed and verified
- ✅ Virtual environment created and activated
- ✅ Project initialized with `pyproject.toml`
- ✅ Basic project structure (notebooks and data directories)
- ✅ Data files copied to `data/01_raw/`
- ✅ All required packages installed

## Troubleshooting

### uv command not found

If you get a "command not found" error:
1. **Check PATH:** Ensure the installation directory is in your PATH
2. **Restart terminal:** Close and reopen your terminal
3. **Manual PATH addition:** Add the installation directory to your shell profile:
   - macOS/Linux: `~/.bashrc` or `~/.zshrc`
   - Windows: System Environment Variables

## Key Concepts

### What is uv?

- **Fast package manager:** Written in Rust, significantly faster than pip
- **Dependency resolver:** Handles complex dependency resolution efficiently
- **Virtual environment manager:** Can create and manage virtual environments
- **Project manager:** Supports `pyproject.toml` and modern Python packaging standards

### Why use uv?

1. **Speed:** 10-100x faster than pip for many operations
2. **Modern:** Built for modern Python packaging standards
3. **Reliable:** Better dependency resolution and conflict detection
4. **Compatible:** Drop-in replacement for many pip workflows