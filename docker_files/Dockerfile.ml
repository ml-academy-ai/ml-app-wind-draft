FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for compiling packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy entire project (needed for uv sync to read version from __init__.py)
COPY . .

# Install dependencies using uv sync
# Main dependencies already include all ML packages (kedro, mlflow, catboost, etc.)
# The --extra ml would add ml extras if defined, but main deps are sufficient
RUN uv sync --frozen

# Set Python path and ensure venv is in PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "--version"]

