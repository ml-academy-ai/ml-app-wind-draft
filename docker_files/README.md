# Optimized Docker Setup

This directory contains optimized Dockerfiles that install only the dependencies needed for each service, significantly reducing build time and image sizes.

## Files

- `Dockerfile.ml` - For ML services (training, inference, kedro-viz) - includes Kedro, MLflow, CatBoost, scikit-learn, etc.
- `Dockerfile.mlflow` - Minimal MLflow server - only MLflow and pyyaml
- `Dockerfile.data` - Data streaming service - only pandas and pyyaml
- `Dockerfile.ui` - UI service - Dash, MLflow (for model info), plotly
- `docker-compose.yml` - Docker Compose configuration using the optimized Dockerfiles
- `pyproject.toml.optional-deps` - Optional dependency groups to add to your main pyproject.toml

## Setup Instructions

1. **Add optional dependencies to your main pyproject.toml**
   
   Copy the dependency groups from `pyproject.toml.optional-deps` and add them to your main `pyproject.toml` under `[project.optional-dependencies]`:
   
   - `ml` - For ML services (training/inference)
   - `mlflow` - For MLflow server only
   - `data` - For data streaming service
   - `ui` - For UI service

2. **Update the lock file (optional)**
   
   If you want to use the optional dependencies for local development, update the lock file:
   
   ```bash
   uv lock
   ```
   
   Note: The minimal Dockerfiles (mlflow, data, ui) install packages directly with `uv pip install`, so they don't strictly need the lock file. Only Dockerfile.ml uses `uv sync` which requires the lock file.

3. **Build and run**
   
   From the project root:
   ```bash
   docker-compose -f docker_files/docker-compose.yml up --build
   ```
   
   Or from the `docker_files/` directory:
   ```bash
   docker-compose -f docker-compose.yml up --build
   ```

## Benefits

- **Faster builds**: Each service only installs what it needs
- **Smaller images**: Reduced image sizes (especially mlflow and data services)
- **Better caching**: Less dependency overlap means better Docker layer caching
- **Maintainable**: Single source of truth for dependencies in pyproject.toml

## Service Dependencies

| Service | Dockerfile | Dependencies |
|---------|-----------|--------------|
| mlflow | Dockerfile.mlflow | mlflow, pyyaml |
| app-ml-train | Dockerfile.ml | kedro, mlflow, catboost, scikit-learn, optuna, etc. |
| app-ml-inference | Dockerfile.ml | kedro, mlflow, catboost, scikit-learn, etc. |
| kedro-viz | Dockerfile.ml | kedro, kedro-viz |
| app-stream-data | Dockerfile.data | pandas, pyarrow, pyyaml |
| app-ui | Dockerfile.ui | dash, mlflow, pandas, numpy, plotly |

## Notes

- The docker-compose.yml uses `context: ..` to set the build context to the project root
- All volumes use `../` paths since docker-compose.yml is in a subdirectory
- **Dockerfile.ml** uses `uv sync --frozen --extra ml` because it needs the full project installed (for Kedro)
- **Other Dockerfiles** (mlflow, data, ui) use `uv pip install` to install only specific packages without the full project
- Source code is mounted as volumes at runtime, so it doesn't need to be in the image for most services
- The optional dependency groups in pyproject.toml are primarily for reference/documentation - the Dockerfiles install packages directly for minimal services

