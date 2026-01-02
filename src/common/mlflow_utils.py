"""MLflow utility functions for model registry operations."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def get_model_info_by_alias(
    alias: str,
    mlflow_tracking_uri: str | None = None,
    model_name: str | None = None,
) -> dict | None:
    """
    Get information about a model from MLflow registry by alias.

    This function can retrieve information for any model with a specific alias
    (e.g., "champion", "challenger", etc.) in MLflow.

    This function:
    1. Connects to MLflow tracking server
    2. Finds model version with the specified alias
    3. Retrieves model metadata (version, last updated)
    4. Extracts test set metrics from the training run
    5. Handles various metric name formats (test_mae, test_MAE, etc.)

    Args:
        alias: Model alias to search for (e.g., "champion", "challenger")
        mlflow_tracking_uri: MLflow tracking server URI (optional)
                            If None, uses default or environment variable
        model_name: Name of registered model (optional)
                   If None, reads from config file

    Returns:
        Dictionary with keys:
            - model_name: Name of the model
            - version: Model version number (string)
            - last_updated: Datetime when model was last updated
            - test_mae: Mean Absolute Error on test set (float or None)
            - test_mape: Mean Absolute Percentage Error on test set (float or None)

        Returns None if:
            - Model not found
            - Model has no alias matching the specified alias
            - Error occurs during query
    """
    # Get tracking URI from environment if not provided, default to localhost:5001
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Get model name from parameters if not provided
    if model_name is None:
        project_root = Path(__file__).resolve().parents[2]
        parameters_path = project_root / "conf" / "base" / "parameters.yml"
        with open(parameters_path) as f:
            params = yaml.safe_load(f)
        model_name = params["mlflow"]["registered_model_name"]

    client = MlflowClient()

    # Get the model version by the specified alias
    try:
        model_version_info = client.get_model_version_by_alias(
            name=model_name, alias=alias
        )
    except Exception as e:
        logger.debug(f"Model '{model_name}' with alias '{alias}' not found: {e}")
        return None

    # Extract metadata
    version = model_version_info.version

    # Validate required fields
    if model_version_info.last_updated_timestamp is None:
        logger.warning(f"Model version {version} has no timestamp")
        return None

    if model_version_info.run_id is None:
        logger.warning(f"Model version {version} has no run_id")
        return None

    last_updated = datetime.fromtimestamp(
        model_version_info.last_updated_timestamp / 1000.0
    )

    # Get the run associated with this model version to extract metrics
    try:
        run = client.get_run(model_version_info.run_id)
    except Exception as e:
        logger.warning(f"Failed to get run {model_version_info.run_id}: {e}")
        return None

    # Extract test set error metrics from run metrics
    metrics = run.data.metrics

    # Try multiple variations of MAE metric names
    mae_variations = [
        "test_mae",
        "test_MAE",
        "test_mae_err",
        "test_MAE_err",
        "mae_test",
        "MAE_test",
    ]
    test_mae = next((metrics[key] for key in mae_variations if key in metrics), None)

    # Try multiple variations of MAPE metric names
    mape_variations = [
        "test_mape",
        "test_MAPE",
        "test_mape_err",
        "test_MAPE_err",
        "mape_test",
        "MAPE_test",
    ]
    test_mape = next((metrics[key] for key in mape_variations if key in metrics), None)

    return {
        "model_name": model_name,
        "version": version,
        "last_updated": last_updated,
        "test_mae": test_mae,
        "test_mape": test_mape,
    }


def load_model_by_alias(
    model_name: str,
    alias: str = "champion",
    mlflow_tracking_uri: str | None = None,
) -> Any:
    """
    Load a model from MLflow registry by alias.

    Args:
        model_name: Name of the registered model
        alias: Model alias (default: "champion")
        mlflow_tracking_uri: MLflow tracking server URI (optional)

    Returns:
        Loaded MLflow pyfunc model
    """
    # Get tracking URI from environment if not provided
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Get the model version by the specified alias
    model_version_info = client.get_model_version_by_alias(name=model_name, alias=alias)

    # Load the model using the version number
    model_uri = f"models:/{model_name}/{model_version_info.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model
