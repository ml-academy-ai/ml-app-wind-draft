import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
from app_data_manager.data_manager import DataManager  # type: ignore
from mlflow.tracking import MlflowClient

# Add app-data-manager to path for DataManager import
project_root = Path(__file__).resolve().parents[4]
app_data_manager_path = project_root / "src" / "app_data_manager"
sys.path.insert(0, str(app_data_manager_path))


def load_from_registry(registered_model_name: str) -> Any:
    """
    Load the champion model from MLflow model registry by alias.
    Parameters
    ----------
    registered_model_name : str
        The name of the registered model in MLflow (e.g., 'wind_power_predictor').

    Returns
    -------
    Any
        The loaded champion model (MLflow pyfunc model) with scaler bundled.
        This model can be used directly for predictions as it handles scaling internally.
    """
    client = MlflowClient()

    # Get the model version by the "champion" alias
    model_version_info = client.get_model_version_by_alias(
        name=registered_model_name, alias="champion"
    )

    # Load the model using the version number
    model_uri = f"models:/{registered_model_name}/{model_version_info.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(x: pd.DataFrame, best_model: Any) -> pd.Series:
    """
    Make predictions using a trained model on new data.

    This function applies feature scaling (if a scaler is provided) and then
    generates predictions using the trained model. The input features are scaled
    using the same scaler that was fitted during training to ensure consistency.

    Parameters
    ----------
    x : pd.DataFrame
        Input features for prediction. Must have the same columns as the
        training data (excluding the target column).
    best_model : Any
        Trained model object (e.g., MLflow pyfunc model, RandomForestRegressor,
        CatBoostRegressor) that has a `predict` method.

    Returns
    -------
    pd.Series
        Predicted target values for the input features.
    """
    return pd.Series(best_model.predict(x))


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Compute metrics for the predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    )
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
    return metrics


def save_predictions_to_db(
    y_pred: pd.Series,
    data_timestamps: pd.Timestamp,
    data_manager_config: dict[str, Any],
) -> None:
    """
    Save predictions to the SQLite database using DataManager.

    Args:
        y_pred: Predicted values as a Series
        data_manager_config: DataManager configuration dictionary
    """
    if y_pred is None or len(y_pred) == 0:
        return

    # Initialize DataManager
    data_manager = DataManager({"data_manager": data_manager_config})

    # Convert input timestamps to pandas datetime
    timestamps = pd.to_datetime(data_timestamps)

    # Normalize timestamps to string format expected by the database
    # This works for both Series-like inputs and single Timestamp values
    if isinstance(timestamps, pd.Series):
        timestamps_str = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamps_str = timestamps.strftime("%Y-%m-%d %H:%M:%S")

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "Timestamps": timestamps_str,
            "predicted_power": y_pred.values,
        }
    )

    # Save to predictions table
    data_manager.insert_data_to_db(
        new_data=predictions_df,
        table_name=data_manager.predictions_table_name,
    )
