from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app_data_manager.data_manager import DataManager  # type: ignore
from common.metrics import compute_metrics as _compute_metrics
from common.mlflow_utils import load_model_by_alias

# Add app-data-manager to path for DataManager import
project_root = Path(__file__).resolve().parents[4]


def load_from_registry(registered_model_name: str) -> Any:
    """
    Load the champion model from MLflow model registry by alias.

    Node function that wraps the common load_model_by_alias implementation.

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
    return load_model_by_alias(registered_model_name, alias="champion")


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

    Node function that wraps the common compute_metrics implementation.
    """
    return _compute_metrics(y_true, y_pred)


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

    # Ensure predictions table exists before saving
    data_manager.init_predictions_db_table()

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
