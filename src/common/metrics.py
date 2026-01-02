"""Metric computation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and MAPE metrics for regression predictions.

    Args:
        y_true: Ground truth values (Series or array)
        y_pred: Predicted values (Series or array)

    Returns:
        Dictionary with keys: 'mae', 'rmse', 'mape'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = (
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
