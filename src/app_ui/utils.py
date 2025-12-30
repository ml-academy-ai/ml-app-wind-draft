"""
Utility Functions for Dash Application

This module contains reusable functions used across multiple pages:
1. Data Loading: Load production data from database
2. Error Computation: Calculate MAE and MAPE metrics
3. MLflow Integration: Query champion model information
4. Data Preparation: Process data for visualization
5. Plot Creation: Generate Plotly figures for charts
6. Plot Synchronization: Sync x-axis between multiple plots

All functions are designed to be reusable and independent of specific pages.
"""

import copy
import traceback
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import no_update
from mlflow.tracking import MlflowClient

from app_data_manager.data_manager import DataManager
from app_data_manager.utils import read_config

# Configuration setup
# Read configuration once at module level for efficiency
# Config contains: database settings, anomaly thresholds, MLflow settings
project_root = Path(__file__).resolve().parents[2]
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

df_c = pd.DataFrame(
    {"datetime": pd.date_range(start="2025-01-01", end="2025-01-02", freq="h")}
)
df_c["true_value"] = np.random.randn(len(df_c))
df_c["prediction"] = np.random.randn(len(df_c))
df_c["mae"] = np.abs(df_c["true_value"] - df_c["prediction"])
df_c["mape"] = (
    np.abs(
        (df_c["true_value"] - df_c["prediction"])
        / np.clip(df_c["true_value"], 1e-8, None)
    )
    * 100
)
df_c["mae_rolling"] = df_c["mae"].rolling(window=24, min_periods=1).mean()
df_c["mape_rolling"] = df_c["mape"].rolling(window=24, min_periods=1).mean()


# Data loading functions
def load_prod_data(
    n_data_points: int = 1000000,
    start_timestamp: str | pd.Timestamp | None = None,
    end_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load production data with predictions and true values from database.

    This function queries the SQLite database to retrieve:
    - Raw data (true values) from 'raw_data' table
    - Predictions from 'predictions' table
    - Merges them on Timestamps for comparison

    Two query modes:
    1. By data points: Gets last N points (default: 1,000,000)
    2. By date range: Gets data between start_timestamp and end_timestamp

    Args:
        n_data_points: Number of data points to retrieve. Defaults to 1,000,000.
                       Only used if start_timestamp and end_timestamp are None.
        start_timestamp: Start timestamp for data range (string or pd.Timestamp).
                        If provided with end_timestamp, uses date range query.
        end_timestamp: End timestamp for data range (string or pd.Timestamp).
                       If provided with start_timestamp, uses date range query.

    Returns:
        DataFrame with columns:
            - Timestamps: Original timestamp column
            - true_value: Actual values (from 'Power' column)
            - prediction: Model predictions (from 'predicted_power' column)

    Raises:
        ValueError: If no raw data found in database
    """
    data_manager = DataManager(config)

    if start_timestamp is not None and end_timestamp is not None:
        raw_data = data_manager.get_data_by_timestamp_range(
            start_timestamp, end_timestamp, table_name="raw_data"
        )
        predictions = data_manager.get_data_by_timestamp_range(
            start_timestamp, end_timestamp, table_name="predictions"
        )
    else:
        raw_data = data_manager.get_last_n_points(n_data_points, table_name="raw_data")
        predictions = data_manager.get_last_n_points(
            n_data_points, table_name="predictions"
        )

    if raw_data.empty:
        raise ValueError("No raw data found in database")

    if not predictions.empty:
        df = pd.merge(
            raw_data[["Timestamps", "Power"]],
            predictions[["Timestamps", "predicted_power"]],
            on="Timestamps",
            how="inner",
        )
    else:
        df = raw_data[["Timestamps", "Power"]].copy()
        df["predicted_power"] = np.nan

    df["true_value"] = df["Power"]
    df["prediction"] = df["predicted_power"]
    return df.sort_values("Timestamps").reset_index(drop=True)


def compute_errors(df):
    """
    Compute error metrics between true values and predictions.

    Calculates two error metrics:
    1. MAE (Mean Absolute Error): |true - prediction|
    2. MAPE (Mean Absolute Percentage Error): |(true - prediction) / true| * 100

    MAPE uses clipping to avoid division by zero (clips to minimum 1e-8).

    Args:
        df: DataFrame with 'true_value' and 'prediction' columns

    Returns:
        DataFrame with added columns:
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error (as percentage, e.g., 5.0 = 5%)
    """
    df = df.copy()
    df["mae"] = np.abs(df["true_value"] - df["prediction"])
    df["mape"] = (
        np.abs(
            (df["true_value"] - df["prediction"])
            / np.clip(df["true_value"], 1e-8, None)
        )
        * 100
    )
    return df


# MLflow integration functions
def get_model_info_by_alias(
    alias: str, mlflow_tracking_uri=None, model_name=None
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
    try:
        # Set tracking URI
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Get model name from parameters if not provided
        if model_name is None:
            with open(parameters_path) as f:
                params = yaml.safe_load(f)
            model_name = params.get("mlflow", {}).get(
                "registered_model_name", "wind_power_predictor"
            )

        client = MlflowClient()

        # Get the model version by the specified alias
        try:
            model_version_info = client.get_model_version_by_alias(
                name=model_name, alias=alias
            )
        except Exception:
            # If alias doesn't exist, return None
            return None

        version = model_version_info.version

        # Handle potentially None timestamp
        if model_version_info.last_updated_timestamp is None:
            return None
        last_updated = datetime.fromtimestamp(
            model_version_info.last_updated_timestamp / 1000.0
        )

        # Get the run associated with this model version to extract metrics
        run_id = model_version_info.run_id
        if run_id is None:
            return None
        run = client.get_run(run_id)

        # Extract test set error metrics directly from MLflow registry run metrics
        # Check all possible metric name variations
        metrics = run.data.metrics

        # Try multiple variations of MAE metric names
        test_mae = None
        for metric_key in [
            "test_mae",
            "test_MAE",
            "test_mae_err",
            "test_MAE_err",
            "mae_test",
            "MAE_test",
        ]:
            if metric_key in metrics:
                test_mae = metrics[metric_key]
                break

        # Try multiple variations of MAPE metric names
        test_mape = None
        for metric_key in [
            "test_mape",
            "test_MAPE",
            "test_mape_err",
            "test_MAPE_err",
            "mape_test",
            "MAPE_test",
        ]:
            if metric_key in metrics:
                test_mape = metrics[metric_key]
                break

        # If MAPE is stored as percentage, ensure it's in the right format
        # (some logs might store it as 0.05 for 5%, others as 5.0)
        if test_mape is not None and test_mape < 1:
            # Assume it's stored as decimal (0.05 = 5%), convert to percentage
            test_mape = test_mape * 100

        return {
            "model_name": model_name,
            "version": version,
            "last_updated": last_updated,
            "test_mae": test_mae,
            "test_mape": test_mape,
        }
    except Exception:
        traceback.print_exc()
        return None


# Plot creation functions
def create_error_plot(
    df: pd.DataFrame, metric_config: dict, metric_threshold: float, rolling_window: int
) -> go.Figure:
    """
    Create Plotly figure for error metrics plot.

    This plot shows:
    1. Raw error values (light line, semi-transparent)
    2. Rolling average (bold line, fully opaque)
    3. Anomaly threshold (dashed black line)

    The threshold line helps identify when errors exceed acceptable levels.

    Args:
        df: DataFrame with datetime and error columns
        metric_config: Configuration dict from get_metric_config()
        metric_threshold: Threshold value for anomaly detection
        rolling_window: Window size used for rolling average (for legend label)

    Returns:
        Plotly Figure object ready for display
    """
    x_min, x_max = df["datetime"].min(), df["datetime"].max()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=metric_config["raw_data"],
            name=metric_config["name"],
            mode="lines",
            line=dict(color=metric_config["raw_color"], width=1.5),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=metric_config["rolling_data"],
            name=f"{metric_config['name']} Rolling ({rolling_window} pts)",
            mode="lines",
            line=dict(color=metric_config["rolling_color"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[metric_threshold, metric_threshold],
            name=f"Threshold: {metric_threshold:.3f}",
            mode="lines",
            line=dict(
                color="#111827", width=2.5, dash="dash"
            ),  # Dark grey for threshold
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=30, b=40),
        height=None,
        showlegend=True,
        xaxis_title="Time",
        yaxis_title=metric_config["label"],
        xaxis=dict(range=[x_min, x_max], gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        yaxis=dict(gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1B1D23", size=12),
    )
    return fig


def get_date_range_from_lookback(lookback_days: int) -> tuple[str, str]:
    """
    Calculate start and end timestamps based on lookback days.

    This function:
    1. Loads a small sample of data to find the maximum timestamp
    2. Calculates start date = max_date - lookback_days
    3. Returns both as formatted strings for database queries

    Args:
        lookback_days: Number of days to look back from latest data point

    Returns:
        Tuple of (start_timestamp, end_timestamp) as formatted strings
        Format: "YYYY-MM-DD HH:MM:SS"

    Raises:
        ValueError: If no data available in database
    """
    df_sample = load_prod_data(n_data_points=1000)
    if df_sample.empty:
        raise ValueError("No data available in database")
    df_sample["datetime"] = pd.to_datetime(
        df_sample["Timestamps"], format="mixed", errors="coerce"
    )
    max_date = df_sample["datetime"].max()
    min_date = max_date - pd.Timedelta(days=lookback_days)
    return min_date.strftime("%Y-%m-%d %H:%M:%S"), max_date.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def load_and_prepare_error_data(
    start_dt: str, end_dt: str, rolling_window: int
) -> pd.DataFrame:
    """
    Load production data, compute errors, and add rolling averages.

    This is a convenience function that combines multiple steps:
    1. Load data from database for the specified date range
    2. Compute MAE and MAPE error metrics
    3. Add datetime column for plotting
    4. Calculate rolling averages for smoothing

    Rolling averages help identify trends by reducing noise in error metrics.

    Args:
        start_dt: Start timestamp string (format: "YYYY-MM-DD HH:MM:SS")
        end_dt: End timestamp string (format: "YYYY-MM-DD HH:MM:SS")
        rolling_window: Window size for rolling average (in data points, not time)
                       Example: 288 points = 12 days if data is hourly

    Returns:
        DataFrame with columns:
            - Timestamps, true_value, prediction (from load_prod_data)
            - mae, mape (from compute_errors)
            - datetime (converted from Timestamps)
            - mae_rolling, mape_rolling (rolling averages)
    """
    df = load_prod_data(start_timestamp=start_dt, end_timestamp=end_dt)
    df = compute_errors(df)
    df["datetime"] = pd.to_datetime(df["Timestamps"], format="mixed", errors="coerce")
    df["mae_rolling"] = df["mae"].rolling(window=rolling_window, min_periods=1).mean()
    df["mape_rolling"] = df["mape"].rolling(window=rolling_window, min_periods=1).mean()
    return df


def get_metric_config(error_metric: str, df: pd.DataFrame) -> dict:
    """
    Get configuration dictionary for selected error metric.

    Returns color scheme, labels, and data columns for either MAE or MAPE.
    This centralizes styling and data selection for plots.

    Args:
        error_metric: Either "mae" or "mape" (case-sensitive)
        df: DataFrame with error columns (mae, mape, mae_rolling, mape_rolling)

    Returns:
        Dictionary with keys:
            - name: Short name ("MAE" or "MAPE")
            - label: Full label ("Mean Absolute Error" or "Mean Absolute Percentage Error")
            - raw_color: Color for raw error line (hex code)
            - rolling_color: Color for rolling average line (hex code)
            - raw_data: Series of raw error values
            - rolling_data: Series of rolling average values
    """
    if error_metric == "mae":
        return {
            "name": "MAE",
            "label": "Mean Absolute Error",
            "raw_color": "#60a5fa",  # Lighter blue for raw data (matches sidebar hover)
            "rolling_color": "#2563eb",  # Matches sidebar active gradient end
            "raw_data": df["mae"],
            "rolling_data": df["mae_rolling"],
        }
    return {
        "name": "MAPE",
        "label": "Mean Absolute Percentage Error",
        "raw_color": "#60a5fa",  # Lighter blue for raw data
        "rolling_color": "#2563eb",  # Matches sidebar active gradient end
        "raw_data": df["mape"],
        "rolling_data": df["mape_rolling"],
    }


def create_timeseries_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create Plotly figure for time series plot (predictions vs true values).

    This plot shows:
    1. True values (actual wind power) - blue line
    2. Predictions (model output) - teal/cyan line

    Allows visual comparison of model performance over time.

    Args:
        df: DataFrame with datetime, true_value, and prediction columns

    Returns:
        Plotly Figure object ready for display
    """
    x_min, x_max = df["datetime"].min(), df["datetime"].max()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["true_value"],
            name="True Values",
            mode="lines",
            line=dict(color="#16A34A", width=2.5),  # Success green for true values
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["prediction"],
            name="Predictions",
            mode="lines",
            line=dict(color="#2563eb", width=2.5),  # Matches sidebar active color
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=30, b=40),
        height=None,
        showlegend=True,
        xaxis_title="Time",
        yaxis_title="Power",
        xaxis=dict(range=[x_min, x_max], gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        yaxis=dict(gridcolor="#CBD5E1", linecolor="#E3E8EF"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1B1D23", size=12),
    )
    return fig


def create_error_figure(error_msg: str) -> tuple[go.Figure, go.Figure]:
    """
    Create error display figures for exception handling.

    When data loading or processing fails, this function creates two
    Plotly figures that display the error message instead of crashing.
    This ensures the UI always shows something, even on errors.

    Args:
        error_msg: Error message string to display

    Returns:
        Tuple of (figure1, figure2) - both show the same error message
        These can be returned to replace the normal plots
    """
    fig1 = go.Figure()
    fig2 = go.Figure()
    error_layout = {
        "template": "plotly_white",
        "margin": dict(l=40, r=40, t=30, b=40),
        "height": None,
        "annotations": [
            {
                "text": f"Error loading data: {error_msg}",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
        "plot_bgcolor": "#fff",
        "paper_bgcolor": "#fff",
    }
    fig1.update_layout(**error_layout)
    fig2.update_layout(**error_layout)
    return fig1, fig2


# Plot synchronization functions
def sync_xaxis(relayout_data, current_figure):
    """
    Synchronize x-axis range from one plot to another.

    This function extracts the x-axis range from a plot's relayoutData
    (generated when user zooms/pans) and applies it to another plot's figure.

    Key features:
    - Handles zoom events (xaxis.range[0] and xaxis.range[1])
    - Handles autorange resets (xaxis.autorange)
    - Uses deepcopy to avoid mutating the original figure
    - Returns no_update if no relevant changes detected

    Args:
        relayout_data: Dictionary from Plotly's relayoutData property
                      Contains zoom/pan information when user interacts with plot
        current_figure: Current figure dictionary of the plot to update
                      This is the plot that will receive the new x-axis range

    Returns:
        Tuple of (updated_figure, x_range):
            - updated_figure: New figure with updated x-axis range (or no_update)
            - x_range: New x-axis range [min, max] or None (or no_update)
    """
    if not relayout_data or not current_figure:
        return no_update, no_update

    # Check if there's a range change
    if not any(
        key in relayout_data
        for key in ("xaxis.range[0]", "xaxis.range", "xaxis.autorange")
    ):
        return no_update, no_update

    updated_figure = copy.deepcopy(current_figure)
    if "xaxis" not in updated_figure["layout"]:
        updated_figure["layout"]["xaxis"] = {}

    # Handle autorange reset
    if "xaxis.autorange" in relayout_data:
        updated_figure["layout"]["xaxis"].pop("range", None)
        updated_figure["layout"]["xaxis"]["autorange"] = True
        return updated_figure, None

    # Handle range changes
    x_range = None
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        x_range = [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
    elif "xaxis.range" in relayout_data and isinstance(
        relayout_data["xaxis.range"], list
    ):
        x_range = relayout_data["xaxis.range"]

    if x_range:
        updated_figure["layout"]["xaxis"]["range"] = x_range
        updated_figure["layout"]["xaxis"]["autorange"] = False
        return updated_figure, x_range

    return no_update, no_update
