from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from app_data_manager.data_manager import DataManager  # type: ignore

# Add app-data-manager to path for DataManager import
project_root = Path(__file__).resolve().parents[4]
app_data_manager_path = project_root / "src" / "app-data-manager"
sys.path.insert(0, str(app_data_manager_path))


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load data from a dataframe.
    """
    return df


def rename_columns(df: pd.DataFrame, rename_columns: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns in a dataframe.
    """
    return df.rename(columns=rename_columns)


def drop_columns(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    """
    Drop columns in a dataframe.
    """
    return df.drop(columns=drop_columns)


def remove_diff_outliers(
    df: pd.DataFrame, diff_thresholds: dict[str, float]
) -> pd.DataFrame:
    """
    Remove outliers based on absolute first-order diff and forward-fill the gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    diff_thresholds : Dict[str, float]
        Dictionary of column names and their absolute diff thresholds.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe with forward fill.
    outlier_idx : pd.Index
        Indices of removed outliers.
    """

    df_clean = df.copy()

    for col, threshold in diff_thresholds.items():
        # 1. Compute absolute diff
        diff_vals = df_clean[col].diff(1).abs()

        # 2. Outlier mask
        outlier_mask = diff_vals > threshold
        outlier_idx = df_clean.index[outlier_mask]

        # 3. Remove outliers and fill nan
        df_clean.loc[outlier_idx, col] = np.nan
        df_clean[col] = df_clean[col].ffill().bfill()

    return df_clean


def smooth_signal(df, columns, window, method="median"):
    """
    Smooth a time-series column using rolling mean or median.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : str
        Columns to smooth.
    window : int
        Rolling window size.
    method : str
        "mean"  -> rolling mean filter
        "median" -> rolling median filter (robust smoothing)

    Returns
    -------
    df_smoothed : pd.DataFrame
        DataFrame with smoothed column.
    """

    df_smoothed = df.copy()

    for column in columns:
        if method == "mean":
            df_smoothed[column] = (
                df_smoothed[column]
                .rolling(window=window, min_periods=1, center=False)
                .mean()
            )

        elif method == "median":
            df_smoothed[column] = (
                df_smoothed[column]
                .rolling(window=window, min_periods=1, center=False)
                .median()
            )

        else:
            raise ValueError("method must be 'mean' or 'median'")

    return df_smoothed


def add_lag_features(df: pd.DataFrame, lags_dict: dict[str, list[int]], drop_na=False):
    """
    Add lag features to DataFrame.
    Parameters:
    - df: DataFrame
    - columns: list of column names (default: all numeric)
    - lags: int or list of lag periods (default: 1)
    - drop_na: bool, drop NaN rows (default: True)
    Returns: DataFrame with lag features
    """
    df_result = df.copy()

    # Create lag features
    for col, lags in lags_dict.items():
        for lag in lags:
            df_result[f"{col}_lag{lag}"] = df_result[col].shift(lag)

    if drop_na:
        return df_result.dropna()
    else:
        return df_result.bfill()  # Backward fill NaNs


def add_rolling_features(
    df,
    stats_window_dict: dict[str, dict[str, list]],
    drop_na=False,
):
    """
    Add rolling features to DataFrame.
    Parameters:
    - df: DataFrame
    - columns: list of column names
    - window_sizes: int or list of window sizes (default: 7)
    - stats: list of statistics ['mean', 'median', 'std', 'min', 'max', 'skew', 'kurt']
    - drop_na: bool, drop NaN rows (default: True)

    Returns: DataFrame with rolling features
    """
    df_result = df.copy()

    for col in stats_window_dict.keys():
        for window in stats_window_dict[col]["windows"]:
            rolling = df_result[col].rolling(window)
            for stat in stats_window_dict[col]["stats"]:
                if stat == "mean":
                    df_result[f"{col}_roll{window}_mean"] = rolling.mean()
                elif stat == "median":
                    df_result[f"{col}_roll{window}_median"] = rolling.median()
                elif stat == "std":
                    df_result[f"{col}_roll{window}_std"] = rolling.std()
                elif stat == "min":
                    df_result[f"{col}_roll{window}_min"] = rolling.min()
                elif stat == "max":
                    df_result[f"{col}_roll{window}_max"] = rolling.max()
                elif stat == "skew":
                    df_result[f"{col}_roll{window}_skew"] = rolling.skew()
                elif stat == "kurt":
                    df_result[f"{col}_roll{window}_kurt"] = rolling.kurt()
    if drop_na:
        return df_result.dropna()
    else:
        return df_result.bfill()  # Backward fill NaNs


def get_features_and_target(
    df: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Get the features and target from the dataframe.
    """
    x = df.drop(columns=params["target"]).copy()
    y = df[params["target"]].copy()
    return x, y


def get_data_timestamps(df: pd.DataFrame) -> pd.Timestamp:
    """
    Get the current timestamp from the dataframe.
    """
    return df["Timestamps"]


def load_training_data_from_db(
    start_timestamp: str,
    end_timestamp: str,
    table_name: str,
    data_manager_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load training data from SQLite database by timestamp range.

    Args:
        start_timestamp: Start timestamp (inclusive)
        end_timestamp: End timestamp (inclusive)
        table_name: Name of the table to read from
        data_manager_config: DataManager configuration dictionary

    Returns:
        DataFrame containing data within the timestamp range
    """
    data_manager = DataManager({"data_manager": data_manager_config})

    df = data_manager.get_data_by_timestamp_range(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        table_name=table_name,
    )

    return df


def get_last_n_points_from_db(
    n: int,
    table_name: str,
    data_manager_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Get the last N data points from SQLite database.

    Args:
        n: Number of points to retrieve
        table_name: Name of the table to read from
        data_manager_config: DataManager configuration dictionary

    Returns:
        DataFrame containing the last N rows, ordered by Timestamps
    """
    data_manager = DataManager({"data_manager": data_manager_config})

    df = data_manager.get_last_n_points(n=n, table_name=table_name)

    return df
