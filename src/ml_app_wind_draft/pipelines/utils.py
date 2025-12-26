"""Shared utility functions for pipelines."""

from typing import Any

import pandas as pd


def get_features_and_target(
    df: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from a dataframe based on configuration.

    This function separates the input dataframe into features (X) and target (y)
    based on the target column name specified in the parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing both features and target column.
    params : Dict[str, Any]
        Configuration dictionary containing:
        - 'target': str, name of the target column to extract

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing:
        - x: pd.DataFrame with all columns except the target
        - y: pd.Series with the target column values

    Examples
    --------
    >>> params = {"target": "power"}
    >>> x, y = get_features_and_target(df, params)
    """
    x = df.drop(columns=params["target"]).copy()
    y = df[params["target"]].copy()
    return x, y
