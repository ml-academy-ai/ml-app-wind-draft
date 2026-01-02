from typing import Any, Literal

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from common.metrics import compute_metrics as _compute_metrics


class MLModelWrapper(mlflow.pyfunc.PythonModel):
    """
    PyFunc model wrapper with bundled scaler.
    Expects unscaled, feature-engineered DataFrame.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()

    def load_context(self, context):
        # load scaler
        self.scaler = joblib.load(context.artifacts["scaler"])

        # load model
        if self.model_name == "catboost":
            self.model = CatBoostRegressor()
            self.model.load_model(context.artifacts["model"])
        elif self.model_name in ["rf", "random_forest"]:
            self.model = joblib.load(context.artifacts["model"])
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame
    ) -> np.ndarray:  # ty: ignore[invalid-method-override]
        X_scaled = self.scaler.transform(model_input)
        return self.model.predict(X_scaled)


def objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> float:
    """
    Optuna objective for the ML model. Minimizes cross-validated MAPE (cv_mape).
    """
    np.random.seed(42)

    model = params["optuna_search"]["model"]
    sampled_model_params = sample_optuna_params(
        params["optuna_search"]["model_params"][model], trial
    )

    # Evaluate using CV evaluator (TimeSeriesSplit cross-validation)
    eval_results = eval_model(
        x_train=x_train,
        y_train=y_train,
        model_name=model,
        model_params=sampled_model_params,
    )
    return eval_results["cv_mape"]


def sample_optuna_params(
    model_params: dict[str, Any], trial: optuna.Trial
) -> dict[str, Any]:
    """
    Samples Optuna hyperparameters from a trial.

    Parameters
    ----------
    model_params : Dict[str, Any]
        Dictionary mapping parameter names to their definitions. Can be:
        - Fixed values: int, float, str, bool (e.g., iterations: 50)
        - Search space: dict with "range", "log", "type" keys (e.g., depth: {range: [3, 10], log: False, type: int})
        - Categorical: list of values (e.g., loss_function: ["RMSE", "MAE"])
    trial : optuna.Trial
        The Optuna trial object used to suggest parameter values during optimization.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to their sampled values from the trial.
    """

    param_grid = {}
    # Iterate through each parameter in the model configuration
    for param, values in model_params.items():
        # Check if parameter has a continuous range definition (with "range" and "log" keys)
        # This indicates a numeric parameter that should be sampled from a range
        if isinstance(values, dict) and "range" in values and "log" in values:
            # Check if the parameter type is integer (default is float)
            if values.get("type", "float") == "int":
                # Sample integer value from the specified range with optional log scaling
                param_grid[param] = trial.suggest_int(
                    param, values["range"][0], values["range"][1], log=values["log"]
                )
            else:
                # Sample float value from the specified range with optional log scaling
                param_grid[param] = trial.suggest_float(
                    param, values["range"][0], values["range"][1], log=values["log"]
                )
        elif isinstance(values, list):
            # Parameter is categorical - sample from a discrete list of possible values
            param_grid[param] = trial.suggest_categorical(param, values)
        else:
            # Parameter is a fixed value (int, float, str, bool, etc.) - pass through as-is
            param_grid[param] = values

    return param_grid


def eval_model(  # noqa: PLR0913
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 3,
    model_name: Literal["RF", "CatBoost"] = "RF",
    model_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a time series model using TimeSeriesSplit cross-validation
    on the training set, then refit on the full train data and evaluate on test.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_splits : int, default=3
        Number of time-series CV folds.
    model_name : {'RF', CatBoost'}, default='RF'
        Model identifier.
    model_params : dict or None
        Keyword arguments for the selected model.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        - 'cv_mae'   : average MAE over CV folds
        - 'cv_rmse'  : average RMSE over CV folds
        - 'cv_mape'  : average MAPE over CV folds
        - 'model'      : fitted final model
        - 'x_scaler'   : fitted StandardScaler for X features
    """
    np.random.seed(seed)

    if model_params is None:
        model_params = {}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mae_list = []
    cv_rmse_list = []
    cv_mape_list = []

    # Sliding Time Series Cross-Validation
    for _, (train_idx, val_idx) in enumerate(tscv.split(x_train), 1):
        x_train_cv = x_train.iloc[train_idx, :].copy()
        x_val_cv = x_train.iloc[val_idx, :].copy()
        y_train_cv = y_train.iloc[train_idx].copy()
        y_val_cv = y_train.iloc[val_idx].copy()

        # Scale features (to make it more general and applicable not only for tree-based models)
        x_scaler = StandardScaler()
        x_scaled_cv_train = x_scaler.fit_transform(x_train_cv)
        x_scaled_cv_val = x_scaler.transform(x_val_cv)

        y_train_cv_vals = y_train_cv.values.ravel()
        y_val_cv_vals = y_val_cv.values.ravel()

        # Construct model
        if model_name == "RF":
            model = RF(**model_params)
        elif model_name == "CatBoost":
            model = CatBoostRegressor(
                **model_params, allow_writing_files=False, verbose=False
            )

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # Fit model and predict on validation set
        model.fit(x_scaled_cv_train, y_train_cv_vals)
        y_pred_cv = model.predict(x_scaled_cv_val)

        mae_err, rmse_err, mape_err = compute_metrics(y_val_cv_vals, y_pred_cv)

        cv_mae_list.append(float(mae_err))
        cv_rmse_list.append(float(rmse_err))
        cv_mape_list.append(float(mape_err))

    # Compute average metrics over all folds
    cv_mae = round(np.mean(cv_mae_list), 2)
    cv_rmse = round(np.mean(cv_rmse_list), 2)
    cv_mape = round(np.mean(cv_mape_list), 2)

    return {
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_mape": cv_mape,
        "model": model,
        "x_scaler": x_scaler,
    }


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute metrics for the predictions.

    Returns tuple for backward compatibility with existing code.
    """
    metrics = _compute_metrics(y_true, y_pred)
    return metrics["mae"], metrics["rmse"], metrics["mape"]


def train_val_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    finetune_cfg: dict[str, Any],
):
    """
    Explicit temporal train / val / test split using index boundaries.
    """
    val_fraction = float(finetune_cfg.get("val_size", 0.15))
    test_fraction = float(finetune_cfg.get("test_size", 0.15))

    n_samples = features.shape[0]

    n_test = int(n_samples * test_fraction)
    n_val = int(n_samples * val_fraction)

    train_idx = n_samples - n_val - n_test
    val_idx = n_samples - n_test

    if train_idx <= 0:
        raise ValueError("not enough samples after train/val/test split")

    x_train = features.iloc[:train_idx].copy()
    y_train = target.iloc[:train_idx].values.ravel()

    x_val = features.iloc[train_idx:val_idx].copy()
    y_val = target.iloc[train_idx:val_idx].values.ravel()

    x_test = features.iloc[val_idx:].copy()
    y_test = target.iloc[val_idx:].values.ravel()

    return x_train, y_train, x_val, y_val, x_test, y_test
