from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from .utils import MLModelWrapper, compute_metrics, eval_model, objective


def train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Explicit temporal train / val / test split using index boundaries.
    """
    test_idx = int(features.shape[0] * (1 - params["test_fraction"]))
    x_train, x_test = features.iloc[:test_idx].copy(), features.iloc[test_idx:].copy()
    y_train, y_test = target.iloc[:test_idx].copy(), target.iloc[test_idx:].copy()

    return x_train, y_train, x_test, y_test


def tune_hyperparameters(
    x_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]
) -> dict[str, Any]:
    """
    Train a model by optimizing hyperparameters using Optuna.
    Parameters
    ----------
    x_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target values.
    params : Dict[str, Any]
        Configuration dictionary containing:
        - 'optuna_search': dict with keys:
            - 'n_trials': int, number of optimization trials to run
            - 'model': str, model name (e.g., 'CatBoost', 'RF')
            - 'model_params': dict, model-specific parameter search spaces
        - 'n_folds': int, number of CV folds (default: 3)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'best_params': Dict[str, Any], best hyperparameters found during optimization
        - 'cv_results': Dict[str, Any], cross-validation results from best model:
            - 'cv_mae': float, average MAE over CV folds
            - 'cv_rmse': float, average RMSE over CV folds
            - 'cv_mape': float, average MAPE over CV folds
    """
    seed = 42
    # 1) Create the study and optimize CV MAPE via objective()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, params),
        n_trials=params["optuna_search"]["n_trials"],
    )

    # 2) Get best parameters
    best_params = study.best_params

    # 3) Re-evaluate best model to get full CV results
    model_name = params["optuna_search"]["model"]

    cv_results = eval_model(
        x_train=x_train,
        y_train=y_train,
        n_splits=params["n_folds"],
        model_name=model_name,
        model_params=best_params,
        seed=seed,
    )

    return {
        "best_params": best_params,
        "cv_metrics": {
            "cv_mae": cv_results["cv_mae"],
            "cv_rmse": cv_results["cv_rmse"],
            "cv_mape": cv_results["cv_mape"],
        },
    }


def fit_best_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, Any],
    tuning_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Trains a model using optimized hyperparameters found during hyperparameter tuning.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training features. Should contain all feature columns used during
        hyperparameter optimization.
    y_train : pd.Series
        Training target values. Must have the same length as x_train.
    params : Dict[str, Any]
        Configuration dictionary containing:
        - 'optuna_search': dict with keys:
            - 'model': str, model name to train. Supported values: 'CatBoost', 'RF'
    tuning_results : Dict[str, Any]
        Dictionary from tune_hyperparameters containing:
        - 'best_params': Dict[str, Any], optimized hyperparameters
        - 'cv_results': Dict[str, Any], cross-validation results (optional, will be
          included in return if present)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Trained model instance (CatBoostRegressor or RandomForestRegressor)
        - 'best_params': Dict[str, Any], copy of the best_params used for training
        - 'x_scaler': StandardScaler, fitted scaler used to transform features
        - 'input_example': pd.DataFrame, first 5 rows of x_train (used for MLflow
          signature inference)
        - 'cv_results': Dict[str, Any], cross-validation results from best model
          (if present in tuning_results)
    """

    model_name = params["optuna_search"]["model"]

    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_train)

    if model_name == "CatBoost":
        model = CatBoostRegressor(
            **tuning_results["best_params"],
            allow_writing_files=False,
            verbose=False,
        )
    elif model_name == "RF":
        model = RF(**tuning_results["best_params"])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(x_scaled, y_train)

    y_pred_test = model.predict(x_scaler.transform(x_test))
    mae_err_test, rmse_err_test, mape_err_test = compute_metrics(y_test, y_pred_test)

    return {
        "model": model,
        "x_scaler": x_scaler,
        "input_example": x_train.iloc[:5].copy(),
        "test_metrics": {
            "test_mae": mae_err_test,
            "test_rmse": rmse_err_test,
            "test_mape": mape_err_test,
        },
    }


def log_to_mlflow(
    hyperparams_tuning_results: dict[str, Any],
    train_results: dict[str, Any],
    train_pipeline_params: dict[str, Any],
    mlflow_params: dict[str, Any],
) -> str:
    """
    Logs params to MLflow and creates a candidate model.
    """
    # Set MLflow tracking URI and experiment
    mlflow.set_experiment(mlflow_params["prod_experiment_name"])

    # Get model name and determine run name
    model_name = train_pipeline_params["optuna_search"]["model"]
    # Default run name with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}_{timestamp}_candidate"

    # Create temporary directory for artifacts
    temp_models_dir = Path("data/02_models")
    temp_models_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        # Log metrics and parameters
        mlflow.log_metrics(hyperparams_tuning_results["cv_metrics"])
        mlflow.log_metrics(train_results["test_metrics"])
        mlflow.log_params(hyperparams_tuning_results["best_params"])

        # Create signature from input example
        input_example = train_results["input_example"]
        x_scaler = train_results["x_scaler"]
        best_model = train_results["model"]
        y_example = best_model.predict(x_scaler.transform(input_example))
        signature = infer_signature(input_example, y_example)

        # Save model and scaler artifacts depending on the model type
        if model_name.lower() == "catboost":
            model_path = temp_models_dir / "catboost_model.cbm"
            best_model.save_model(str(model_path))
        elif model_name.lower() in ["rf", "random_forest"]:
            model_path = temp_models_dir / "random_forest_model.pkl"
            joblib.dump(best_model, model_path)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        scaler_path = temp_models_dir / "x_scaler.joblib"
        joblib.dump(x_scaler, scaler_path)

        # Prepare artifacts dictionary for MLModelWrapper
        artifacts = {
            "model": str(model_path),
            "scaler": str(scaler_path),
        }

        # Log PyFunc model using MLModelWrapper
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=MLModelWrapper(model_name=model_name),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            tags={
                "best_model": "true",
                "model_type": model_name,
                "run_name": run_name,
            },
        )
    return model_info.model_uri


def register_model(
    model_uri: str,
    registered_model_name: str,
) -> str:
    """
    Register a model in MLflow, add 'challenger' alias,
    and transition it to Staging.
    """
    client = MlflowClient()

    # 1) Register model
    model_info = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    version = str(model_info.version)

    # 2) Add alias
    client.set_registered_model_alias(
        name=registered_model_name, alias="challenger", version=version
    )
    return version


def validate_challenger(registered_model_name: str, model_version: str) -> None:
    """
    Validates challenger model against champion and promote if better (lower MAPE).

    Parameters
    ----------
    registered_model_name : str
        The name of the registered model in MLflow to validate.

    Returns
    -------
    None
        This function does not return any value. It modifies model aliases in MLflow.
    """
    client = MlflowClient()

    def get_test_mape(alias: str) -> float:
        """
        Helper function to retrieve test_mape metric from MLflow for a model alias.
        """
        info = client.get_model_version_by_alias(registered_model_name, alias)
        assert info.run_id is not None
        run = client.get_run(info.run_id)
        return float(run.data.metrics["test_mape"])

    challenger_mape = get_test_mape("challenger")

    # get champion mape and if not exists, set it to infinity, so challenger will be promoted
    try:
        champion_mape = get_test_mape("champion")
    except Exception:
        champion_mape = float("inf")

    if challenger_mape < champion_mape:
        client.delete_registered_model_alias(registered_model_name, "champion")
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=client.get_model_version_by_alias(
                registered_model_name, "challenger"
            ).version,
        )
        # delete challenger alias after promotion
        client.delete_registered_model_alias(registered_model_name, "challenger")
