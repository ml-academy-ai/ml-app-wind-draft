"""Project hooks."""

import os

import mlflow
from kedro.framework.hooks import hook_impl


class MLFlowHook:
    """Project hooks for MLflow tracking URI setup."""

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """Set MLflow tracking URI before pipeline runs.

        Uses environment variable if set, otherwise falls back to localhost:5001
        for local development. In Docker, the environment variable should be set
        to http://mlflow:5001 (internal service name).
        """
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
