"""Project hooks."""

import os

from kedro.framework.hooks import hook_impl

import mlflow


class MLFlowHook:
    """Project hooks for MLflow tracking URI setup."""

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """Set MLflow tracking URI before pipeline runs."""
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
        mlflow.set_tracking_uri(tracking_uri)
