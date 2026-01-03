"""Programmatic entrypoint for running the Kedro training pipeline periodically."""

import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from loguru import logger  # noqa: E402

# Add src directory to path before imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)

from app_data_manager.utils import read_config  # noqa: E402, type: ignore
from common.mlflow_utils import get_latest_model_timestamp  # noqa: E402, type: ignore

# Read configuration
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Read environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


def run_training_pipeline(
    env: str = "local",
    pipeline_name: str = "training",
) -> None:
    """Run the Kedro training pipeline programmatically."""
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


def run_training_real_time() -> None:
    """
    Continuously check MLflow for latest model and run training if enough time has passed.

    Configuration is read from parameters.yml under the 'training_real_time' section.
    """
    # Get configuration from parameters.yml
    training_config = config["training_pipeline"]["training_real_time"]
    training_frequency = training_config["training_frequency"]
    check_interval_seconds = training_config["check_frequency"] * 60.0
    env = "local"  # Default Kedro environment
    model_name = config["mlflow"]["registered_model_name"]

    logger.info(
        f"Starting training monitor (checking every {training_config['check_frequency']} minutes, "
        f"training if {training_frequency} minutes passed since last model)..."
    )

    while True:
        try:
            # Get the timestamp of the latest model (champion or challenger)
            latest_timestamp = get_latest_model_timestamp(
                MLFLOW_TRACKING_URI, model_name
            )

            if latest_timestamp is None:
                logger.warning(
                    "No champion or challenger model found. "
                    "Running training to create initial model..."
                )
                logger.info("Starting training pipeline...")
                run_training_pipeline(env=env, pipeline_name="training")
                logger.info("Training completed successfully")
            else:
                # Calculate time elapsed since last model
                time_elapsed = datetime.now() - latest_timestamp
                time_elapsed_minutes = time_elapsed.total_seconds() / 60.0

                if time_elapsed_minutes >= training_frequency:
                    logger.info(
                        f"Time threshold reached ({time_elapsed_minutes:.2f} >= "
                        f"{training_frequency} minutes). Starting training pipeline..."
                    )
                    run_training_pipeline(env=env, pipeline_name="training")
                    logger.info("Training completed successfully")

            # Wait before next check
            time.sleep(check_interval_seconds)

        except Exception as e:
            logger.exception(f"Error during training check: {e}")
            time.sleep(check_interval_seconds)


if __name__ == "__main__":
    run_training_real_time()
