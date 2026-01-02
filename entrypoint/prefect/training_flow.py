import logging
import os
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, task

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))
os.chdir(project_root)

from app_data_manager.utils import read_config  # noqa: E402, type: ignore
from common.mlflow_utils import get_latest_model_timestamp  # noqa: E402, type: ignore

logger = logging.getLogger(__name__)

# Read configuration
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Read environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


@task(name="check-latest-model-timestamp")
def check_latest_model_timestamp_task(
    mlflow_tracking_uri: str, model_name: str
) -> datetime | None:
    """Check MLflow for the latest model timestamp (champion or challenger)."""
    return get_latest_model_timestamp(mlflow_tracking_uri, model_name)


@task(name="should-train")
def should_train_task(
    latest_timestamp: datetime | None, training_frequency: float
) -> bool:
    """
    Determine if training should run based on the latest model timestamp.

    Returns True if:
    - No model exists (latest_timestamp is None), or
    - Enough time has passed since the last model (>= training_frequency minutes)
    """
    if latest_timestamp is None:
        logger.warning(
            "No champion or challenger model found. Training should run to create initial model."
        )
        return True

    time_elapsed = datetime.now() - latest_timestamp
    time_elapsed_minutes = time_elapsed.total_seconds() / 60.0

    if time_elapsed_minutes >= training_frequency:
        logger.info(
            f"Time threshold reached ({time_elapsed_minutes:.2f} >= "
            f"{training_frequency} minutes). Training should run."
        )
        return True

    logger.info(
        f"Time threshold not reached ({time_elapsed_minutes:.2f} < "
        f"{training_frequency} minutes). Skipping training."
    )
    return False


@task(name="training-task")
def training_task(env: str = "local", pipeline_name: str = "training"):
    """Prefect task to run the Kedro training pipeline."""
    logger.info("Starting training pipeline...")
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)

    logger.info("Training completed successfully")


@flow(name="training-flow")
def training_flow(env: str = "local"):
    """
    Prefect flow for training that checks MLflow and runs training if needed.

    Configuration is read from parameters.yml under the 'training_real_time' section.
    """
    # Get configuration from parameters.yml
    training_config = config["training_pipeline"]["training_real_time"]
    training_frequency = training_config["training_frequency"]
    model_name = config["mlflow"]["registered_model_name"]

    logger.info(
        f"Training flow started (training if {training_frequency} minutes passed since last model)..."
    )

    # Check for latest model timestamp
    latest_timestamp = check_latest_model_timestamp_task(
        MLFLOW_TRACKING_URI, model_name
    )

    # Decide if training should run
    should_train = should_train_task(latest_timestamp, training_frequency)

    # Run training if needed
    if should_train:
        training_task(env=env)
    else:
        logger.info("Training skipped - not enough time has passed since last model.")


if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    # Get check frequency from parameters.yml
    training_config = config["training_pipeline"]["training_real_time"]
    check_frequency_minutes = training_config["check_frequency"]

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=check_frequency_minutes),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )

    # training_flow.serve(
    #     name="training-flow",
    #     interval=timedelta(minutes=1),
    #     parameters={"env": os.getenv("KEDRO_ENV", "local")},
    # )

# if __name__ == "__main__":
#     os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
#     training_flow(env="local")
