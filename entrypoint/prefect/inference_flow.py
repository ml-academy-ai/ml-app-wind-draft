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

from app_data_manager.data_manager import DataManager  # noqa: E402, type: ignore
from app_data_manager.utils import read_config  # noqa: E402, type: ignore

logger = logging.getLogger(__name__)

# Read configuration
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)


@task(name="get-latest-raw-timestamp")
def get_latest_raw_timestamp_task(table_name: str = "raw_data") -> str | None:
    """Get the latest timestamp from the raw_data table."""
    try:
        data_manager = DataManager(config)
        df = data_manager.get_last_n_points(1, table_name=table_name)
        if df.empty:
            return None
        return str(df.iloc[-1]["Timestamps"])
    except Exception as e:
        logger.error(f"Error getting latest raw timestamp: {e}")
        return None


@task(name="get-latest-prediction-timestamp")
def get_latest_prediction_timestamp_task(
    table_name: str = "predictions",
) -> datetime | None:
    """Get the latest timestamp from the predictions table as datetime."""
    try:
        data_manager = DataManager(config)
        df = data_manager.get_last_n_points(1, table_name=table_name)
        if df.empty:
            return None
        timestamp_str = str(df.iloc[-1]["Timestamps"])
        # Parse timestamp string (format: "%Y-%m-%d %H:%M:%S")
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error getting latest prediction timestamp: {e}")
        return None


@task(name="should-run-inference")
def should_run_inference_task(
    latest_raw_timestamp: str | None,
    last_processed_timestamp: datetime | None,
    inference_frequency: float,
) -> bool:
    """
    Determine if inference should run based on timestamp comparison and frequency.

    Returns True if:
    - No raw data exists (latest_raw_timestamp is None) - skip
    - No predictions exist yet (last_processed_timestamp is None) AND there's new data - run
    - Latest raw timestamp is greater than last processed AND enough time has passed - run
    """
    if latest_raw_timestamp is None:
        logger.debug("No data in raw_data table yet. Skipping inference.")
        return False

    # Check if there's new data
    has_new_data = False
    if last_processed_timestamp is None:
        has_new_data = True
        logger.info(
            f"No predictions found. New data detected! Latest timestamp: {latest_raw_timestamp}"
        )
    else:
        # Compare timestamps (both are strings in "%Y-%m-%d %H:%M:%S" format)
        try:
            last_processed_str = last_processed_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if latest_raw_timestamp > last_processed_str:
                has_new_data = True
        except Exception as e:
            logger.warning(
                f"Error comparing timestamps: {e}. Assuming new data exists."
            )
            has_new_data = True

    if not has_new_data:
        logger.debug(f"No new data. Last timestamp: {latest_raw_timestamp}")
        return False

    # Check if enough time has passed since last inference
    if last_processed_timestamp is None:
        # No previous inference, so we can run
        return True

    time_elapsed = datetime.now() - last_processed_timestamp
    time_elapsed_seconds = time_elapsed.total_seconds()

    if time_elapsed_seconds >= inference_frequency:
        logger.info(
            f"Time threshold reached ({time_elapsed_seconds:.2f} >= "
            f"{inference_frequency} seconds). Inference should run."
        )
        return True

    logger.info(
        f"Time threshold not reached ({time_elapsed_seconds:.2f} < "
        f"{inference_frequency} seconds). Skipping inference."
    )
    return False


@task(name="inference-task")
def inference_task(env: str = "local", pipeline_name: str = "inference"):
    """Prefect task to run the Kedro inference pipeline."""
    logger.info("Starting inference pipeline...")
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)

    logger.info("Inference completed successfully")


@task(name="init-predictions-table")
def init_predictions_table_task():
    """Initialize predictions table if needed."""
    data_manager = DataManager(config)
    data_manager.init_predictions_db_table()


@flow(name="inference-flow")
def inference_flow(env: str = "local"):
    """
    Prefect flow for inference that checks for new data and runs inference if needed.

    Configuration is read from parameters.yml under the 'inference_pipeline' section.
    """
    # Get configuration from parameters.yml
    inference_config = config["inference_pipeline"]
    inference_frequency = inference_config["inference_frequency"]

    logger.info(
        f"Inference flow started (inference if {inference_frequency} seconds passed since last inference and new data available)..."
    )

    # Initialize predictions table
    init_predictions_table_task()

    # Get latest timestamps
    latest_raw_timestamp = get_latest_raw_timestamp_task("raw_data")
    last_processed_timestamp = get_latest_prediction_timestamp_task("predictions")

    # Decide if inference should run
    should_run = should_run_inference_task(
        latest_raw_timestamp, last_processed_timestamp, inference_frequency
    )

    # Run inference if needed
    if should_run:
        inference_task(env=env)
        logger.info(
            f"Inference completed. Last processed timestamp: {latest_raw_timestamp}"
        )
    else:
        logger.info("Inference skipped - conditions not met.")


if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    inference_flow.serve(
        name="inference-flow",
        interval=timedelta(seconds=config["inference_pipeline"]["inference_frequency"]),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )
