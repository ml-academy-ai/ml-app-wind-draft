"""Programmatic entrypoint for running inference pipeline when new data is available."""

import logging
import os
import sys
import time
import tomllib
from pathlib import Path

# Add src directory to path before imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.append(str(project_root))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)


from kedro.framework.project import configure_project  # noqa: E402
from kedro.framework.session import KedroSession  # noqa: E402
from kedro.framework.startup import bootstrap_project  # noqa: E402

from app_data_manager.data_manager import DataManager  # noqa: E402, type: ignore
from app_data_manager.utils import read_config  # noqa: E402, type: ignore

logger = logging.getLogger(__name__)


def get_latest_timestamp(
    data_manager: DataManager, table_name: str = "raw_data"
) -> str | None:
    """Get the latest timestamp from the specified table."""
    df = data_manager.get_last_n_points(1, table_name=table_name)
    if df.empty:
        return None
    return str(df.iloc[-1]["Timestamps"])


def run_inference_pipeline(
    env: str = "local", pipeline_name: str = "inference"
) -> None:
    """Run the Kedro inference pipeline programmatically."""
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


def run_inference_real_time(
    check_interval_seconds: float = 5.0,
    env: str = "local",
) -> None:
    """
    Continuously check for new data and run inference pipeline when new data is available.

    Args:
        check_interval_seconds: Number of seconds to wait between checks for new data.
        env: Kedro environment name.
    """
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config)

    # Initialize predictions table if needed
    data_manager.init_predictions_db_table()

    logger.info(
        f"Starting inference monitor (checking every {check_interval_seconds} seconds)..."
    )
    logger.info("Press Ctrl+C to stop")

    last_processed_timestamp: str | None = None

    while True:
        try:
            # Get latest timestamp from raw_data table
            latest_timestamp = get_latest_timestamp(data_manager, table_name="raw_data")

            if latest_timestamp is None:
                logger.info("No data in raw_data table yet. Waiting...")
            elif latest_timestamp != last_processed_timestamp:
                logger.info(f"New data detected! Latest timestamp: {latest_timestamp}")
                logger.info("Running inference pipeline...")

                # Run inference pipeline
                run_inference_pipeline(env=env, pipeline_name="inference")

                # Update last processed timestamp
                last_processed_timestamp = latest_timestamp
                logger.info(
                    f"Inference completed. Last processed timestamp: {last_processed_timestamp}"
                )
            else:
                logger.debug(f"No new data. Last timestamp: {latest_timestamp}")

            # Wait before next check
            time.sleep(check_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Inference monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Error during inference check: {e}", exc_info=True)
            time.sleep(check_interval_seconds)


if __name__ == "__main__":
    run_inference_real_time()
