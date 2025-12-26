"""Programmatic entrypoint for streaming data point-by-point to the database."""

import os
import sys
import time
from pathlib import Path

# Add src directory to path before imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)

import pandas as pd  # noqa: E402

from app_data_manager.data_manager import DataManager  # noqa: E402, type: ignore
from app_data_manager.utils import read_config  # noqa: E402, type: ignore


def stream_data_to_db(
    sleep_seconds: float = 1.0,
    table_name: str = "raw_data",
) -> None:
    """
    Stream data point-by-point to the database with a delay between each insertion.

    Args:
        sleep_seconds: Number of seconds to sleep between each data point insertion.
        table_name: Name of the table to insert data into.
    """
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config)

    # Initialize predictions table (only once)
    data_manager.init_predictions_db_table()

    # Load inference data from config
    inference_data_folder = config["data_manager"]["inference_data_folder"]
    inference_data_filename = config["data_manager"]["inference_data_filename"]
    data_path = os.path.join(
        project_root, inference_data_folder, inference_data_filename
    )
    inference_data = pd.read_parquet(data_path)

    # print(f"Starting to stream {len(inference_data)} data points to database...")
    # print(f"Sleep interval: {sleep_seconds} seconds between insertions")

    while True:
        # Clean database by reinitializing the raw data table
        data_manager.init_raw_db_table()

        # Iterate over each row and insert one at a time
        for idx, row in inference_data.iterrows():
            # Convert single row to DataFrame for insertion
            row_df = row.to_frame().T

            # Insert single row
            data_manager.insert_data_to_db(row_df, table_name=table_name)

            # print(
            #     f"Inserted row {idx + 1}/{len(inference_data)}: {row.get('Timestamps', 'N/A')}"
            # )

            # Sleep before next insertion
            if idx < len(inference_data) - 1:  # Don't sleep after last row
                time.sleep(sleep_seconds)

        # print(
        #     f"Successfully streamed all {len(inference_data)} data points to database!"
        # )


if __name__ == "__main__":
    stream_data_to_db()
