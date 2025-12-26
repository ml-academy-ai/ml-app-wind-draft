import os
import sys
from pathlib import Path

import pandas as pd
from data_manager import DataManager  # type: ignore
from utils import read_config  # type: ignore

# Add project root and app_ui directory to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.chdir(project_root)


if __name__ == "__main__":
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config)

    data_manager.init_raw_db_table()
    data_manager.init_predictions_db_table()
    inference_data = pd.read_parquet(
        os.path.join(project_root, "data", "01_raw", "inference_data.parquet")
    )
    data_manager.insert_data_to_db(inference_data, table_name="raw_data")

    # Read data

    # df = data_manager.get_last_n_points(10, table_name="raw_data")

    # train_df = data_manager.get_data_by_timestamp_range(
    #     start_timestamp=config["training_pipeline"]["start_timestamp"],
    #     end_timestamp=config["training_pipeline"]["end_timestamp"],
    #     table_name="raw_data",
    # )
