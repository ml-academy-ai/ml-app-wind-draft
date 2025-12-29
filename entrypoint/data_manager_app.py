import os
from pathlib import Path

import pandas as pd

from app_data_manager.data_manager import DataManager  # noqa: E402, type: ignore
from app_data_manager.utils import read_config  # noqa: E402, type: ignore

project_root = Path(__file__).resolve().parents[1]
os.chdir(project_root)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config)

    data_manager.init_raw_db_table()
    data_manager.init_predictions_db_table()
    inference_data = pd.read_parquet(
        os.path.join(project_root, "data", "01_raw", "inference_data.parquet")
    )
    data_manager.insert_data_to_db(inference_data, table_name="raw_data")
