import os
import sys
from pathlib import Path

# Add src directory to path before imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)

import pandas as pd  # noqa: E402

from app_data_manager.data_manager import DataManager  # noqa: E402, type: ignore
from app_data_manager.utils import read_config  # noqa: E402, type: ignore

if __name__ == "__main__":
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config)

    data_manager.init_raw_db_table()
    data_manager.init_predictions_db_table()
    inference_data = pd.read_parquet(
        os.path.join(project_root, "data", "01_raw", "inference_data.parquet")
    )
    data_manager.insert_data_to_db(inference_data, table_name="raw_data")
