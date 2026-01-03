# Lesson 3: Creating a Data Management Module

## Step 1: Create Directory Structure

Create `app-data-manager` directory under `src/`:
```
src/
  app_data_manager/
    __init__.py
    data_manager.py
    utils.py
    app.py
```

Create `history_data` folder at project root:
```
history_data/
```

## Step 2: Configure Paths

Add `history_data` paths to the config file (`conf/base/parameters.yml`):
```yaml
data_manager:
  history_data_path: "history_data/history_database.parquet"
  prod_data_path: "data/prod_data/prod_database.parquet"
  db_path: "data/sqlite/app.db"
  raw_data_table_name: "raw_data"
  predictions_table_name: "predictions"
```

## Step 3: Create DataManager Class

Create `DataManager` class in `src/app_data_manager/data_manager.py`:

```python
class DataManager:
    def __init__(self, config):
        self.config = config
        # Initialize database connection
        # Read paths from config
```

**Read the configs** from the parameters file.

## Step 4: Create init_raw_db_table() Method

Create `init_raw_db_table()` method that:
- Creates SQLite database if it doesn't exist
- Creates `raw_data` table with appropriate schema
- Handles table creation if it already exists

## Step 5: Create app.py and Initialize Database

Create `app.py` and run the `initialize_prod_db()` method:

```python
from app_data_manager.data_manager import DataManager
from app_data_manager.utils import read_config

config = read_config("conf/base/parameters.yml")
data_manager = DataManager(config)
data_manager.init_raw_db_table()
```

Run this to create the database and table.

## Step 6: Create init_predictions_db_table() Method

Create `init_predictions_db_table()` method and run it.

This creates a separate table for storing model predictions.

## Step 7: Create get_last_n_points() Method

Create `get_last_n_points()` method:
- Takes `n` as parameter
- Takes `table_name` as parameter
- Returns the last N rows from the specified table
- Orders by timestamp (most recent first)

Run it with `print` to verify it works:
```python
data = data_manager.get_last_n_points(100, table_name="raw_data")
print(data.head())
```

## Step 8: Create insert_data_to_db() Method

Create `insert_data_to_db()` method:
- Takes DataFrame and table name
- Inserts data into the specified table
- Handles duplicates (if needed)

## Step 9: Load and Insert Inference Data

Read `inference_data.parquet` from `data/01_raw`:
```python
import pandas as pd

df = pd.read_parquet("data/01_raw/inference_data.parquet")
data_manager.insert_data_to_db(df, table_name="raw_data")
print("Data appended to database")
```

Append data to db and print that it's appended.

## Step 10: Create get_data_by_timestamp_range() Method

Create `get_data_by_timestamp_range()` method:
- Takes `start_timestamp` and `end_timestamp`
- Takes `table_name` as parameter
- Returns data within the specified time range
- Handles timestamp parsing

## Step 11: Configure Training Pipeline for Date Range

Add to the `training_pipeline` config:
```yaml
training_pipeline:
  start_timestamp: '2007-07-29 03:10:00'
  end_timestamp: '2008-02-22 11:00:00'
```

Run in the `app.py` to test:
```python
data = data_manager.get_data_by_timestamp_range(
    start_timestamp='2007-07-29 03:10:00',
    end_timestamp='2008-02-22 11:00:00',
    table_name="raw_data"
)
print(f"Retrieved {len(data)} rows")
```

## Step 12: Implement load_training_data_from_db() Node

In the `feature_engineering_pipeline`, implement `load_training_data_from_db()` node:
- Reads from config: `start_timestamp` and `end_timestamp`
- Uses `DataManager.get_data_by_timestamp_range()`
- Returns loaded DataFrame

## Step 13: Create load_db_training_data() Pipeline

Create pipeline - `load_db_training_data()` and add this to `feature_engineering_pipeline` and test it.

**Consider making the training data smaller for faster iterations** (use a smaller date range in config for testing).

## Step 14: Create get_last_n_points_from_db() Node

Create `get_last_n_points_from_db()` node:
- Reads `batch_size` from config
- Uses `DataManager.get_last_n_points()`
- Returns loaded DataFrame

## Step 15: Configure Inference Pipeline

Add to the inference pipeline config:
```yaml
inference_pipeline:
  batch_size: 32
```

## Step 16: Implement load_inference_data_from_db() Pipeline

In the `feature_engineering_pipeline`, implement `load_inference_data_from_db()` pipeline:
- Uses `get_last_n_points_from_db()` node
- Reads `batch_size` from config
- Add this to the feature engineering pipeline
- Test the `inference_pipeline`

## Step 17: Implement save_predictions_to_db() Node

Implement `save_predictions_to_db()` node and run inference pipeline.

**Problem:** To do that, we need to first get the timestamps for the data that we predict on.

**Solution:** 
1. In the inference pipeline, implement `get_data_timestamps()` node
2. This node extracts timestamps from the input data
3. Then consume this data in the `save_predictions()` node in the inference pipeline

**Implementation:**
```python
def get_data_timestamps(data: pd.DataFrame) -> pd.Series:
    """Extract timestamps from data."""
    return data["Timestamps"]

def save_predictions_to_db(
    predictions: np.ndarray,
    timestamps: pd.Series,
    config: dict
):
    """Save predictions to database."""
    # Create DataFrame with predictions and timestamps
    df = pd.DataFrame({
        "Timestamps": timestamps,
        "predicted_power": predictions
    })
    
    # Use DataManager to insert
    data_manager = DataManager(config)
    data_manager.insert_data_to_db(df, table_name="predictions")
```

## Step 18: Test Complete Pipeline

Run the complete inference pipeline:
```bash
kedro run --pipeline=inference
```

Verify:
- Data loads from database
- Predictions are generated
- Predictions are saved back to database
- Timestamps match correctly

---

## Summary

**What we built:**
1. **DataManager class**: Centralized database operations
2. **Database initialization**: Tables for raw data and predictions
3. **Data loading methods**: By timestamp range and by number of points
4. **Data insertion**: Saving predictions back to database
5. **Pipeline integration**: Connecting Kedro pipelines to database

**Key concepts:**
- SQLite for local data storage
- Timestamp-based queries
- Batch processing for inference
- Data lineage tracking


