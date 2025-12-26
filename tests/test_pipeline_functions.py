import pandas as pd

from ml_app_wind_draft.pipelines.feature_eng.nodes import (
    add_lag_features,
    remove_diff_outliers,
)
from ml_app_wind_draft.pipelines.inference.nodes import (
    compute_metrics,
)

# Note: Fixtures (dataset_with_outliers, training_data, inference_data) are defined
# in conftest.py and are automatically available to all test files.


def test_remove_diff_outliers_removes_large_jumps(dataset_with_outliers):
    """
    Unit Test 1: Test that remove_diff_outliers correctly identifies and removes outliers
    based on absolute first-order differences.
    """
    df = dataset_with_outliers.copy()

    # Set threshold for power column (should catch the large jump at index 75)
    diff_thresholds = {"power": 30.0}

    # Apply outlier removal
    df_cleaned = remove_diff_outliers(df, diff_thresholds)

    # Verify that the large jump was removed (value at index 75 should be filled)
    # The diff between index 74 and 75 was 50, which exceeds threshold of 30
    assert df_cleaned.loc[75, "power"] != df.loc[75, "power"], (
        "Outlier at index 75 should have been removed and forward-filled"
    )

    # Verify forward-fill worked (no NaN values in cleaned data)
    assert not df_cleaned["power"].isna().any(), (
        "Cleaned power column should not contain NaN values"
    )

    # Verify the cleaned data has the same shape
    assert df_cleaned.shape == df.shape, (
        "Cleaned DataFrame should have the same shape as input"
    )


def test_add_lag_features_creates_correct_lags(dataset_with_outliers):
    """
    Unit Test 2: Test that add_lag_features correctly creates lag features.
    """
    df = dataset_with_outliers.copy()

    # Define lag features to create
    lags_dict = {
        "power": [1, 2, 3],
        "wind_speed": [1],
    }

    df_with_lags = add_lag_features(df, lags_dict, drop_na=False)

    # Verify new columns were created
    expected_columns = [
        "power_lag1",
        "power_lag2",
        "power_lag3",
        "wind_speed_lag1",
    ]
    for col in expected_columns:
        assert col in df_with_lags.columns, (
            f"Expected lag column {col} not found in DataFrame"
        )

    # Verify lag values are correct (power_lag1 should equal power shifted by 1)
    # At index 1, power_lag1 should equal power at index 0
    assert df_with_lags.loc[1, "power_lag1"] == df.loc[0, "power"], (
        "power_lag1 at index 1 should equal power at index 0"
    )

    # Verify power_lag2 at index 2 equals power at index 0
    assert df_with_lags.loc[2, "power_lag2"] == df.loc[0, "power"], (
        "power_lag2 at index 2 should equal power at index 0"
    )


def test_compute_metrics_correct_calculation(dataset_with_outliers):
    """
    Unit Test 3: Test that compute_metrics correctly calculates MAE, RMSE, and MAPE.
    """
    # Create simple test data (perfect predictions for easy verification)
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])  # Perfect predictions

    metrics = compute_metrics(y_true, y_pred)

    # Verify return type
    assert isinstance(metrics, dict), "compute_metrics should return a dictionary"

    # Verify expected keys
    expected_keys = ["mae", "rmse", "mape"]
    for key in expected_keys:
        assert key in metrics, f"Expected metric {key} not found in results"

    # For perfect predictions, all metrics should be 0
    assert metrics["mae"] == 0.0, "MAE should be 0 for perfect predictions"
    assert metrics["rmse"] == 0.0, "RMSE should be 0 for perfect predictions"
    assert metrics["mape"] == 0.0, "MAPE should be 0 for perfect predictions"
