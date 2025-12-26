# Pytest Tutorial: Step-by-Step Guide to Writing Tests

This tutorial provides a complete step-by-step process for creating and writing tests using pytest. You'll learn how to test pipeline functions with unit tests and integration tests.

## Table of Contents

1. [Introduction to Pytest](#introduction-to-pytest)
2. [Setting Up the Test Environment](#setting-up-the-test-environment)
3. [Step 1: Create Test Directory Structure](#step-1-create-test-directory-structure)
4. [Step 2: Create Your First Fixture](#step-2-create-your-first-fixture)
5. [Step 3: Write Your First Unit Test](#step-3-write-your-first-unit-test)
6. [Step 4: Create conftest.py for Shared Fixtures](#step-4-create-conftestpy-for-shared-fixtures)
7. [Step 5: Write Multiple Unit Tests](#step-5-write-multiple-unit-tests)
8. [Step 6: Write Integration Tests](#step-6-write-integration-tests)
9. [Step 7: Running Tests](#step-7-running-tests)
10. [Best Practices and Tips](#best-practices-and-tips)

---

## Introduction to Pytest

**Pytest** is a Python testing framework that makes it easy to write simple and scalable tests. Key features:

- **Fixtures**: Reusable test data and setup code
- **Parametrization**: Run the same test with different inputs
- **Assertions**: Simple assert statements (no need to remember many assertion methods)
- **Automatic discovery**: Finds and runs tests automatically
- **Rich output**: Clear error messages and test reports

### Key Concepts

- **Test function**: Any function starting with `test_` is a test
- **Fixture**: A function decorated with `@pytest.fixture` that provides test data
- **Assertion**: Using `assert` to verify expected behavior
- **conftest.py**: Special file where shared fixtures are stored

---

## Setting Up the Test Environment

### Prerequisites

1. Ensure pytest is installed:
   ```bash
   pip install pytest pytest-cov
   ```

2. Verify installation:
   ```bash
   pytest --version
   ```

### Project Structure

Your project should have a `tests/` directory at the root level:

```
ml-app-wind-draft/
├── src/
│   └── ml_app_wind_draft/
│       └── pipelines/
│           ├── feature_eng/
│           ├── training/
│           └── inference/
├── tests/
│   ├── conftest.py
│   └── test_pipeline_functions.py
└── pyproject.toml
```

---

## Step 1: Create Test Directory Structure

1. **Create the tests directory** (if it doesn't exist):
   ```bash
   mkdir tests
   ```

2. **Create an empty `__init__.py`** (optional, but good practice):
   ```bash
   touch tests/__init__.py
   ```

3. **Verify your project structure**:
   ```bash
   tree tests/  # or ls tests/
   ```

---

## Step 2: Create Your First Fixture

**What is a fixture?** A fixture is a function that provides test data. It runs before each test that uses it.

### Create `conftest.py`

Create `tests/conftest.py` with your first fixture:

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dataset_with_outliers():
    """
    Fixture that generates a synthetic dataset with outliers.
    
    This fixture creates a time-series dataset with:
    - Normal time-series data (wind power, wind speed, gen rpm)
    - Intentional outliers (spikes, drops, extreme values)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['power', 'wind_speed', 'gen_rpm', 'Timestamps']
        Contains intentional outliers for testing outlier removal functions.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate base time-series data (100 data points)
    n_samples = 100
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="h")
    
    # Create normal time-series data
    power = 50 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    wind_speed = 10 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    gen_rpm = 15 + 10 * np.cos(np.linspace(0, 4 * np.pi, n_samples))
    noise = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(
        {
            "power": power + noise,
            "wind_speed": wind_speed + noise,
            "gen_rpm": gen_rpm + noise,
            "Timestamps": timestamps,
        }
    )
    
    # Introduce intentional outliers at specific indices
    # 1. Extreme spike in power (index 20)
    df.loc[20, "power"] = 200  # Normal range is ~30-70
    
    # 2. Sudden drop in wind_speed (index 45)
    df.loc[45, "wind_speed"] = -5  # Negative value (impossible)
    
    # 3. Extreme gen_rpm value (index 60)
    df.loc[60, "gen_rpm"] = 100  # Normal range is ~5-25
    
    # 4. Large jump in power (index 75) - creates large diff
    df.loc[75, "power"] = df.loc[74, "power"] + 50
    
    return df
```

### Key Points About Fixtures

- **`@pytest.fixture`**: Decorator that marks a function as a fixture
- **Automatic discovery**: Fixtures in `conftest.py` are automatically available to all test files
- **Scope**: By default, fixtures run before each test (function scope)
- **Return value**: The return value is passed to the test function as a parameter

---

## Step 3: Write Your First Unit Test

### Create `test_pipeline_functions.py`

Create `tests/test_pipeline_functions.py` with your first test:

```python
import pandas as pd

from ml_app_wind_draft.pipelines.feature_eng.nodes import remove_diff_outliers


def test_remove_diff_outliers_removes_large_jumps(dataset_with_outliers):
    """
    Unit Test 1: Test that remove_diff_outliers correctly identifies and removes outliers
    based on absolute first-order differences.
    """
    # Use the fixture - it's automatically passed as a parameter
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
```

### Understanding the Test

1. **Function name**: Must start with `test_` for pytest to discover it
2. **Fixture parameter**: `dataset_with_outliers` is automatically injected by pytest
3. **Assertions**: Use `assert` with descriptive error messages
4. **Test structure**: Arrange → Act → Assert pattern

### Run Your First Test

```bash
pytest tests/test_pipeline_functions.py::test_remove_diff_outliers_removes_large_jumps -v
```

Expected output:
```
tests/test_pipeline_functions.py::test_remove_diff_outliers_removes_large_jumps PASSED
```

---

## Step 4: Create conftest.py for Shared Fixtures

**Why conftest.py?** It's a special file where pytest automatically looks for fixtures. Fixtures defined here are available to all test files without imports.

### Complete conftest.py Example

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dataset_with_outliers():
    """
    Fixture that generates a synthetic dataset with outliers.
    """
    np.random.seed(42)
    
    n_samples = 100
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="h")
    
    power = 50 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    wind_speed = 10 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    gen_rpm = 15 + 10 * np.cos(np.linspace(0, 4 * np.pi, n_samples))
    noise = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(
        {
            "power": power + noise,
            "wind_speed": wind_speed + noise,
            "gen_rpm": gen_rpm + noise,
            "Timestamps": timestamps,
        }
    )
    
    # Add outliers
    df.loc[20, "power"] = 200
    df.loc[45, "wind_speed"] = -5
    df.loc[60, "gen_rpm"] = 100
    df.loc[75, "power"] = df.loc[74, "power"] + 50
    
    return df
```

### Key Benefits

- **No imports needed**: Fixtures are automatically available
- **Shared across tests**: One fixture can be used by many test files
- **Centralized**: All test data setup in one place

---

## Step 5: Write Multiple Unit Tests

### Test 2: Testing Lag Features

Add another test to `test_pipeline_functions.py`:

```python
from ml_app_wind_draft.pipelines.feature_eng.nodes import add_lag_features


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
    
    # Verify lag values are correct
    # At index 1, power_lag1 should equal power at index 0
    assert df_with_lags.loc[1, "power_lag1"] == df.loc[0, "power"], (
        "power_lag1 at index 1 should equal power at index 0"
    )
    
    # Verify power_lag2 at index 2 equals power at index 0
    assert df_with_lags.loc[2, "power_lag2"] == df.loc[0, "power"], (
        "power_lag2 at index 2 should equal power at index 0"
    )
```

### Test 3: Testing Metrics

Add a test for metric computation:

```python
from ml_app_wind_draft.pipelines.inference.nodes import compute_metrics


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
```

### Complete Test File Structure

Your `test_pipeline_functions.py` should now look like:

```python
import pandas as pd

from ml_app_wind_draft.pipelines.feature_eng.nodes import (
    add_lag_features,
    remove_diff_outliers,
)
from ml_app_wind_draft.pipelines.inference.nodes import (
    compute_metrics,
)

# Note: Fixtures are defined in conftest.py and are automatically available


def test_remove_diff_outliers_removes_large_jumps(dataset_with_outliers):
    """Unit Test 1: Test outlier removal."""
    # ... test code ...


def test_add_lag_features_creates_correct_lags(dataset_with_outliers):
    """Unit Test 2: Test lag feature creation."""
    # ... test code ...


def test_compute_metrics_correct_calculation(dataset_with_outliers):
    """Unit Test 3: Test metric computation."""
    # ... test code ...
```

---

## Step 6: Write Integration Tests

**Integration tests** verify that multiple components work together correctly.

### Example: Testing Training Pipeline

```python
import pytest
from unittest.mock import MagicMock, patch

from ml_app_wind_draft.pipelines.training.nodes import (
    fit_best_model,
    log_to_mlflow,
    register_model,
    train_test_split,
    tune_hyperparameters,
    validate_challenger,
)


@patch("ml_app_wind_draft.pipelines.training.nodes.mlflow")
@patch("ml_app_wind_draft.pipelines.training.nodes.MlflowClient")
def test_training_pipeline_integration(
    mock_mlflow_client_class,
    mock_mlflow,
    training_data,  # Another fixture you'd create
):
    """
    Integration Test: Test the complete training pipeline workflow.
    """
    features, target = training_data
    
    # Mock MLflow components
    mock_client = MagicMock()
    mock_mlflow_client_class.return_value = mock_client
    
    # ... rest of test code ...
```

### Key Concepts for Integration Tests

1. **Mocking**: Use `unittest.mock` to replace external dependencies
2. **End-to-end**: Test the complete workflow, not just individual functions
3. **Real logic**: Test actual business logic, mock external services

---

## Step 7: Running Tests

### Basic Commands

**Run all tests:**
```bash
pytest tests/
```

**Run a specific test file:**
```bash
pytest tests/test_pipeline_functions.py
```

**Run a specific test:**
```bash
pytest tests/test_pipeline_functions.py::test_remove_diff_outliers_removes_large_jumps
```

**Verbose output:**
```bash
pytest tests/ -v
```

**Show print statements:**
```bash
pytest tests/ -s
```

**Stop on first failure:**
```bash
pytest tests/ -x
```

**Run with coverage:**
```bash
pytest tests/ --cov=ml_app_wind_draft.pipelines --cov-report=term-missing
```

### Understanding Test Output

**Successful test:**
```
tests/test_pipeline_functions.py::test_remove_diff_outliers_removes_large_jumps PASSED
```

**Failed test:**
```
tests/test_pipeline_functions.py::test_remove_diff_outliers_removes_large_jumps FAILED
...
AssertionError: Outlier at index 75 should have been removed and forward-filled
```

---

## Best Practices and Tips

### 1. Test Naming

- Use descriptive names: `test_remove_diff_outliers_removes_large_jumps`
- Start with `test_` prefix
- Use underscores, not hyphens

### 2. Test Organization

- One test per function/behavior
- Group related tests in the same file
- Use docstrings to explain what each test does

### 3. Assertions

- Always include descriptive error messages:
  ```python
  assert condition, "Clear explanation of what failed"
  ```
- Test one thing per assertion
- Use specific assertions (check exact values, not just "not None")

### 4. Fixtures

- Put shared fixtures in `conftest.py`
- Use meaningful fixture names
- Document what each fixture provides
- Set random seeds for reproducibility

### 5. Test Data

- Use synthetic data for unit tests
- Make test data realistic but controlled
- Include edge cases (outliers, missing values, etc.)

### 6. Test Independence

- Each test should be independent
- Don't rely on test execution order
- Clean up after tests if needed

### 7. Mocking

- Mock external services (databases, APIs, MLflow)
- Don't mock the code you're testing
- Verify that mocked functions were called

### 8. Coverage

- Aim for high coverage of critical functions
- Don't obsess over 100% coverage
- Focus on testing business logic

---

## Common Patterns

### Pattern 1: Testing Data Transformations

```python
def test_function_transforms_data_correctly(dataset_with_outliers):
    df = dataset_with_outliers.copy()
    result = transform_function(df)
    
    assert result.shape[0] == df.shape[0]  # Same number of rows
    assert "new_column" in result.columns  # New column created
    assert not result["new_column"].isna().any()  # No NaN values
```

### Pattern 2: Testing Error Cases

```python
def test_function_raises_error_on_invalid_input(dataset_with_outliers):
    df = dataset_with_outliers.copy()
    
    with pytest.raises(ValueError, match="expected error message"):
        function_with_error(df, invalid_param="bad_value")
```

### Pattern 3: Testing Multiple Scenarios

```python
@pytest.mark.parametrize("threshold,expected_count", [
    (10.0, 5),
    (20.0, 3),
    (50.0, 1),
])
def test_outlier_removal_with_different_thresholds(
    dataset_with_outliers, threshold, expected_count
):
    df = dataset_with_outliers.copy()
    result = remove_outliers(df, threshold)
    
    assert len(result) == expected_count
```

---

## Troubleshooting

### Problem: Fixture not found

**Solution**: Make sure fixture is in `conftest.py` or imported correctly

### Problem: Import errors

**Solution**: Check that your project structure matches the imports

### Problem: Tests pass locally but fail in CI

**Solution**: Check for hardcoded paths, ensure random seeds are set

### Problem: Slow tests

**Solution**: Use smaller datasets, mock expensive operations, use fixtures with appropriate scope

---

## Summary

1. **Create `conftest.py`** with shared fixtures
2. **Write test functions** starting with `test_`
3. **Use fixtures** by adding them as function parameters
4. **Write assertions** with clear error messages
5. **Run tests** with `pytest tests/`
6. **Mock external dependencies** in integration tests
7. **Follow best practices** for maintainable tests

---

## Next Steps

- Add more unit tests for other pipeline functions
- Create integration tests for complete workflows
- Add parametrized tests for multiple scenarios
- Set up continuous integration (CI) to run tests automatically
- Explore pytest plugins (pytest-mock, pytest-asyncio, etc.)

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

