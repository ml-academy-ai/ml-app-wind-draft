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
