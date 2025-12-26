# Wind Power Prediction Dashboard

Multi-page Dash application for ML model monitoring and visualization.

## Features

1. **Model Prediction Page**
   - Time series plot showing predictions vs true values on test set
   - Error metrics (MAE and MAPE) with rolling averages
   - Configurable lookback window
   - Configurable rolling window for error smoothing

2. **Model Drift Monitoring**
   - PCA reconstruction error visualization
   - Rolling mean and threshold indicators
   - Statistical summary

3. **Architecture Diagram**
   - Display application architecture as PNG or GIF
   - Place diagram in one of these locations:
     - `docs/architecture.png` or `docs/architecture.gif`
     - `docs/diagram.png` or `docs/diagram.gif`
     - `architecture.png` or `diagram.png` in project root

4. **Model Tracking & Registry**
   - Direct link to MLflow Tracking UI
   - Quick access to experiments, models, and runs
   - MLflow server status information

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

### Simple Method (Recommended)

From the `src/app_ui` directory:

```bash
cd src/app_ui
python app.py
```

Or from the project root:

```bash
python -m src.app_ui.app
```

The app will be available at `http://localhost:8050`

### Using Python directly

```bash
python src/app_ui/app.py
```

## Configuration

### MLflow Tracking URI

Set the MLflow tracking URI via environment variable:

```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
```

Or it defaults to `http://localhost:8080` if not set.

### Data Sources

The app reads data from the `data/` directory:
- Test data: `data/01_raw/df_train_test.parquet`
- Inference/drift data: `data/01_raw/df_inference.parquet`

The app will automatically detect datetime and target columns, or use defaults if not found.

## Project Structure

```
src/app_ui/
├── app.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── assets/
│   └── custom.css         # Custom styling
└── pages/
    ├── __init__.py
    ├── home.py            # Model Prediction page with time series and errors
    ├── drift_monitoring.py # Model drift monitoring page
    ├── architecture.py    # Architecture diagram page
    └── model_tracking.py  # MLflow tracking page
```

## Design

The app uses:
- **Dash** for the web framework
- **Dash Bootstrap Components** for UI components
- **Plotly** for interactive visualizations
- Bootstrap theme for consistent styling

Design inspiration taken from `ml-project-blueprint-real-time/app_ui`.

