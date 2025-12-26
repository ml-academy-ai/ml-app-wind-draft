# Quick Start Guide

## How to Run the Dashboard

### Option 1: Direct Run (Recommended)

1. **Install dependencies:**
   ```bash
   cd src/app_ui
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   - Navigate to: `http://localhost:8050`
   - The dashboard will open automatically

### Option 2: Run from Project Root

```bash
# From project root directory
python -m src.app_ui.app
```

Or simply:

```bash
python src/app_ui/app.py
```

## Prerequisites

Make sure you have Python 3.11+ installed. The app requires:
- `dash==3.0.4`
- `dash-bootstrap-components==1.6.0`
- `plotly==6.1.2`
- `pandas==2.2.2`
- `numpy==1.26.4`
- `pyarrow==20.0.0`

## Troubleshooting

### Port Already in Use
If port 8050 is already in use, you can modify the port in `app.py`:
```python
app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8051)  # Change port
```

### Missing Data Files
The app will work with dummy data if actual data files are not found. To use real data:
- Place test data in: `data/01_raw/df_train_test.parquet`
- Place inference data in: `data/01_raw/df_inference.parquet`

### MLflow Connection
The Model Tracking page links to MLflow. Make sure MLflow is running:
```bash
mlflow ui --port 8080
```

## Next Steps

1. **Add Architecture Diagram:**
   - Place `architecture.png` or `diagram.png` in `docs/` or project root
   - It will automatically appear on the Architecture page

2. **Configure MLflow:**
   - Set `MLFLOW_TRACKING_URI` environment variable if needed
   - Default: `http://localhost:8080`

3. **Customize Data Sources:**
   - Modify data loading functions in `pages/home.py` and `pages/drift_monitoring.py`
   - Adjust column detection logic as needed

