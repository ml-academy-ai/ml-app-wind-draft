# Lesson: Building the Dash UI Application

## Setup and Installation

### Step 1: Install Dependencies
```bash
uv add dash dash_bootstrap_components pyyaml pyarrow python-dateutil gunicorn
```

Explain:
- `dash`: Core framework for building web applications
- `dash_bootstrap_components`: Pre-built UI components styled with Bootstrap
- `pyyaml`: For reading configuration files
- `pyarrow`: For Parquet file support (if needed)
- `python-dateutil`: For date parsing
- `gunicorn`: Production WSGI server (optional, for deployment)

### Step 2: Create Directory Structure
Create the following structure under `src/`:
```
src/
  app_ui/
    __init__.py
    app.py
    pages/
      __init__.py
      home.py
      model_tracking.py
      ml_pipelines.py
      architecture.py
    components/
      __init__.py
      sidebar.py
    assets/
      custom.css
      responsive-sidebar.css
    utils.py
```

Explain:
- `app.py`: Main application entry point
- `pages/`: Each file becomes a separate page in the multi-page app
- `components/`: Reusable UI components (like sidebar)
- `assets/`: CSS files (automatically loaded by Dash)
- `utils.py`: Shared utility functions

## Lesson 1: Basic App Structure

### Step 1: Create Main App File (`app.py`)
Create `src/app_ui/app.py`:

```python
import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

# Path setup for working directory
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Font Awesome for icons
FA = "https://use.fontawesome.com/releases/v5.15.1/css/all.css"

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, FA],
    suppress_callback_exceptions=True,
    use_pages=True,
    pages_folder="pages",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Basic layout (we'll add sidebar later)
app.layout = html.Div(
    id="main-layout",
    children=[
        dash.page_container,
    ],
)

server = app.server

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8050)
```

Explain:
- `use_pages=True`: Enables multi-page functionality
- `pages_folder="pages"`: Tells Dash where to find page files
- `suppress_callback_exceptions=True`: Allows callbacks in page files
- `dash.page_container`: Where page content will be rendered

### Step 2: Create First Page (`home.py`)
Create `src/app_ui/pages/home.py`:

```python
import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.H1("Model Prediction Dashboard"),
        html.P("This is the home page"),
    ],
    fluid=True,
)
```

Explain:
- `dash.register_page()`: Registers this file as a page
- `path="/"`: Makes it the root/home page
- `layout`: Variable that Dash looks for to render the page

Run the app: `python -m src.app_ui.app` and verify the page loads.

## Lesson 2: Building the Sidebar Component

### Step 1: Create Sidebar Component
Create `src/app_ui/components/sidebar.py`:

```python
import dash_bootstrap_components as dbc
from dash import html

sidebar_header = dbc.Row(
    [
        dbc.Col(
            html.H2(
                [
                    html.Span("Wind Turbine ML", style={"display": "block"}),
                    html.Span("Anomaly Platform", style={"display": "block"}),
                ],
                className="display-4",
            )
        ),
        dbc.Col(
            html.Button(
                html.Span(className="navbar-toggler-icon"),
                className="navbar-toggler",
                id="sidebar-toggle",
            ),
            width="auto",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.I(className="fas fa-chart-line mr-2"),
                        html.Span("Model Prediction", className="nav_link_span"),
                    ],
                    href="/",
                    active="exact",
                    className="sidebar_nav_link",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-database mr-2"),
                        html.Span("Model Tracking & Registry", className="nav_link_span"),
                    ],
                    href="/model-tracking",
                    active="exact",
                    className="sidebar_nav_link",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-project-diagram mr-2"),
                        html.Span("ML Pipelines Structure", className="nav_link_span"),
                    ],
                    href="/ml-pipelines",
                    active="exact",
                    className="sidebar_nav_link",
                ),
                dbc.NavLink(
                    [
                        html.I(className="fas fa-sitemap mr-2"),
                        html.Span("Solution Architecture", className="nav_link_span"),
                    ],
                    href="/architecture",
                    active="exact",
                    className="sidebar_nav_link",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
)
```

Explain:
- Font Awesome icons: `fas fa-*` classes for icons
- `dbc.NavLink`: Creates clickable navigation links
- `active="exact"`: Highlights the current page
- `vertical=True, pills=True`: Styling options

### Step 2: Update `app.py` to Include Sidebar
Update `src/app_ui/app.py`:

```python
from components.sidebar import sidebar  # type: ignore

# App layout with sidebar
content = html.Div(id="page-content", children=[dash.page_container])

app.layout = html.Div(
    id="main-layout",
    children=[
        dcc.Location(id="url"),
        sidebar,
        content,
    ],
)
```

### Step 3: Add Sidebar Toggle Callback
Add to `app.py`:

```python
from dash import Input, Output, State, dcc

@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    """Toggle sidebar collapsed state."""
    if n and classname == "":
        return "collapsed"
    return ""
```

Explain:
- Callbacks use `@app.callback` decorator
- `Input`: Component that triggers the callback
- `Output`: Component that gets updated
- `State`: Component value read but doesn't trigger callback

### Step 4: Create CSS Files
Create `src/app_ui/assets/responsive-sidebar.css` for sidebar styling and `src/app_ui/assets/custom.css` for general styling.

Explain:
- Files in `assets/` folder are automatically loaded by Dash
- CSS variables can be used for theming
- Responsive design for mobile/desktop

## Lesson 3: Creating Utility Functions

### Step 1: Create `utils.py` Structure
Create `src/app_ui/utils.py` with helper functions:

```python
import sys
from pathlib import Path
from app_data_manager.utils import read_config
from app_data_manager.data_manager import DataManager

# Configuration setup
project_root = Path(__file__).resolve().parents[2]
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)
```

Explain:
- Centralized utility functions for reuse across pages
- Global config reading at module level
- Path setup for imports

### Step 2: Add Data Loading Function
Add `load_prod_data()` function:

```python
def load_prod_data(
    n_data_points: int = 1000000,
    start_timestamp: str | pd.Timestamp | None = None,
    end_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load production data from database."""
    data_manager = DataManager(config)
    
    if start_timestamp is not None and end_timestamp is not None:
        raw_data = data_manager.get_data_by_timestamp_range(
            start_timestamp, end_timestamp, table_name="raw_data"
        )
        predictions = data_manager.get_data_by_timestamp_range(
            start_timestamp, end_timestamp, table_name="predictions"
        )
    else:
        raw_data = data_manager.get_last_n_points(n_data_points, table_name="raw_data")
        predictions = data_manager.get_last_n_points(n_data_points, table_name="predictions")
    
    # Merge and return
    # ...
```

Explain:
- Two query modes: by date range or by number of points
- Uses DataManager from app_data_manager module
- Returns merged DataFrame with predictions and true values

### Step 3: Create Common MLflow Utilities

**First, create the common MLflow utilities:**

1. Create `src/common/mlflow_utils.py`:
   ```python
   from mlflow.tracking import MlflowClient
   import mlflow
   import os
   from pathlib import Path
   import yaml
   
   def get_model_info_by_alias(
       alias: str, mlflow_tracking_uri=None, model_name=None
   ) -> dict | None:
       """Get model information by alias (champion/challenger)."""
       # Implementation details...
   ```

2. **In `app_ui/utils.py`:** Import from common module:
   ```python
   from common.mlflow_utils import get_model_info_by_alias
   ```

Explain:
- Common utilities are shared across pipelines, UI, and entrypoints
- Generic function that works for any alias
- Connects to MLflow tracking server
- Extracts model metadata and metrics
- Single source of truth for MLflow operations

## Lesson 4: Building the Home Page

### Step 1: Create Basic Layout
Update `src/app_ui/pages/home.py`:

```python
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

dash.register_page(__name__, path="/")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Control Panel"),
                        html.Label("Lookback Days"),
                        dcc.Input(
                            id="lookback-days",
                            type="number",
                            value=7,
                            min=1,
                            max=365,
                        ),
                        html.Label("Anomaly Error Metric"),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("MAE", id="metric-mae", color="primary"),
                                dbc.Button("MAPE", id="metric-mape", color="secondary"),
                            ],
                            id="metric-selector",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="error-plot"),
                        dcc.Graph(id="time-series-plot"),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)
```

### Step 2: Add Data Loading Callback
Add callback to load and prepare data:

```python
from app_ui.utils import load_and_prepare_error_data, get_date_range_from_lookback

@callback(
    [Output("error-plot", "figure"), Output("time-series-plot", "figure")],
    [
        Input("lookback-days", "value"),
        Input("metric-selector", "value"),
    ],
)
def update_plots(lookback_days, metric):
    """Update plots based on user inputs."""
    # Get date range from lookback days
    start_timestamp, end_timestamp = get_date_range_from_lookback(lookback_days)
    
    # Load and prepare data
    df = load_and_prepare_error_data(start_timestamp, end_timestamp)
    
    # Create plots
    error_fig = create_error_plot(df, metric, config)
    timeseries_fig = create_timeseries_plot(df)
    
    return error_fig, timeseries_fig
```

### Step 3: Add Plot Synchronization
Add x-axis synchronization between plots:

```python
from app_ui.utils import sync_xaxis

dcc.Store(id="xaxis-range-store", data=None),

@callback(
    [Output("error-plot", "figure"), Output("xaxis-range-store", "data")],
    [Input("time-series-plot", "relayoutData")],
    [State("error-plot", "figure"), State("xaxis-range-store", "data")],
    allow_duplicate=True,
)
def sync_timeseries_to_error(relayout_data, error_fig, stored_range):
    """Sync error plot x-axis when time-series plot is zoomed."""
    if relayout_data and "xaxis.range[0]" in relayout_data:
        error_fig = sync_xaxis(error_fig, relayout_data)
    return error_fig, relayout_data
```

Explain:
- `dcc.Store`: Client-side storage for sharing data between callbacks
- `allow_duplicate=True`: Allows multiple callbacks to update same output
- `sync_xaxis()`: Helper function to update x-axis range

### Step 4: Add Plot Creation Functions
Add to `utils.py`:

```python
import plotly.graph_objects as go

def create_error_plot(df, metric, config):
    """Create error metrics plot with rolling average and threshold."""
    fig = go.Figure()
    
    # Add raw error line
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df[f"{metric}_raw"],
        mode="lines",
        name=f"Raw {metric}",
        line=dict(color="#60a5fa", width=1),
    ))
    
    # Add rolling average
    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df[f"{metric}_rolling"],
        mode="lines",
        name=f"Rolling {metric}",
        line=dict(color="#2563eb", width=2),
    ))
    
    # Add threshold line
    threshold = config["anomaly_thresholds"][f"{metric}_threshold"]
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold}",
    )
    
    return fig
```

## Lesson 5: Model Tracking Page

### Step 1: Create Basic Layout
Create `src/app_ui/pages/model_tracking.py`:

```python
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html
from common.mlflow_utils import get_model_info_by_alias
from app_data_manager.utils import read_config

dash.register_page(__name__, path="/model-tracking")

# Read config
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

layout = dbc.Container(
    [
        # MLflow Integration Card
        html.Div([...]),
        
        # Champion Model Information Card
        html.Div(
            [
                html.H5("Champion Model Information"),
                html.Div(id="champion-model-info"),
            ]
        ),
        
        # Challenger Model Information Card
        html.Div(
            [
                html.H5("Challenger Model Information"),
                html.Div(id="challenger-model-info"),
            ]
        ),
    ],
    fluid=True,
)
```

### Step 2: Add Auto-Refresh Components
Add interval and store components:

```python
dcc.Interval(
    id="champion-info-interval",
    interval=60000,  # 60 seconds
    n_intervals=0,
),
dcc.Store(id="champion-info-trigger", data=0),
```

### Step 3: Add Dynamic Loading Callback
Add callback to load champion model info:

```python
@callback(
    Output("champion-model-info", "children"),
    [
        Input("champion-info-interval", "n_intervals"),
        Input("champion-info-trigger", "data"),
    ],
)
def update_champion_info(n_intervals, trigger):
    """Load and update champion model information."""
    current_config = read_config(parameters_path)
    model_name = current_config["mlflow"]["registered_model_name"]
    
    champion_info = get_model_info_by_alias("champion", MLFLOW_TRACKING_URI, model_name)
    
    return _create_champion_info_content(champion_info)
```

Explain:
- `dcc.Interval`: Triggers callback periodically
- Dynamic config reading: Allows config changes without restart
- Error handling: Returns user-friendly message if model not found

## Lesson 6: ML Pipelines Page

### Step 1: Create Basic Layout
Create `src/app_ui/pages/ml_pipelines.py`:

```python
import dash
import dash_bootstrap_components as dbc
from dash import html
import os

dash.register_page(__name__, path="/ml-pipelines")

KEDRO_VIZ_URI = os.getenv("KEDRO_VIZ_URI", "http://localhost:4141")

layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5("Pipeline Details"),
                                # Pipeline explanations...
                            ]
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H5("ML Pipeline Visualization"),
                        html.Iframe(
                            src=KEDRO_VIZ_URI,
                            style={"width": "100%", "height": "calc(100vh - 80px)"},
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)
```

Explain:
- Embedded Kedro-Viz iframe for pipeline visualization
- Environment variable for Kedro-Viz URI
- Two-column layout: info panel + visualization

## Lesson 7: Architecture Page

### Step 1: Create Basic Layout
Create `src/app_ui/pages/architecture.py`:

```python
import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/architecture")

layout = dbc.Container(
    [
        html.H4("Solution Architecture"),
        html.Img(
            src="/assets/app_architecture.gif",
            style={"width": "100%", "height": "auto"},
        ),
    ],
    fluid=True,
)
```

Explain:
- Static image display
- Assets folder: Files in `assets/` are served at `/assets/` URL

## Lesson 8: Styling and Theming

### Step 1: Create CSS Variables
In `src/app_ui/assets/custom.css`:

```css
:root {
    --primary-blue: #2563eb;
    --darker-blue: #1d4ed8;
    --light-blue: #93c5fd;
    --true-values-green: #22c55e;
    --threshold-red: #f97373;
    --grid-lines: #cbd5e1;
}

body {
    background-color: var(--background_active);
    color: var(--body_text);
}
```

### Step 2: Style Sidebar
In `src/app_ui/assets/responsive-sidebar.css`:

```css
#sidebar {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 16rem;
    background-color: var(--sidebar_navy);
    transition: margin 0.3s ease-in-out;
}

#page-content {
    margin-left: calc(16rem + 5px);
    padding: 0.5rem 5px;
}
```

Explain:
- CSS variables for consistent theming
- Fixed sidebar with responsive content area
- Transitions for smooth animations

## Lesson 9: Error Handling and Best Practices

### Step 1: Add Error Handling to Data Loading
Update `load_prod_data()`:

```python
def load_prod_data(...):
    try:
        data_manager = DataManager(config)
        # ... loading logic
    except ValueError as e:
        raise ValueError(f"Data loading failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
```

### Step 2: Add Loading States
Add to callbacks:

```python
@callback(
    Output("error-plot", "figure"),
    Input("lookback-days", "value"),
    prevent_initial_call=False,
)
def update_plots(lookback_days):
    """Update plots with loading state."""
    try:
        # Load data
        df = load_and_prepare_error_data(...)
        return create_error_plot(df)
    except Exception as e:
        # Return error message in figure
        return go.Figure().add_annotation(
            text=f"Error loading data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
        )
```

### Step 3: Add Educational Comments
Add comprehensive comments explaining:
- File purpose and structure
- Configuration setup
- Callback logic
- Data flow

Explain:
- Comments help students understand the code
- Document design decisions
- Explain complex logic

## Lesson 10: Testing and Deployment

### Step 1: Test Locally
Run the app:
```bash
python -m src.app_ui.app
```

Verify:
- All pages load correctly
- Navigation works
- Callbacks function properly
- Data loads from database
- MLflow integration works

### Step 2: Production Deployment
Create `entrypoint/ml_app/app.py`:

```python
from src.app_ui.app import server

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8050)
```

Run with Gunicorn:
```bash
gunicorn entrypoint.ml_app.app:server --bind 0.0.0.0:8050 --workers 4
```

Explain:
- `server` object from Dash app can be used with WSGI servers
- Gunicorn for production deployment
- Multiple workers for handling concurrent requests

## Summary

Key concepts covered:
1. **Multi-page Dash apps**: Using `use_pages=True` and `dash.register_page()`
2. **Components**: Reusable UI elements (sidebar)
3. **Callbacks**: Interactive functionality with `@app.callback`
4. **Data integration**: Connecting to database and MLflow
5. **Styling**: CSS theming and responsive design
6. **Error handling**: Graceful error messages
7. **Best practices**: Code organization and documentation

