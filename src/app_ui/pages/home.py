"""
Model Prediction Dashboard - Home Page

This page displays:
1. Time series plot: Shows ML model predictions vs actual values
2. Error metrics plot: Shows MAE/MAPE with rolling averages and anomaly thresholds
3. Interactive controls: Lookback days and error metric selection
4. Synchronized zooming: Both plots share x-axis for easy comparison

Key Features:
- Real-time data loading from production database
- Configurable rolling window for error smoothing
- Anomaly detection based on threshold violations
- Bidirectional plot synchronization (zoom one, both update)
"""

import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, callback_context, dcc, html

from src.app_data_manager.utils import read_config
from src.app_ui.utils import (
    create_error_figure,
    create_error_plot,
    create_timeseries_plot,
    get_date_range_from_lookback,
    get_metric_config,
    load_and_prepare_error_data,
    sync_xaxis,
)

# Configuration setup
# Set up project paths and read configuration file
# Config contains: anomaly thresholds, rolling window size, MLflow settings
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Read configuration once at module level (not in callbacks) for efficiency
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Register this page with Dash - makes it accessible at the root URL "/"
dash.register_page(__name__, path="/")

# Page layout - define the visual structure of the page using Dash Bootstrap Components
# Layout is a 2-column design: Control Panel (left) + Plots (right)
layout = dbc.Container(
    [
        dbc.Row(
            [
                # ========================================================================
                # LEFT COLUMN: Control Panel (3/12 width)
                # ========================================================================
                # Contains user inputs and dashboard overview
                dbc.Col(
                    [
                        # Control Panel Card
                        html.Div(
                            [
                                html.H4(
                                    "Control Panel",
                                    style={"color": "#222", "marginBottom": "16px"},
                                ),
                                # Lookback Days Input
                                # Controls how many days of historical data to display
                                html.Label(
                                    "Lookback Days",
                                    style={"color": "#222", "marginBottom": "8px"},
                                ),
                                dcc.Input(
                                    id="lookback-days",
                                    type="number",
                                    min=1,
                                    step=1,
                                    value=7,  # Default 7 days
                                    style={
                                        "marginBottom": "16px",
                                        "width": "100%",
                                        "padding": "8px",
                                    },
                                ),
                                # Error Metric Dropdown
                                # Allows switching between MAE and MAPE for anomaly detection
                                html.Label(
                                    "Anomaly Error Metric",
                                    style={
                                        "color": "#222",
                                        "marginBottom": "8px",
                                        "marginTop": "16px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="error-metric",
                                    options=[
                                        {
                                            "label": "Mean Absolute Error",
                                            "value": "mae",
                                        },
                                        {
                                            "label": "Mean Absolute Percentage Error",
                                            "value": "mape",
                                        },
                                    ],
                                    value="mape",  # Default to MAPE
                                    style={
                                        "marginBottom": "16px",
                                        "width": "100%",
                                        "fontSize": "12px",
                                    },
                                    clearable=False,
                                ),
                            ],
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #e0e0e0",
                                "marginBottom": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.H5(
                                    "Dashboard Overview",
                                    style={"color": "#222", "marginBottom": "12px"},
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            "Time series plot shows predictions vs true values on test set."
                                        ),
                                        html.Li(
                                            "Error plots display MAE and MAPE metrics with rolling averages."
                                        ),
                                        html.Li(
                                            "Adjust lookback window to focus on specific time periods."
                                        ),
                                        html.Li(
                                            "Adjust rolling window to smooth error metrics."
                                        ),
                                    ],
                                    style={
                                        "color": "#444",
                                        "fontSize": "14px",
                                        "lineHeight": "1.6",
                                    },
                                ),
                            ],
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "20px",
                                "border": "1px solid #e0e0e0",
                            },
                        ),
                    ],
                    width=3,
                ),
                # ========================================================================
                # RIGHT COLUMN: Visualization Area (9/12 width)
                # ========================================================================
                # Contains the two main plots: Error metrics and Time series
                dbc.Col(
                    [
                        # Store component to share x-axis range between callbacks
                        # Used for synchronizing zoom/pan between the two plots
                        dcc.Store(id="xaxis-range-store", data=None),
                        # ================================================================
                        # PLOT 1: Error Metrics Plot
                        # ================================================================
                        # Shows MAE or MAPE with:
                        # - Raw error values (light line)
                        # - Rolling average (bold line)
                        # - Anomaly threshold (red dashed line)
                        html.H5(
                            "Anomaly Indicator – ML Model Error",
                            style={"marginBottom": "10px", "color": "#222"},
                        ),
                        dcc.Graph(
                            id="error-plot",
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "8px",
                                "height": "calc(50vh - 100px)",
                                "minHeight": "400px",
                                "marginBottom": "20px",
                            },
                        ),
                        # ================================================================
                        # PLOT 2: Time Series Plot
                        # ================================================================
                        # Shows:
                        # - True values (actual wind power)
                        # - Predictions (model output)
                        # - Error bars or shaded regions
                        html.H5(
                            "Power: Predictions vs True Values",
                            style={"marginBottom": "10px", "color": "#222"},
                        ),
                        dcc.Graph(
                            id="time-series-plot",
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "8px",
                                "height": "calc(50vh - 100px)",
                                "minHeight": "400px",
                            },
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
    style={
        "paddingTop": "20px",
        "paddingLeft": "5px",
        "paddingRight": "5px",
        "height": "calc(100vh - 60px)",
    },
)


# Callback 1: Main Plot Update
# This callback is triggered when user changes:
# - Lookback days (how many days to display)
# - Error metric (MAE vs MAPE)
#
# Data Flow:
# 1. User changes input → callback triggers
# 2. Calculate date range from lookback days
# 3. Load data from database for that date range
# 4. Compute error metrics (MAE/MAPE) and rolling averages
# 5. Create two Plotly figures
# 6. Return figures to update both plots
@callback(
    [Output("time-series-plot", "figure"), Output("error-plot", "figure")],
    [
        Input("lookback-days", "value"),
        Input("error-metric", "value"),
    ],
)
def update_plots(lookback_days, error_metric):
    """
    Main callback to update both plots when user changes controls.

    Args:
        lookback_days: Number of days to look back (from user input)
        error_metric: Either "mae" or "mape" (from dropdown)

    Returns:
        Tuple of (time_series_figure, error_figure)
    """
    try:
        # Get rolling window size from config (in data points, not time)
        # Default: 288 points = 12 days if data is hourly (288 hours)
        rolling_window = config["anomaly_thresholds"].get(
            "rolling_window_data_points", 288
        )

        # Validate and set defaults
        error_metric = error_metric or "mape"
        lookback_days = lookback_days if lookback_days and lookback_days >= 1 else 7

        # Step 1: Calculate date range from lookback days
        # Returns start and end timestamps for database query
        start_dt, end_dt = get_date_range_from_lookback(lookback_days)

        # Step 2: Load data from database and prepare it
        # - Queries database for the date range
        # - Computes MAE and MAPE errors
        # - Adds datetime column for plotting
        # - Calculates rolling averages for smoothing
        df = load_and_prepare_error_data(start_dt, end_dt, rolling_window)

        # Step 3: Get configuration for selected metric
        # Returns: colors, labels, and data columns for MAE or MAPE
        metric_config = get_metric_config(error_metric, df)

        # Step 4: Get anomaly threshold from config
        # Used to draw threshold line on error plot
        metric_threshold = config["anomaly_thresholds"][error_metric]

        # Step 5: Create Plotly figures
        fig1 = create_timeseries_plot(df)  # Predictions vs True values
        fig2 = create_error_plot(
            df, metric_config, metric_threshold, rolling_window
        )  # Error metrics

        return fig1, fig2

    except Exception as e:
        # If anything fails, show error message in a figure
        # This ensures the UI always shows something, even on errors
        return create_error_figure(str(e))


# Callback 2: Plot Synchronization (Bidirectional)
# This callback keeps the x-axis synchronized between the two plots.
# When user zooms/pans one plot, the other automatically updates to match.
#
# How it works:
# 1. Monitors relayoutData from BOTH plots (triggers on zoom/pan)
# 2. Uses callback_context to determine which plot was interacted with
# 3. Updates ONLY the other plot (the one not zoomed)
# 4. Uses State (not Input) for figures to avoid infinite loops
#
# Key Concepts:
# - Input: relayoutData (triggers callback on user interaction)
# - State: figure (reads current state without triggering)
# - allow_duplicate=True: Allows multiple callbacks to update same output
# - prevent_initial_call=True: Don't run on page load
@callback(
    [
        Output("error-plot", "figure", allow_duplicate=True),
        Output("time-series-plot", "figure", allow_duplicate=True),
        Output("xaxis-range-store", "data", allow_duplicate=True),
    ],
    [
        Input(
            "error-plot", "relayoutData"
        ),  # Triggers when error plot is zoomed/panned
        Input(
            "time-series-plot", "relayoutData"
        ),  # Triggers when time-series plot is zoomed/panned
    ],
    [
        State("error-plot", "figure"),  # Current state of error plot (read-only)
        State(
            "time-series-plot", "figure"
        ),  # Current state of time-series plot (read-only)
    ],
    prevent_initial_call=True,  # Don't run when page first loads
)
def sync_plots_xaxis(
    error_relayout_data, timeseries_relayout_data, error_figure, timeseries_figure
):
    """
    Synchronize x-axis between plots when user zooms or pans.

    This enables bidirectional synchronization:
    - Zoom error plot → time-series plot updates
    - Zoom time-series plot → error plot updates

    Args:
        error_relayout_data: Zoom/pan data from error plot (None if not triggered)
        timeseries_relayout_data: Zoom/pan data from time-series plot (None if not triggered)
        error_figure: Current error plot figure state
        timeseries_figure: Current time-series plot figure state

    Returns:
        Tuple of (error_figure, timeseries_figure, x_range)
        Uses dash.no_update for plots that don't need updating
    """
    # Get which input triggered this callback
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update

    # Extract which plot was interacted with (e.g., "error-plot" or "time-series-plot")
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "error-plot":
        # User zoomed/panned the error plot
        # Update the time-series plot to match the new x-axis range
        updated_figure, x_range = sync_xaxis(error_relayout_data, timeseries_figure)
        return dash.no_update, updated_figure, x_range  # Only update time-series plot

    elif trigger_id == "time-series-plot":
        # User zoomed/panned the time-series plot
        # Update the error plot to match the new x-axis range
        updated_figure, x_range = sync_xaxis(timeseries_relayout_data, error_figure)
        return updated_figure, dash.no_update, x_range  # Only update error plot

    # Fallback (shouldn't happen, but good practice)
    return dash.no_update, dash.no_update, dash.no_update
