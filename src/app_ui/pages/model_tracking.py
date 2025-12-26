"""
Model Tracking & Registry Page - MLflow Integration

This page provides:
1. Champion Model Information: Displays current production model details
   - Model name, version, last updated timestamp
   - Test set metrics (MAE, MAPE)
   - Auto-refreshes every 60 seconds to show latest model
2. MLflow UI Access: Links to open MLflow Tracking UI in new tab
   - Direct access to experiments, runs, and model registry
   - Quick links to Experiments and Models pages

Key Features:
- Dynamic model information loading from MLflow registry
- Periodic refresh to ensure UI shows latest champion model
- Integration with MLflow tracking server
- Displays test set performance metrics
"""

import os
from pathlib import Path
import sys

import dash
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc

from src.app_data_manager.utils import read_config  # type: ignore
from src.app_ui.utils import get_model_info_by_alias  # type: ignore

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Read configuration file (contains MLflow model name, etc.)
parameters_path = project_root / "conf" / "base" / "parameters.yml"
config = read_config(parameters_path)

# Register this page with Dash - accessible at "/model-tracking"
dash.register_page(__name__, path="/model-tracking")

# MLflow configuration
# Get MLflow tracking URI from environment variable or use default
# Environment variable allows different URIs for dev/staging/prod
# Default: http://127.0.0.1:8080 (local MLflow server)
# Using 127.0.0.1 instead of localhost helps with some browser security policies
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")

# Page layout - Single column layout, stacked vertically, full screen
layout = dbc.Container(
    [
        # Hidden components for auto-refresh
        # dcc.Interval: Triggers callback every 60 seconds
        # This ensures champion model info stays up-to-date even if
        # a new model is promoted to production while the page is open
        dcc.Interval(
            id="champion-info-interval",
            interval=60000,  # 60,000 ms = 60 seconds
            n_intervals=0,  # Counter starts at 0
        ),
        # dcc.Store: Can be used to manually trigger refresh if needed
        # Currently used as additional trigger for the callback
        dcc.Store(id="champion-info-trigger", data=0),
        # Interval and Store for challenger info refresh
        dcc.Interval(
            id="challenger-info-interval",
            interval=60000,  # 60,000 ms = 60 seconds
            n_intervals=0,  # Counter starts at 0
        ),
        dcc.Store(id="challenger-info-trigger", data=0),
        # MLflow Integration & UI Access Card - combined
        html.Div(
            [
                html.H4(
                    "MLflow Integration",
                    style={"color": "#222", "marginBottom": "16px"},
                ),
                html.P(
                    "This page provides access to the MLflow Tracking UI, where you can "
                    "view experiment runs and metrics, compare model performance, track hyperparameters, "
                    "manage model versions, and register and promote models. "
                    "Click the buttons below to access different sections of the MLflow Tracking UI.",
                    style={
                        "color": "#444",
                        "fontSize": "14px",
                        "lineHeight": "1.6",
                        "marginBottom": "20px",
                    },
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "Open MLflow UI",
                                href=MLFLOW_TRACKING_URI,
                                target="_blank",
                                color="primary",
                                size="lg",
                                style={"width": "100%"},
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Experiments",
                                href=f"{MLFLOW_TRACKING_URI}/#/experiments",
                                target="_blank",
                                color="primary",
                                size="lg",
                                style={"width": "100%"},
                            ),
                            width=4,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Models",
                                href=f"{MLFLOW_TRACKING_URI}/#/models",
                                target="_blank",
                                color="primary",
                                size="lg",
                                style={"width": "100%"},
                            ),
                            width=4,
                        ),
                    ],
                    className="g-3",
                ),
            ],
            style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "30px",
                "border": "1px solid #e0e0e0",
                "marginBottom": "20px",
            },
        ),
        # Champion Model Information Card - displays current production model details
        # Content is dynamically loaded via callback (see below)
        html.Div(
            [
                html.H5(
                    "Champion Model Information",
                    style={"marginBottom": "20px", "color": "#222"},
                ),
                # This div's children are updated by the callback
                # Initial state shows "Loading..." message
                html.Div(
                    id="champion-model-info",
                    children=[
                        html.Div(
                            "Loading champion model information...",
                            style={"color": "#666", "fontSize": "14px"},
                        ),
                    ],
                ),
            ],
            style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "30px",
                "border": "1px solid #e0e0e0",
                "marginBottom": "20px",
            },
        ),
        # Challenger Model Information Card - displays latest challenger model details
        # Content is dynamically loaded via callback (see below)
        html.Div(
            [
                html.H5(
                    "Challenger Model Information",
                    style={"marginBottom": "20px", "color": "#222"},
                ),
                # This div's children are updated by the callback
                # Initial state shows "Loading..." message
                html.Div(
                    id="challenger-model-info",
                    children=[
                        html.Div(
                            "Loading challenger model information...",
                            style={"color": "#666", "fontSize": "14px"},
                        ),
                    ],
                ),
            ],
            style={
                "backgroundColor": "#fff",
                "borderRadius": "12px",
                "padding": "30px",
                "border": "1px solid #e0e0e0",
                "marginBottom": "20px",
            },
        ),
    ],
    fluid=True,
    style={"paddingTop": "0px", "height": "100vh", "overflowY": "auto"},
)


# Helper function: Create Challenger Model Info Display
def _create_challenger_info_content(challenger_info):
    """
    Helper function to create HTML layout for challenger model information.

    This function takes the challenger model data dictionary and formats it
    into a nice horizontal layout using Dash Bootstrap Components.

    Args:
        challenger_info: Dictionary with keys:
            - model_name: Name of the model
            - version: Model version number
            - last_updated: Datetime when model was last updated
            - test_mae: Mean Absolute Error on test set
            - test_mape: Mean Absolute Percentage Error on test set
            Returns None if no challenger model found

    Returns:
        HTML Div containing formatted model information, or error message
    """
    if not challenger_info:
        return html.P(
            "No challenger model found in registry. Make sure MLflow is running and a model is registered with 'challenger' alias.",
            style={"color": "#666", "fontSize": "14px"},
        )

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H6(
                        "Model Name",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        challenger_info.get("model_name", "N/A"),
                        style={
                            "color": "#222",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6(
                        "Version",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        f"v{challenger_info['version']}"
                        if challenger_info.get("version")
                        else "N/A",
                        style={
                            "color": "#222",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.H6(
                        "Last Updated",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        challenger_info["last_updated"].strftime("%Y-%m-%d %H:%M:%S")
                        if challenger_info.get("last_updated")
                        else "N/A",
                        style={
                            "color": "#222",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6(
                        "Test Set Error Metrics",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        [
                            html.P(
                                f"MAE: {challenger_info['test_mae']:.4f}"
                                if challenger_info.get("test_mae") is not None
                                else "MAE: N/A",
                                style={
                                    "color": "#222",
                                    "fontSize": "14px",
                                    "marginBottom": "4px",
                                },
                            ),
                            html.P(
                                f"MAPE: {challenger_info['test_mape']:.4f}%"
                                if challenger_info.get("test_mape") is not None
                                else "MAPE: N/A",
                                style={
                                    "color": "#222",
                                    "fontSize": "14px",
                                    "marginBottom": "0px",
                                },
                            ),
                        ],
                        style={"marginBottom": "0px"},
                    ),
                ],
                width=4,
            ),
        ],
        className="g-4",
    )


# Helper function: Create Champion Model Info Display
def _create_champion_info_content(champion_info):
    """
    Helper function to create HTML layout for champion model information.

    This function takes the champion model data dictionary and formats it
    into a nice horizontal layout using Dash Bootstrap Components.

    Args:
        champion_info: Dictionary with keys:
            - model_name: Name of the model
            - version: Model version number
            - last_updated: Datetime when model was last updated
            - test_mae: Mean Absolute Error on test set
            - test_mape: Mean Absolute Percentage Error on test set
            Returns None if no champion model found

    Returns:
        HTML Div containing formatted model information, or error message
    """
    if not champion_info:
        return html.P(
            "No champion model found in registry. Make sure MLflow is running and a model is registered in Production stage.",
            style={"color": "#666", "fontSize": "14px"},
        )

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.H6(
                        "Model Name",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        champion_info.get("model_name", "N/A"),
                        style={
                            "color": "#222",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6(
                        "Version",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        f"v{champion_info['version']}"
                        if champion_info.get("version")
                        else "N/A",
                        style={
                            "color": "#222",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    html.H6(
                        "Last Updated",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.P(
                        champion_info["last_updated"].strftime("%Y-%m-%d %H:%M:%S")
                        if champion_info.get("last_updated")
                        else "N/A",
                        style={
                            "color": "#222",
                            "fontSize": "16px",
                            "fontWeight": "bold",
                            "marginBottom": "0px",
                        },
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.H6(
                        "Test Set Error Metrics",
                        style={
                            "color": "#666",
                            "fontSize": "12px",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        [
                            html.P(
                                f"MAE: {champion_info['test_mae']:.4f}"
                                if champion_info.get("test_mae") is not None
                                else "MAE: N/A",
                                style={
                                    "color": "#222",
                                    "fontSize": "14px",
                                    "marginBottom": "4px",
                                },
                            ),
                            html.P(
                                f"MAPE: {champion_info['test_mape']:.4f}%"
                                if champion_info.get("test_mape") is not None
                                else "MAPE: N/A",
                                style={
                                    "color": "#222",
                                    "fontSize": "14px",
                                    "marginBottom": "0px",
                                },
                            ),
                        ],
                        style={"marginBottom": "0px"},
                    ),
                ],
                width=4,
            ),
        ],
        className="g-4",
    )


# Callback: Dynamic Champion Model Information Loading
# This callback automatically refreshes champion model information
#
# Why dynamic loading?
# - Model might be updated/promoted while page is open
# - Ensures UI always shows current production model
# - No need to manually refresh the page
#
# How it works:
# 1. Triggered by dcc.Interval every 60 seconds (automatic refresh)
# 2. Also triggered by dcc.Store changes (manual refresh if needed)
# 3. Reads model name from config (in case it changes)
# 4. Queries MLflow registry for champion model
# 5. Updates the UI with latest information
@callback(
    Output("champion-model-info", "children"),
    [
        Input("champion-info-interval", "n_intervals"),  # Triggers every 60s
        Input("champion-info-trigger", "data"),  # Manual trigger
    ],
)
def update_champion_info(n_intervals, trigger):
    """
    Load and update champion model information dynamically.

    This callback:
    1. Reads the model name from config (allows config changes without restart)
    2. Queries MLflow registry for the model with "champion" alias
    3. Extracts model metadata and test set metrics
    4. Formats and displays the information

    Args:
        n_intervals: Counter from dcc.Interval (increments every 60s)
        trigger: Data from dcc.Store (can be used for manual refresh)

    Returns:
        HTML content displaying champion model information
    """
    # Read config fresh each time (in case config file was updated)
    # This allows changing model name without restarting the app
    current_config = read_config(parameters_path)
    model_name = current_config["mlflow"]["registered_model_name"]

    # Query MLflow registry for champion model information
    # Returns dict with model details or None if not found
    champion_info = get_model_info_by_alias(
        "champion", MLFLOW_TRACKING_URI, model_name=model_name
    )

    # Format and return HTML content
    return _create_champion_info_content(champion_info)


# Callback: Dynamic Challenger Model Information Loading
# This callback automatically refreshes challenger model information
#
# Why dynamic loading?
# - Model might be updated/promoted while page is open
# - Ensures UI always shows current challenger model
# - No need to manually refresh the page
#
# How it works:
# 1. Triggered by dcc.Interval every 60 seconds (automatic refresh)
# 2. Also triggered by dcc.Store changes (manual refresh if needed)
# 3. Reads model name from config (in case it changes)
# 4. Queries MLflow registry for challenger model by alias
# 5. Updates the UI with latest information
@callback(
    Output("challenger-model-info", "children"),
    [
        Input("challenger-info-interval", "n_intervals"),  # Triggers every 60s
        Input("challenger-info-trigger", "data"),  # Manual trigger
    ],
)
def update_challenger_info(n_intervals, trigger):
    """
    Load and update challenger model information dynamically.

    This callback:
    1. Reads the model name from config (allows config changes without restart)
    2. Queries MLflow registry for the model with "challenger" alias
    3. Extracts model metadata and test set metrics
    4. Formats and displays the information

    Args:
        n_intervals: Counter from dcc.Interval (increments every 60s)
        trigger: Data from dcc.Store (can be used for manual refresh)

    Returns:
        HTML content displaying challenger model information
    """
    # Read config fresh each time (in case config file was updated)
    # This allows changing model name without restarting the app
    current_config = read_config(parameters_path)
    model_name = current_config["mlflow"]["registered_model_name"]

    # Query MLflow registry for challenger model information
    # Returns dict with model details or None if not found
    challenger_info = get_model_info_by_alias(
        "challenger", MLFLOW_TRACKING_URI, model_name=model_name
    )

    # Format and return HTML content
    return _create_challenger_info_content(challenger_info)
