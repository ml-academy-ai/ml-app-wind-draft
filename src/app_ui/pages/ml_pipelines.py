import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Register this page with Dash - accessible at "/ml-pipelines"
dash.register_page(__name__, path="/ml-pipelines")

# Kedro-Viz configuration
# Get Kedro-Viz URI from environment variable or use default
# Kedro-Viz is a separate service that visualizes Kedro pipelines
# Default: http://localhost:4141 (local Kedro-Viz server)
# Can be configured via KEDRO_VIZ_URI environment variable
KEDRO_VIZ_URI = os.getenv("KEDRO_VIZ_URI", "http://localhost:4141")


# Page layout - 2-column layout: Info panel (left) + Visualization (right)
layout = dbc.Container(
    [
        dbc.Row(
            [
                # Left column: Pipeline Information Panel (3/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Pipeline Details",
                                    style={"color": "#222", "marginBottom": "12px"},
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Feature Engineering Pipeline",
                                            style={
                                                "color": "#222",
                                                "marginBottom": "8px",
                                                "marginTop": "16px",
                                                "fontSize": "15px",
                                            },
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Standardizes column names and removes irrelevant columns"
                                                ),
                                                html.Li(
                                                    "Detects and handles outliers using difference-based methods"
                                                ),
                                                html.Li(
                                                    "Applies signal smoothing to reduce noise"
                                                ),
                                                html.Li(
                                                    "Creates lag features to capture temporal dependencies"
                                                ),
                                                html.Li(
                                                    "Generates rolling statistical features (mean, std, min, max)"
                                                ),
                                                html.Li(
                                                    "Separates features from target variable (power output)"
                                                ),
                                                html.Li(
                                                    "Ensures consistent transformations across training and production"
                                                ),
                                            ],
                                            style={
                                                "color": "#444",
                                                "fontSize": "13px",
                                                "lineHeight": "1.6",
                                                "marginBottom": "12px",
                                            },
                                        ),
                                        html.H6(
                                            "Training Pipeline",
                                            style={
                                                "color": "#222",
                                                "marginBottom": "8px",
                                                "marginTop": "16px",
                                                "fontSize": "15px",
                                            },
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Splits data into training and test sets"
                                                ),
                                                html.Li(
                                                    "Performs hyperparameter tuning using cross-validation"
                                                ),
                                                html.Li(
                                                    "Trains the best model on full training set"
                                                ),
                                                html.Li(
                                                    "Evaluates model performance on test set"
                                                ),
                                                html.Li(
                                                    "Logs results, metrics, and model to MLflow"
                                                ),
                                                html.Li(
                                                    "Registers model in MLflow model registry"
                                                ),
                                                html.Li(
                                                    "Validates model as challenger candidate"
                                                ),
                                            ],
                                            style={
                                                "color": "#444",
                                                "fontSize": "13px",
                                                "lineHeight": "1.6",
                                                "marginBottom": "12px",
                                            },
                                        ),
                                        html.H6(
                                            "Inference Pipeline",
                                            style={
                                                "color": "#222",
                                                "marginBottom": "8px",
                                                "marginTop": "16px",
                                                "fontSize": "15px",
                                            },
                                        ),
                                        html.Ul(
                                            [
                                                html.Li(
                                                    "Loads champion model from MLflow registry"
                                                ),
                                                html.Li(
                                                    "Applies feature engineering transformations"
                                                ),
                                                html.Li(
                                                    "Generates predictions on production data"
                                                ),
                                                html.Li(
                                                    "Computes prediction metrics (MAE, MAPE)"
                                                ),
                                                html.Li(
                                                    "Saves predictions to database for monitoring"
                                                ),
                                            ],
                                            style={
                                                "color": "#444",
                                                "fontSize": "13px",
                                                "lineHeight": "1.6",
                                                "marginBottom": "12px",
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "15px",
                                "border": "1px solid #e0e0e0",
                            },
                        ),
                    ],
                    width=3,
                ),
                # Right column: Kedro-Viz Visualization (9/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                # Header section with title only
                                html.H5(
                                    "ML Pipeline Visualization",
                                    style={
                                        "marginBottom": "12px",
                                        "color": "#222",
                                    },
                                ),
                                # Kedro-Viz iframe - embeds visualization in an iframe
                                # This allows seamless integration without leaving the Dash app
                                # Height is calculated to fill viewport minus header/navbar
                                html.Iframe(
                                    src=KEDRO_VIZ_URI,
                                    style={
                                        "width": "100%",
                                        "height": "calc(100vh - 80px)",  # Viewport height minus minimal UI elements
                                        "border": "1px solid #e0e0e0",
                                        "borderRadius": "8px",
                                        "display": "block",
                                        "flex": "1",  # Takes remaining space in flex container
                                    },
                                    allow="fullscreen",
                                ),
                            ],
                            style={
                                "backgroundColor": "#fff",
                                "borderRadius": "12px",
                                "padding": "12px",
                                "border": "1px solid #e0e0e0",
                                "height": "calc(100vh - 60px)",  # Full viewport height minus minimal navbar
                                "display": "flex",
                                "flexDirection": "column",  # Vertical layout for header + iframe
                            },
                        ),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,  # Full-width container
    style={"paddingTop": "0px", "height": "100vh"},
)
