from pathlib import Path
import sys

import dash
from dash import html
import dash_bootstrap_components as dbc

# Configuration setup
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Register this page with Dash - accessible at "/architecture"
dash.register_page(__name__, path="/architecture")

# Asset path - architecture diagram is stored in the assets folder
# Dash automatically serves files from /assets/ at the root URL path
ARCHITECTURE_GIF_PATH = "/assets/app_architecture.gif"

# Page layout - 2-column layout: Info panel (left) + Architecture diagram (right)
layout = dbc.Container(
    [
        # Page Title
        html.H1("Solution Architecture", className="mb-4", style={"color": "#222"}),
        dbc.Row(
            [
                # Left column: Information Panel (3/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "System Overview",
                                    style={"color": "#222", "marginBottom": "16px"},
                                ),
                                html.P(
                                    "The application architecture diagram below illustrates the "
                                    "end-to-end ML pipeline, including data ingestion, preprocessing, "
                                    "feature engineering, model training, inference, and monitoring components.",
                                    style={
                                        "color": "#444",
                                        "fontSize": "14px",
                                        "lineHeight": "1.6",
                                    },
                                ),
                                html.Hr(),
                                html.H5(
                                    "Components",
                                    style={"color": "#222", "marginTop": "20px"},
                                ),
                                # List of key system components
                                html.Ul(
                                    [
                                        html.Li(
                                            "Data Pipeline: Raw data processing and feature engineering"
                                        ),
                                        html.Li(
                                            "Model Training: Hyperparameter tuning and model selection"
                                        ),
                                        html.Li(
                                            "Model Registry: MLflow-based model versioning"
                                        ),
                                        html.Li(
                                            "Inference Service: Real-time prediction generation"
                                        ),
                                        html.Li(
                                            "Monitoring: Performance tracking and anomaly detection"
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
                                "marginBottom": "20px",
                            },
                        ),
                    ],
                    width=3,
                ),
                # Right column: Architecture Diagram (9/12 width)
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Architecture Diagram",
                                    style={"marginBottom": "20px", "color": "#222"},
                                ),
                                # Image container - displays the architecture GIF
                                # The GIF shows the complete system architecture
                                html.Div(
                                    id="architecture-image-container",
                                    children=[
                                        html.Img(
                                            src=ARCHITECTURE_GIF_PATH,
                                            style={
                                                "width": "100%",
                                                "height": "auto",  # Maintains aspect ratio
                                                "borderRadius": "12px",
                                                "border": "1px solid #e0e0e0",
                                            },
                                            alt="Application Architecture Diagram",
                                        ),
                                    ],
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
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,  # Full-width container
)
