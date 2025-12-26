"""Sidebar component with icons and collapsible functionality."""

from dash import html
import dash_bootstrap_components as dbc

# Sidebar header with title and toggle button
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
            [
                html.Button(
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    id="navbar-toggle",
                ),
                html.Button(
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    style={
                        "width": "40px",
                        "height": "40px",
                        "padding": "8px",
                        "font-size": "1.2em",
                    },
                    id="sidebar-toggle",
                ),
            ],
            width="auto",
            align="center",
        ),
    ]
)

# Sidebar with navigation links and icons
sidebar = html.Div(
    [
        sidebar_header,
        html.Div(
            [
                html.Hr(),
            ],
            id="blurb",
        ),
        # Collapsible navigation menu
        dbc.Collapse(
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
                            html.Span(
                                "Model Tracking & Registry", className="nav_link_span"
                            ),
                        ],
                        href="/model-tracking",
                        active="exact",
                        className="sidebar_nav_link",
                    ),
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-project-diagram mr-2"),
                            html.Span(
                                "ML Pipelines Structure", className="nav_link_span"
                            ),
                        ],
                        href="/ml-pipelines",
                        active="exact",
                        className="sidebar_nav_link",
                    ),
                    dbc.NavLink(
                        [
                            html.I(className="fas fa-sitemap mr-2"),
                            html.Span(
                                "Solution Architecture", className="nav_link_span"
                            ),
                        ],
                        href="/architecture",
                        active="exact",
                        className="sidebar_nav_link",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
            id="collapse",
        ),
    ],
    id="sidebar",
)
