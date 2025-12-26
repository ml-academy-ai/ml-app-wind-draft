"""Multi-page Dash application for ML model monitoring and visualization."""

import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from components.sidebar import sidebar  # type: ignore
from dash import Input, Output, State, dcc, html

# Path setup for working directory and other runtime needs
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.chdir(project_root)


# Font Awesome for icons
FA = "https://use.fontawesome.com/releases/v5.15.1/css/all.css"

# Initialize the Dash app with Bootstrap theme and Font Awesome
# Note: CSS files in assets folder are automatically loaded
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, FA],
    suppress_callback_exceptions=True,
    use_pages=True,
    pages_folder="pages",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

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


# Callback to toggle sidebar collapse
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


# Callback to toggle mobile menu collapse
@app.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    """Toggle mobile menu collapse."""
    if n:
        return not is_open
    return is_open


server = app.server

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8050)
