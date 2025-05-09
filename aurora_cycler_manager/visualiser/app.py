"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Web-based visualiser for the Aurora cycler manager based on Dash and Plotly.

Allows users to rapidly view and compare data from the Aurora robot and cycler
systems, both of individual samples and of batches of samples.

Allows users to view current information in the database, and control cyclers
remotely, loading, ejecting, and submitting jobs to samples.
"""

from __future__ import annotations

import socket
import webbrowser

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import ClientsideFunction, Dash, Input, Output, _dash_renderer, dcc, html
from waitress import serve

from aurora_cycler_manager.visualiser.batches import batches_layout, register_batches_callbacks
from aurora_cycler_manager.visualiser.db_view import db_view_layout, register_db_view_callbacks
from aurora_cycler_manager.visualiser.notifications import notifications_layout, register_notifications_callbacks
from aurora_cycler_manager.visualiser.samples import register_samples_callbacks, samples_layout

# Need to set this for Mantine notifications to work
_dash_renderer._set_react_version("18.2.0")  # noqa: SLF001

# Spinner
custom_spinner = html.Div(
    style={
        "position": "absolute",
        "top": "50%",
        "left": "50%",
        "transform": "translate(-50%, -50%)",  # Shift spinner up
        "width": "100px",
        "height": "100px",
    },
    children=[
        html.Img(
            src="/assets/spinner-spin.svg",
            className="spinner-spin",
            style={"width": "100px", "height": "100px"},
        ),
        html.Img(
            src="/assets/spinner-stationary.svg",
            style={
                "position": "absolute",
                "top": "0",
                "left": "0",
                "width": "100px",
                "height": "100px",
                "color": "white",
            },
        ),
    ],
)

# Loading messages
loading_message = html.Div(
    "This is some kind of text",
    id="loading-message",
    style={
        "position": "absolute",
        "left": "50%",
        "top": "50%",
        "transform": "translate(-50%, 50px)",  # Shift text down
        "fontSize": "20px",
        "color": "#000000",
        "textAlign": "center",
        "textGlow": "0 0 20px blue",  # Add white glow
        "opacity": 1,
        "transition": "opacity 0.5s ease-in-out",
    },
)

# Define app and layout
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP, dmc.styles.NOTIFICATIONS, "/assets/style.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.clientside_callback(
    ClientsideFunction(namespace="clients", function_name="animateMessage"),
    Output("loading-message", "children"),
    Input("loading-message-store", "data"),
)
app.title = "Aurora Visualiser"
app.layout = dmc.MantineProvider(
    html.Div(
        className="responsive-container",
        children=[
            dcc.Loading(
                custom_spinner=custom_spinner,
                # make it blurry
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                delay_show=300,
                delay_hide=100,
                children=[
                    dcc.Tabs(
                        id="tabs",
                        value="tab-1",
                        content_style={"height": "100%", "overflow": "hidden"},
                        parent_style={"height": "100vh", "overflow": "hidden"},
                        children=[
                            # Samples tab
                            dcc.Tab(
                                label="Sample Plotting",
                                value="tab-1",
                                children=samples_layout,
                            ),
                            # Batches tab
                            dcc.Tab(
                                label="Batch Plotting",
                                value="tab-2",
                                children=batches_layout,
                            ),
                            # Database tab
                            dcc.Tab(
                                label="Database",
                                value="tab-3",
                                children=db_view_layout,
                            ),
                        ],
                    ),
                    dcc.Interval(id="db-update-interval", interval=1000 * 60 * 60),  # Auto-refresh database every hour
                    dcc.Store(id="table-data-store", data={"data": {}, "column_defs": {}}),
                    dcc.Store(id="samples-store", data=[]),
                    dcc.Store(id="batches-store", data={}),
                ],
            ),
            notifications_layout,
            dcc.Store(id="loading-message-store"),
            loading_message,
        ],
    ),
)

# Register all callback functions
register_samples_callbacks(app)
register_batches_callbacks(app)
register_db_view_callbacks(app)
register_notifications_callbacks(app)


def find_free_port(start_port: int = 8050, end_port: int = 8100) -> int:
    """Find a free port between start_port and end_port."""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    msg = f"No free ports available between {start_port} and {end_port}"
    raise RuntimeError(msg)


def main() -> None:
    """Open a web browser and run the app."""
    port = find_free_port()
    print(f"Running aurora-app on http://localhost:{port}")
    webbrowser.open_new(f"http://localhost:{port}")
    serve(app.server, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
