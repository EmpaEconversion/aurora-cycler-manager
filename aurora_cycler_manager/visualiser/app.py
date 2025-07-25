"""Copyright © 2025, Empa.

Web-based visualiser for the Aurora cycler manager based on Dash and Plotly.

Allows users to rapidly view and compare data from the Aurora robot and cycler
systems, both of individual samples and of batches of samples.

Allows users to view current information in the database, and control cyclers
remotely, loading, ejecting, and submitting jobs to samples.
"""

import logging
import socket
import webbrowser

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import ClientsideFunction, Dash, Input, Output, _dash_renderer, dcc, html
from waitress import serve

from aurora_cycler_manager.setup_logging import setup_logging
from aurora_cycler_manager.visualiser.batches import batches_layout, register_batches_callbacks
from aurora_cycler_manager.visualiser.db_view import db_view_layout, register_db_view_callbacks
from aurora_cycler_manager.visualiser.notifications import (
    custom_spinner,
    loading_message,
    notifications_layout,
    register_notifications_callbacks,
)
from aurora_cycler_manager.visualiser.samples import register_samples_callbacks, samples_layout

setup_logging()
logger = logging.getLogger(__name__)

# Need to set this for Mantine notifications to work
_dash_renderer._set_react_version("18.2.0")  # noqa: SLF001

# Define app and layout
external_stylesheets = [dbc.icons.BOOTSTRAP, dmc.styles.NOTIFICATIONS, "/assets/style.css"]
dmc.add_figure_templates()
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.clientside_callback(
    ClientsideFunction(namespace="clients", function_name="animateMessage"),
    Output("loading-message", "children"),
    Input("loading-message-store", "data"),
)
app.title = "Aurora Visualiser"
app.layout = dmc.MantineProvider(
    id="mantine-provider",
    children=html.Div(
        className="responsive-container",
        children=[
            dcc.Loading(
                custom_spinner=custom_spinner,
                # make it blurry
                overlay_style={"visibility": "visible", "filter": "blur(2px)"},
                delay_show=300,
                delay_hide=100,
                children=[
                    dmc.Tabs(
                        [
                            dmc.TabsList(
                                [
                                    dmc.TabsTab("Sample Plotting", value="tab-1", fz="md", pt="md"),
                                    dmc.TabsTab("Batch Plotting", value="tab-2", fz="md", pt="md"),
                                    dmc.TabsTab("Database", value="tab-3", fz="md", pt="md"),
                                ],
                                grow=True,
                                style={"flexShrink": 0},
                            ),
                            dmc.TabsPanel(
                                samples_layout,
                                value="tab-1",
                                p="xs",
                                style={"flex": 1, "display": "flex", "flexDirection": "column", "minHeight": 0},
                            ),
                            dmc.TabsPanel(
                                batches_layout,
                                value="tab-2",
                                p="xs",
                                style={"flex": 1, "display": "flex", "flexDirection": "column", "minHeight": 0},
                            ),
                            dmc.TabsPanel(
                                db_view_layout,
                                value="tab-3",
                                p="xs",
                                style={"flex": 1, "display": "flex", "flexDirection": "column", "minHeight": 0},
                            ),
                        ],
                        id="tabs",
                        value="tab-1",
                        style={"display": "flex", "flexDirection": "column", "height": "100vh"},
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
    logger.info("Running aurora-app on http://localhost:%s", port)
    webbrowser.open_new(f"http://localhost:{port}")
    serve(app.server, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
