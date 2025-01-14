"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Web-based visualiser for the Aurora cycler manager based on Dash and Plotly.

Allows users to rapidly view and compare data from the Aurora robot and cycler
systems, both of individual samples and of batches of samples.

Allows users to view current information in the database, and control cyclers
remotely, loading, ejecting, and submitting jobs to samples.
"""

import webbrowser

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from waitress import serve

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.visualiser.batches import batches_layout, register_batches_callbacks
from aurora_cycler_manager.visualiser.db_view import db_view_layout, register_db_view_callbacks
from aurora_cycler_manager.visualiser.samples import register_samples_callbacks, samples_layout

#======================================================================================================================#
#================================================ GLOBAL VARIABLES ====================================================#
#======================================================================================================================#

# Config file
config = get_config()

# Spinner
custom_spinner=html.Div(
    style={"position": "relative", "width": "50px", "height": "50px"},
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

# Define app and layout
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Aurora Visualiser"
app.layout = html.Div(
    className="responsive-container",
    children = [
        dcc.Loading(
            custom_spinner=custom_spinner,
            # make it blurry
            overlay_style={"visibility": "visible", "filter": "blur(2px)"},
            delay_show=200,
            delay_hide=50,
            children = [
                dcc.Tabs(
                    id = "tabs",
                    value = "tab-1",
                    content_style = {"height": "100%", "overflow": "hidden"},
                    parent_style = {"height": "100vh", "overflow": "hidden"},
                    children = [
                        # Samples tab
                        dcc.Tab(
                            label="Sample Plotting",
                            value="tab-1",
                            children = samples_layout,
                        ),
                        # Batches tab
                        dcc.Tab(
                            label="Batch Plotting",
                            value="tab-2",
                            children = batches_layout,
                        ),
                        # Database tab
                        dcc.Tab(
                            label="Database",
                            value="tab-3",
                            children = db_view_layout(config),
                        ),
                    ],
                ),
                dcc.Interval(id="db-update-interval", interval=1000*60*60), # Auto-refresh database every hour
                dcc.Store(id="config-store", data = config),
                dcc.Store(id="table-data-store", data = {"data": [], "column_defs": []}),
                dcc.Store(id="samples-store", data = []),
                dcc.Store(id="batches-store", data = []),
            ],
        ),
    ],
)

# Register all callback functions
register_samples_callbacks(app,config)
register_batches_callbacks(app,config)
register_db_view_callbacks(app,config)

def main() -> None:
    """Open a web browser and run the app."""
    webbrowser.open_new("http://localhost:8050")
    serve(app.server, host="127.0.0.1", port=8050)

if __name__ == "__main__":
    main()
