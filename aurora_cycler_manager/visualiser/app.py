""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Web-based visualiser for the Aurora cycler manager based on Dash and Plotly.

Allows users to rapidly view and compare data from the Aurora robot and cycler
systems, both of individual samples and of batches of samples.
"""

import os
import sys
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import json
import webbrowser
from threading import Thread
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.visualiser.samples import samples_layout, register_samples_callbacks
from aurora_cycler_manager.visualiser.batches import batches_layout, register_batches_callbacks
from aurora_cycler_manager.visualiser.db_view import db_view_layout, register_db_view_callbacks

#======================================================================================================================#
#================================================ GLOBAL VARIABLES ====================================================#
#======================================================================================================================#

# Config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', '..', 'config.json')
with open(config_path, encoding = 'utf-8') as f:
    config = json.load(f)

# Define app and layout
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Aurora Visualiser"
app.layout = html.Div(
    style = {'height': 'calc(100vh - 30px)','overflow': 'hidden'},
    children = [
        dcc.Tabs(
            id = "tabs",
            value = 'tab-1',
            content_style = {'height': '100%', 'overflow': 'hidden'},
            parent_style = {'height': '100%', 'overflow': 'hidden'},
            children = [
                # Samples tab
                dcc.Tab(
                    label='Samples',
                    value='tab-1',
                    children = samples_layout(config)
                ),
                # Batches tab
                dcc.Tab(
                    label='Batches',
                    value='tab-2',
                    children = batches_layout(config)
                ),
                # Database tab
                dcc.Tab(
                    label='Database',
                    value='tab-3',
                    children = db_view_layout(config)
                )
            ]
        ),
    ]
)

# Register all callback functions
register_samples_callbacks(app,config)
register_batches_callbacks(app,config)
register_db_view_callbacks(app,config)

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://localhost:8050")
    Thread(target=open_browser).start()
    app.run_server(debug=True, use_reloader=False)
