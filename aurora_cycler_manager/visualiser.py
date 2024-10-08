""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Web-based visualiser for the Aurora cycler manager based on Dash and Plotly.

Allows users to rapidly view and compare data from the Aurora robot and cycler
systems, both of individual samples and of batches of samples.
"""

import os
import sys
import dash
from dash import dcc, html, Input, Output, State, no_update, ALL
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
import plotly.graph_objs as go
import plotly.express as px
import textwrap
import yaml
import numpy as np
import json
import gzip
import sqlite3
import base64
import pandas as pd
import paramiko
import webbrowser
from threading import Thread
from datetime import datetime
from scipy import stats
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.server_manager import ServerManager
from aurora_cycler_manager.analysis import combine_jobs, _run_from_sample


#======================================================================================================================#
#================================================ GLOBAL VARIABLES ====================================================#
#======================================================================================================================#

# Config file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config.json')
with open(config_path, encoding = 'utf-8') as f:
    config = json.load(f)
db_path = config['Database path']
graph_config_path = config['Graph config path']
unused_pipelines = config.get('Unused pipelines', [])

# Graphs
graph_template = 'seaborn'
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Aurora Visualiser"

# Server manager, if user cannot ssh connect then they cannot interact with the servers, so disable these features
try:
    sm = ServerManager()
    print("Successfully connected to the servers. You have permissions to alter everything.")
    permissions = True
except paramiko.SSHException as e:
    print(f"You do not have permission to write to the servers. Disabling these features.")
    sm = None
    permissions = False


#======================================================================================================================#
#===================================================== FUNCTIONS ======================================================#
#======================================================================================================================#

def get_sample_names() -> list:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM samples")
        samples = cursor.fetchall()
    return [sample[0] for sample in samples]

def get_batch_names() -> list:
    with open(graph_config_path, 'r') as f:
        graph_config = yaml.safe_load(f)
    return list(graph_config.keys())

def get_database() -> dict:
    pipeline_query = "SELECT * FROM pipelines WHERE " + " AND ".join([f"Pipeline NOT LIKE '{pattern}'" for pattern in unused_pipelines])
    db_data = {
        'samples': pd.read_sql_query("SELECT * FROM samples", sqlite3.connect(db_path)).to_dict("records"),
        'results': pd.read_sql_query("SELECT * FROM results", sqlite3.connect(db_path)).to_dict("records"),
        'jobs': pd.read_sql_query("SELECT * FROM jobs", sqlite3.connect(db_path)).to_dict("records"),
        'pipelines': pd.read_sql_query(pipeline_query, sqlite3.connect(db_path)).to_dict("records")
    }
    db_columns = {
        'samples': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['samples'][0].keys()],
        'results': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['results'][0].keys()],
        'jobs': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['jobs'][0].keys()],
        'pipelines': [{'field' : col, 'filter': True, 'tooltipField': col} for col in db_data['pipelines'][0].keys()],
    }
    return {'data':db_data, 'column_defs': db_columns}
db_data = get_database()

def cramers_v(x, y):
    """ Calculate Cramer's V for two categorical variables. """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def anova_test(x, y):
    """ ANOVA test between categorical and continuous variables."""
    categories = x.unique()
    groups = [y[x == category] for category in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value

def correlation_ratio(categories, measurements):
    """ Measure of the relationship between a categorical and numerical variable. """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def correlation_matrix(
        df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the correlation matrix for a DataFrame including categorical columns.
    For continuous-continuous use Pearson correlation
    For continuous-categorical use correlation ratio
    For categorical-categorical use Cramer's V

    Args:
        df (pd.DataFrame): The DataFrame to calculate the correlation matrix for.
    """
    corr = pd.DataFrame(index=df.columns, columns=df.columns)
    # Calculate the correlation matrix
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                corr.loc[col1, col2] = 1.0
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr.loc[col1, col2] = df[[col1, col2]].corr().iloc[0, 1]
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_object_dtype(df[col2]):
                corr.loc[col1, col2] = correlation_ratio(df[col2], df[col1])
            elif pd.api.types.is_object_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr.loc[col1, col2] = correlation_ratio(df[col1], df[col2])
            elif pd.api.types.is_object_dtype(df[col1]) and pd.api.types.is_object_dtype(df[col2]):
                corr.loc[col1, col2] = cramers_v(df[col1], df[col2])
    return corr

def moving_average(x, npoints=11):
    if npoints % 2 == 0:
        npoints += 1  # Ensure npoints is odd for a symmetric window
    window = np.ones(npoints) / npoints
    xav = np.convolve(x, window, mode='same')
    xav[:npoints // 2] = np.nan
    xav[-npoints // 2:] = np.nan
    return xav

def deriv(x, y):
    with np.errstate(divide='ignore'):
        dydx = np.zeros(len(y))
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

    # for any 3 points where x direction changes sign set to nan
    mask = (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0
    dydx[1:-1][mask] = np.nan
    return dydx

def smoothed_derivative(x, y, npoints=21):
    x_smooth = moving_average(x, npoints)
    y_smooth = moving_average(y, npoints)
    dydx_smooth = deriv(x_smooth, y_smooth)
    dydx_smooth[deriv(x_smooth,np.arange(len(x_smooth))) < 0] *= -1
    dydx_smooth[abs(dydx_smooth) > 100] = np.nan
    return dydx_smooth


#======================================================================================================================#
#======================================================= LAYOUT =======================================================#
#======================================================================================================================#

colorscales = px.colors.named_colorscales()

# Side menu for the samples tab
samples_menu = html.Div(
    style = {'overflow': 'scroll', 'height': '100%'},
    children = [
        html.H5("Select samples to plot:"),
        dcc.Dropdown(
            id='samples-dropdown',
            options=[
                {'label': name, 'value': name} for name in get_sample_names()
            ],
            value=[],
            multi=True,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H5("Time graph"),
        html.Label('X-axis:', htmlFor='samples-time-x'),
        dcc.Dropdown(
            id='samples-time-x',
            options=['Unix time','From start','From formation','From cycling'],
            value='From start',
            multi=False,
        ),
        dcc.Dropdown(
            id='samples-time-units',
            options=['Seconds','Minutes','Hours','Days'],
            value='Hours',
        ),
        html.Div(style={'margin-top': '10px'}),
        html.Label('Y-axis:', htmlFor='samples-time-y'),
        dcc.Dropdown(
            id='samples-time-y',
            options=['V (V)'],
            value='V (V)',
            multi=False,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H5("Cycles graph"),
        html.P("X-axis: Cycle"),
        html.Label('Y-axis:', htmlFor='samples-cycles-y'),
        dcc.Dropdown(
            id='samples-cycles-y',
            options=[
                'Specific discharge capacity (mAh/g)',
                'Efficiency (%)',
            ],
            value='Specific discharge capacity (mAh/g)',
            multi=False,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H5("One cycle graph"),
        html.Label("X-axis:", htmlFor='samples-cycle-x'),
        dcc.Dropdown(
            id='samples-cycle-x',
            options=['Q (mAh)', 'V (V)', 'dQdV (mAh/V)'],
            value='Q (mAh)',
        ),
        html.Div(style={'margin-top': '10px'}),
        html.Label("Y-axis:", htmlFor='samples-cycle-y'),
        dcc.Dropdown(
            id='samples-cycle-y',
            options=['Q (mAh)', 'V (V)', 'dQdV (mAh/V)'],
            value='V (V)',
        ),
        html.Div(style={'margin-top': '100px'})
    ]
)

# Side menu for the batches tab
batches_menu = html.Div(
    style = {'overflow': 'scroll', 'height': '100%'},
    children = [
        html.H5("Select batches to plot:"),
        dcc.Dropdown(
            id='batches-dropdown',
            options=[
                {'label': name, 'value': name} for name in get_batch_names()
            ],
            value=[],
            multi=True,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H5("Cycles graph"),
        html.P("X-axis: Cycle"),
        html.Label("Y-axis:", htmlFor='batch-cycle-y'),
        dcc.Dropdown(
            id='batch-cycle-y',
            options=['Specific discharge capacity (mAh/g)'],
            value='Specific discharge capacity (mAh/g)',
            multi=False,
        ),
        html.Div(style={'margin-top': '10px'}),
        html.Label("Colormap", htmlFor='batch-cycle-colormap'),
        dcc.Dropdown(
            id='batch-cycle-color',
            options=[
                'Run ID'
            ],
            value='Run ID',
            multi=False,
        ),
        dcc.Dropdown(
            id='batch-cycle-colormap',
            options=colorscales,
            value='turbo'
        ),
        html.Div(style={'margin-top': '10px'}),
        html.Label("Style", htmlFor='batch-cycle-style'),
        dcc.Dropdown(
            id='batch-cycle-style',
            options=[
            ],
            multi=False,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H5("Correlation graph"),
        html.Label("X-axis:", htmlFor='batch-correlation-x'),
        dcc.Dropdown(
            id='batch-correlation-x',
        ),
        html.Div(style={'margin-top': '10px'}),
        html.Label("Y-axis:", htmlFor='batch-correlation-y'),
        dcc.Dropdown(
            id='batch-correlation-y',
        ),
        html.Div(style={'margin-top': '10px'}),
        html.Label("Colormap", htmlFor='batch-correlation-color'),
        dcc.Dropdown(
            id='batch-correlation-color',
            options=[
                'Run ID'
            ],
            value='Run ID',
            multi=False,
        ),
        dcc.Dropdown(
            id='batch-correlation-colorscale',
            options=colorscales,
            value='turbo',
            multi=False,
        ),
        html.Div(style={'margin-top': '100px'})
    ]
)

# Main layout
app.layout = html.Div(
    style = {'height': 'calc(100vh - 30px)','overflow': 'hidden'},
    children = [
        dcc.Tabs(
            id = "tabs",
            value = 'tab-1',
            content_style = {'height': '100%', 'overflow': 'hidden'},
            parent_style = {'height': '100%', 'overflow': 'hidden'},
            children = [
                #################### SAMPLES TAB WITH PANELS ####################
                dcc.Tab(
                    label='Samples',
                    value='tab-1',
                    children = [
                        dcc.Store(id='samples-data-store', data={'data_sample_time': {}, 'data_sample_cycle': {}}),
                        html.Div(
                            style={'height': '100%'},
                            children = [
                                PanelGroup(
                                    id="samples-panel-group",
                                    direction="horizontal",
                                    children=[
                                        Panel(
                                            id="samples-menu",
                                            className="menu-panel",
                                            children=samples_menu,
                                            defaultSizePercentage=16,
                                            collapsible=True,
                                        ),
                                        PanelResizeHandle(html.Div(className="resize-handle-horizontal")),
                                        Panel(
                                            id="samples-graphs",
                                            children=[
                                                PanelGroup(
                                                    id="samples-graph-group",
                                                    direction="vertical",
                                                    children=[
                                                        Panel(
                                                            id="samples-top-graph",
                                                            children=[
                                                                dcc.Graph(id='time-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='vs time',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'},showlegend=False)}, config={'scrollZoom':True, 'displaylogo':False},style={'height': '100%'}),
                                                            ]),
                                                        PanelResizeHandle(
                                                            html.Div(className="resize-handle-vertical")
                                                        ),
                                                        Panel(
                                                            id="samples-bottom-graphs",
                                                            children=[
                                                                PanelGroup(
                                                                    id="samples-bottom-graph-group",
                                                                    direction="horizontal",
                                                                    children=[
                                                                        Panel(
                                                                            id="samples-bottom-left-graph",
                                                                            children=[
                                                                                dcc.Graph(id='cycles-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='vs cycle',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'},showlegend=False)}, config={'scrollZoom':True, 'displaylogo':False},style={'height': '100%'}),
                                                                            ]
                                                                        ),
                                                                        PanelResizeHandle(
                                                                            html.Div(className="resize-handle-horizontal")
                                                                        ),
                                                                        Panel(
                                                                            id="samples-bottom-right-graph",
                                                                            children=[
                                                                                dcc.Graph(id='cycle-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='One cycle',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'},showlegend=False)}, config={'scrollZoom':True, 'displaylogo':False},style={'height': '100%'}),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                    
                                                )
                                            ],
                                            minSizePercentage=50,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                #################### BATCHES TAB WITH PANELS ####################
                dcc.Tab(
                    label='Batches',
                    value='tab-2',
                    children = [
                        dcc.Loading(
                            id='loading-page',
                            type='circle',
                            children=[ dcc.Store(id='batches-data-store', data={'data_batch_cycle': {}}),],
                            style={
                                'position': 'fixed',
                                'bottom': '1%',
                                'left': '1%',
                                'size': '500%',
                                'zIndex': 1000
                            },
                            parent_style={
                                'position': 'relative'
                            },
                        ),
                        html.Div([
                            PanelGroup(
                                id="batches-panel-group",
                                children=[
                                    Panel(
                                        id="batches-menu",
                                        className="menu-panel",
                                        children=batches_menu,
                                        defaultSizePercentage=16,
                                        collapsible=True,
                                    ),
                                    PanelResizeHandle(html.Div(className="resize-handle-horizontal")),
                                    Panel(
                                        id="graphs",
                                        children=[
                                            PanelGroup(
                                                id="graph group",
                                                children=[
                                                    Panel(
                                                        id="top graph",
                                                        children=[
                                                            dcc.Graph(
                                                                id='batch-cycle-graph',
                                                                figure={
                                                                    'data': [],
                                                                    'layout': go.Layout(template=graph_template, title='vs cycle', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                                                }, 
                                                                config={'scrollZoom': True, 'displaylogo':False,  'toImageButtonOptions': {'format': 'svg',}},
                                                                style={'height': '100%'}
                                                            ),
                                                        ]),
                                                    PanelResizeHandle(
                                                        html.Div(className="resize-handle-vertical")
                                                    ),
                                                    Panel(
                                                        id="bottom graphs",
                                                        children=[
                                                            PanelGroup(
                                                                id="bottom graph group",
                                                                children=[
                                                                    Panel(
                                                                        id="bottom left graph",
                                                                        children=[
                                                                            dcc.Graph(
                                                                                id='batch-correlation-map',
                                                                                figure={
                                                                                    'data': [],
                                                                                    'layout': go.Layout(template=graph_template, title='Click to show correlation', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                                                                }, 
                                                                                config={'scrollZoom': False, 'displaylogo':False, 'modeBarButtonsToRemove' : ['zoom2d','pan2d','zoomIn2d','zoomOut2d','autoScale2d','resetScale2d'], 'toImageButtonOptions': {'format': 'png', 'width': 1000, 'height': 800}},
                                                                                style={'height': '100%'},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    PanelResizeHandle(
                                                                        html.Div(className="resize-handle-horizontal")
                                                                    ),
                                                                    Panel(
                                                                        id="bottom right graph",
                                                                        children=[
                                                                            dcc.Graph(
                                                                                id='batch-correlation-graph',
                                                                                figure={
                                                                                    'data': [],
                                                                                    'layout': go.Layout(template=graph_template, title='params', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                                                                },
                                                                                config={'scrollZoom': True, 'displaylogo':False, 'toImageButtonOptions': {'format': 'svg'}},
                                                                                style={'height': '100%'},
                                                                            ),
                                                                        ]
                                                                    ),
                                                                ],
                                                                direction="horizontal",
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                                direction="vertical",
                                            )
                                        ],
                                        minSizePercentage=50,
                                    ),
                                ],
                                direction="horizontal",
                            )
                        ],
                        style={'height': '100%'},
                        ),
                    ],
                ),
                ##### Database tab #####
                dcc.Tab(
                    label='Database',
                    value='tab-3',
                    children = [
                        html.Div(
                            style={'height': '100%', 'overflowY': 'scroll', 'overflowX': 'scroll', 'padding': '10px'},
                            children = [
                                dcc.Loading(
                                    id='loading-database',
                                    type='circle',
                                    delay_show=200,
                                    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                                    style={'height': '100%'},
                                    children=[
                                        html.Div(
                                            style={'height': '100%'},
                                            children = [
                                                dcc.Store(id='table-data-store', data=db_data),
                                                # Buttons to refresh or update the database
                                                html.P(children = f"Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", id='last-refreshed'),
                                                html.P(children = f"Last updated: {db_data['data']['pipelines'][0]['Last checked']}", id='last-updated'),
                                                dbc.Button("Refresh database", id='refresh-database', color='primary', outline=True, className='me-1'),
                                                dbc.Button("Force update database", id='update-database', color='warning', outline=True, className='me-1', disabled = not permissions),
                                                html.Div(style={'margin-top': '10px'}),
                                                # Buttons to select which table to display
                                                dbc.RadioItems(
                                                    id='table-select',
                                                    inline=True,
                                                    options=[
                                                        {'label': 'Samples', 'value': 'samples'},
                                                        {'label': 'Results', 'value': 'results'},
                                                        {'label': 'Jobs', 'value': 'jobs'},
                                                        {'label': 'Pipelines', 'value': 'pipelines'},
                                                    ],
                                                    value='pipelines',
                                                ),
                                                html.Div(style={'margin-top': '10px'}),
                                                # Main table for displaying info from database
                                                dag.AgGrid(
                                                    id='table',
                                                    dashGridOptions = {"enableCellTextSelection": False, "tooltipShowDelay": 1000, 'rowSelection': 'multiple'},
                                                    style={"height": "calc(90vh - 220px)", "width": "100%", "minHeight": "400px"},
                                                ),
                                                # Buttons to interact with the database
                                                dbc.Button("Load", id='load-button', color='primary', outline=True, className='me-1'),
                                                dbc.Button("Eject", id='eject-button', color='primary', outline=True, className='me-1'),
                                                dbc.Button("Ready", id='ready-button', color='primary', outline=True, className='me-1'),
                                                dbc.Button("Unready", id='unready-button', color='primary', outline=True, className='me-1'),
                                                dbc.Button("Submit", id='submit-button', color='primary', outline=True, className='me-1'),
                                                dbc.Button("Cancel", id='cancel-button', color='danger', outline=True, className='me-1'),
                                            ]
                                        ),
                                        # Pop up modals for interacting with the database after clicking buttons
                                        # Eject
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Eject")),
                                                dbc.ModalBody(id='eject-modal-body',children="Are you sure you want eject the selected samples?"),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button(
                                                            "Eject", id="eject-yes-close", className="ms-auto", n_clicks=0, color='primary'
                                                        ),
                                                        dbc.Button(
                                                            "Go back", id="eject-no-close", className="ms-auto", n_clicks=0, color='secondary'
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            id="eject-modal",
                                            is_open=False,
                                        ),
                                        # Load
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Load")),
                                                dbc.ModalBody(
                                                    id='load-modal-body',
                                                    children=[
                                                        "Select the samples you want to load",
                                                        dcc.Dropdown(
                                                            id='load-dropdown',
                                                            options=[
                                                                {'label': name, 'value': name} for name in get_sample_names()
                                                            ],
                                                            value=[],
                                                            multi=True,
                                                        ),
                                                    ]
                                                ),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button(
                                                            "Load", id="load-yes-close", className="ms-auto", color='primary', n_clicks=0
                                                        ),
                                                        dbc.Button(
                                                            "Go back", id="load-no-close", className="ms-auto", color='secondary', n_clicks=0
                                                        ),
                                                    ]
                                                ),
                                                dcc.Store(id='load-modal-store', data={}),
                                            ],
                                            id="load-modal",
                                            is_open=False,
                                        ),
                                        # Ready
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Ready")),
                                                dbc.ModalBody(id='ready-modal-body',children="Are you sure you want ready the selected pipelines? You must force update the database afterwards to check if tomato has started the job(s)."),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button(
                                                            "Ready", id="ready-yes-close", className="ms-auto", n_clicks=0, color='primary'
                                                        ),
                                                        dbc.Button(
                                                            "Go back", id="ready-no-close", className="ms-auto", n_clicks=0, color='secondary'
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            id="ready-modal",
                                            is_open=False,
                                        ),
                                        # Unready
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Unready")),
                                                dbc.ModalBody(id='unready-modal-body',children="Are you sure you want un-ready the selected pipelines?"),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button(
                                                            "Unready", id="unready-yes-close", className="ms-auto", n_clicks=0, color='primary'
                                                        ),
                                                        dbc.Button(
                                                            "Go back", id="unready-no-close", className="ms-auto", n_clicks=0, color='secondary'
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            id="unready-modal",
                                            is_open=False,
                                        ),
                                        # Submit
                                        dbc.Modal(
                                            [
                                                dcc.Store(id='payload', data={}),
                                                dbc.ModalHeader(dbc.ModalTitle("Submit")),
                                                dbc.ModalBody(
                                                    id='submit-modal-body',
                                                    style={'width': '100%'},
                                                    children=[
                                                        "Select a tomato .json payload to submit",
                                                        dcc.Upload(
                                                            id='submit-upload',
                                                            children=html.Div([
                                                                'Drag and Drop or ',
                                                                html.A('Select Files')
                                                            ]),
                                                            style={
                                                                'width': '100%',
                                                                'height': '60px',
                                                                'lineHeight': '60px',
                                                                'borderWidth': '1px',
                                                                'borderStyle': 'dashed',
                                                                'borderRadius': '8px',
                                                                'textAlign': 'center',
                                                            },
                                                            accept='.json',
                                                            multiple=False,
                                                        ),
                                                        html.P(children="No file selected", id='validator'),
                                                        html.Div(style={'margin-top': '10px'}),
                                                        html.Div([
                                                            html.Label("Calculate C-rate by:", htmlFor='submit-crate'),
                                                            dcc.Dropdown(
                                                                id='submit-crate',
                                                                options=[
                                                                    {'value': 'areal', 'label': 'areal capacity x area from db'},
                                                                    {'value': 'mass', 'label': 'specific capacity x mass  from db'},
                                                                    {'value': 'nominal', 'label': 'nominal capacity from db'},
                                                                    {'value': 'custom', 'label': 'custom capacity value'},
                                                                ],
                                                            )
                                                        ]),
                                                        html.Div(
                                                            id='submit-capacity-div',
                                                            children=[
                                                                "Capacity = ",
                                                                dcc.Input(id='submit-capacity', type='number', min=0, max=10, step=0.1),
                                                                " mAh"
                                                            ],
                                                            style={'display': 'none'},
                                                        ),
                                                    ]
                                                ),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button(
                                                            "Submit", id="submit-yes-close", className="ms-auto", n_clicks=0, color='primary', disabled=True
                                                        ),
                                                        dbc.Button(
                                                            "Go back", id="submit-no-close", className="ms-auto", n_clicks=0, color='secondary'
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            id="submit-modal",
                                            is_open=False,
                                        ),
                                        # Cancel
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(dbc.ModalTitle("Cancel")),
                                                dbc.ModalBody(id='cancel-modal-body',children="Are you sure you want to cancel the selected jobs?"),
                                                dbc.ModalFooter(
                                                    [
                                                        dbc.Button(
                                                            "Cancel", id="cancel-yes-close", className="ms-auto", n_clicks=0, color='danger'
                                                        ),
                                                        dbc.Button(
                                                            "Go back", id="cancel-no-close", className="ms-auto", n_clicks=0, color='secondary'
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            id="cancel-modal",
                                            is_open=False,
                                        ),
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
    ]
)


#======================================================================================================================#
#===================================================== CALLBACKS ======================================================#
#======================================================================================================================#

#----------------------------- DATABASE CALLBACKS ------------------------------#

# Update the buttons displayed depending on the table selected
@app.callback(
    Output('table', 'rowData'),
    Output('table', 'columnDefs'),
    Output('load-button', 'style'),
    Output('eject-button', 'style'),
    Output('ready-button', 'style'),
    Output('unready-button', 'style'),
    Output('submit-button', 'style'),
    Output('cancel-button', 'style'),
    Input('table-select', 'value'),
    Input('table-data-store', 'data'),
)
def update_table(table, data):
    load = {'display': 'none'}
    eject = {'display': 'none'}
    ready = {'display': 'none'}
    unready = {'display': 'none'}
    cancel = {'display': 'none'}
    submit = {'display': 'none'}
    if table == 'pipelines':
        load = {'display': 'inline-block'}
        eject = {'display': 'inline-block'}
        ready = {'display': 'inline-block'}
        unready = {'display': 'inline-block'}
        cancel = {'display': 'inline-block'}
        submit = {'display': 'inline-block'}
    elif table == 'jobs':
        cancel = {'display': 'inline-block'}
    return data['data'][table], data['column_defs'][table], load, eject, ready, unready, submit, cancel

# Refresh the local data from the database
@app.callback(
    Output('table-data-store', 'data'),
    Output('last-refreshed', 'children'),
    Output('last-updated', 'children'),
    Input('refresh-database', 'n_clicks'),
)
def refresh_database(n_clicks):
    if n_clicks is None:
        return no_update
    db_data = get_database()
    return db_data, f"Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", f"Last updated: {db_data['data']['pipelines'][0]['Last checked']}"

# Update the database i.e. connect to servers and grab new info, then refresh the local data
@app.callback(
    Output('refresh-database', 'n_clicks', allow_duplicate=True),
    Input('update-database', 'n_clicks'),
    prevent_initial_call=True,
)
def update_database(n_clicks):
    if n_clicks is None:
        return no_update
    print("Updating database")
    sm.update_db()
    return 1

# Enable or disable buttons (load, eject, etc.) depending on what is selected in the table
@app.callback(
    Output('load-button', 'disabled'),
    Output('eject-button', 'disabled'),
    Output('ready-button', 'disabled'),
    Output('unready-button', 'disabled'),
    Output('submit-button', 'disabled'),
    Output('cancel-button', 'disabled'),
    Input('table', 'selectedRows'),
    State('table-select', 'value'),
)
def enable_buttons(selected_rows, table):
    load, eject, ready, unready, submit, cancel = True,True,True,True,True,True
    if not permissions:
        return True, True, True, True, True, True
    if not selected_rows:
        return True, True, True, True, True, True
    if table == 'pipelines':
        if all([s['Sample ID'] is not None for s in selected_rows]):
            submit = False
            if all([s['Job ID'] is None for s in selected_rows]):
                eject = False
                ready = False
                unready = False
            elif all([s['Job ID'] is not None for s in selected_rows]):
                cancel = False
        elif all([s['Sample ID']==None for s in selected_rows]):
            load = False
    if table == 'jobs':
        if all([s['Status'] in ['r','q','qw'] for s in selected_rows]):
            cancel = False
    return load, eject, ready, unready, submit, cancel

# Eject button pop up
@app.callback(
    Output("eject-modal", "is_open"),
    Input('eject-button', 'n_clicks'),
    Input('eject-yes-close', 'n_clicks'),
    Input('eject-no-close', 'n_clicks'),
    State('eject-modal', 'is_open'),
)
def eject_sample_button(eject_clicks, yes_clicks, no_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'eject-button':
        return not is_open
    elif button_id == 'eject-yes-close' and yes_clicks:
        return False
    elif button_id == 'eject-no-close' and no_clicks:
        return False
    return is_open, no_update, no_update, no_update
# When eject button confirmed, eject samples and refresh the database
@app.callback(
    Output('loading-database', 'children', allow_duplicate=True),
    Output('refresh-database', 'n_clicks', allow_duplicate=True),
    Input('eject-yes-close', 'n_clicks'),
    State('table-data-store', 'data'),
    State('table', 'selectedRows'),
    prevent_initial_call=True,
)
def eject_sample(yes_clicks, data, selected_rows):
    if not yes_clicks:
        return no_update,0
    for row in selected_rows:
        print(f"Ejecting {row['Pipeline']}")
        sm.eject(row['Pipeline'])
    return no_update,1

# Load button pop up, includes dynamic dropdowns for selecting samples to load
@app.callback(
    Output("load-modal", "is_open"),
    Output("load-modal-body", "children"),
    Input('load-button', 'n_clicks'),
    Input('load-yes-close', 'n_clicks'),
    Input('load-no-close', 'n_clicks'),
    State('load-modal', 'is_open'),
    State('table', 'selectedRows'),
)
def load_sample_button(load_clicks, yes_clicks, no_clicks, is_open, selected_rows):
    ctx = dash.callback_context
    if not selected_rows or not ctx.triggered:
        return is_open, no_update
    options = [{'label': name, 'value': name} for name in get_sample_names()]
    dropdowns = [
        html.Div(
            children=[
                html.Label(
                    f"{s['Pipeline']}",
                    htmlFor=f'dropdown-{s['Pipeline']}',
                    style={'margin-right': '10px'},
                ),
                dcc.Dropdown(
                    id={'type':'load-dropdown','index':i},
                    options=options,
                    value=[],
                    multi=False,
                    style={'width': '100%'}
                ),
            ],
            style={'display': 'flex', 'align-items': 'center', 'padding': '5px'}
        )
        for i,s in enumerate(selected_rows)
    ]
    children = ["Select the samples you want to load"] + dropdowns
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'load-button':
        return not is_open, children
    elif button_id == 'load-yes-close' and yes_clicks:
        return False, no_update
    elif button_id == 'load-no-close' and no_clicks:
        return False, no_update
    return is_open, no_update
# When load is pressed, load samples and refresh the database
@app.callback(
    Output('loading-database', 'children', allow_duplicate=True),
    Output('refresh-database', 'n_clicks', allow_duplicate=True),
    Input('load-yes-close', 'n_clicks'),
    State('table', 'selectedRows'),
    State({"type": "load-dropdown", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def load_sample(yes_clicks, selected_rows, selected_samples):
    if not yes_clicks:
        return no_update,0
    selected_pipelines = [s['Pipeline'] for s in selected_rows]
    for sample, pipeline in zip(selected_samples, selected_pipelines):
        if not sample:
            continue
        print(f"Loading {sample} to {pipeline}")
        sm.load(sample, pipeline)
    return no_update, 1

# Ready button pop up
@app.callback(
    Output("ready-modal", "is_open"),
    Input('ready-button', 'n_clicks'),
    Input('ready-yes-close', 'n_clicks'),
    Input('ready-no-close', 'n_clicks'),
    State('ready-modal', 'is_open'),
)
def ready_pipeline_button(ready_clicks, yes_clicks, no_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'ready-button':
        return not is_open
    elif button_id == 'ready-yes-close' and yes_clicks:
        return False
    elif button_id == 'ready-no-close' and no_clicks:
        return False
    return is_open, no_update, no_update, no_update
# When ready button confirmed, ready pipelines and refresh the database
@app.callback(
    Output('loading-database', 'children', allow_duplicate=True),
    Input('ready-yes-close', 'n_clicks'),
    State('table', 'selectedRows'),
    prevent_initial_call=True,
)
def ready_pipeline(yes_clicks, selected_rows):
    if not yes_clicks:
        return no_update
    for row in selected_rows:
        print(f"Readying {row['Pipeline']}")
        output = sm.ready(row['Pipeline'])
    return no_update

# Unready button pop up
@app.callback(
    Output("unready-modal", "is_open"),
    Input('unready-button', 'n_clicks'),
    Input('unready-yes-close', 'n_clicks'),
    Input('unready-no-close', 'n_clicks'),
    State('unready-modal', 'is_open'),
)
def unready_pipeline_button(unready_clicks, yes_clicks, no_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'unready-button':
        return not is_open
    elif button_id == 'unready-yes-close' and yes_clicks:
        return False
    elif button_id == 'unready-no-close' and no_clicks:
        return False
    return is_open, no_update, no_update, no_update
# When unready button confirmed, unready pipelines and refresh the database
@app.callback(
    Output('loading-database', 'children', allow_duplicate=True),
    Input('unready-yes-close', 'n_clicks'),
    State('table', 'selectedRows'),
    prevent_initial_call=True,
)
def unready_pipeline(yes_clicks, selected_rows):
    if not yes_clicks:
        return no_update
    for row in selected_rows:
        print(f"Unreadying {row['Pipeline']}")
        output = sm.unready(row['Pipeline'])
    return no_update

# Submit button pop up
@app.callback(
    Output("submit-modal", "is_open"),
    Input('submit-button', 'n_clicks'),
    Input('submit-yes-close', 'n_clicks'),
    Input('submit-no-close', 'n_clicks'),
    State('submit-modal', 'is_open'),
)
def submit_pipeline_button(submit_clicks, yes_clicks, no_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'submit-button':
        return not is_open
    elif button_id == 'submit-yes-close' and yes_clicks:
        return False
    elif button_id == 'submit-no-close' and no_clicks:
        return False
    return is_open, no_update, no_update, no_update
# Submit pop up - check that the json file is valid
@app.callback(
    Output('validator', 'children'),
    Output('payload', 'data'),
    Input('submit-upload', 'contents'),
    State('submit-upload', 'filename'),
    prevent_initial_call=True,
)
def check_json(contents, filename):
    if not contents:
        return "No file selected", {}
    content_type, content_string = contents.split(',')
    try:
        decoded = base64.b64decode(content_string).decode('utf-8')
    except UnicodeDecodeError:
        return f"ERROR: {filename} had decoding error", {}
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        return f"ERROR: {filename} is invalid json file", {}
    # TODO use proper tomato schemas to validate the json
    missing_keys = [key for key in ['version','method','tomato'] if key not in payload.keys()]
    if missing_keys:
        return f"ERROR: {filename} is missing keys: {", ".join(["'"+key+"'" for key in missing_keys])}", {}
    return f"{filename} loaded", payload
# Submit pop up - show custom capacity input if custom capacity is selected
@app.callback(
    Output('submit-capacity-div', 'style'),
    Input('submit-crate', 'value'),
    prevent_initial_call=True,
)
def submit_custom_crate(crate):
    if crate == 'custom':
        return {'display': 'block'}
    return {'display': 'none'}
# Submit pop up - enable submit button if json valid and a capacity is given
@app.callback(
    Output('submit-yes-close', 'disabled'),
    Input('payload', 'data'),
    Input('submit-crate', 'value'),
    Input('submit-capacity', 'value'),
    prevent_initial_call=True,
)
def enable_submit(payload, crate, capacity):
    if not payload or not crate:
        return True  # disabled
    if crate == 'custom':
        if not capacity or capacity < 0 or capacity > 10:
            return True  # disabled
    return False  # enabled
# When submit button confirmed, submit the payload with sample and capacity, refresh database
@app.callback(
    Output('loading-database', 'children', allow_duplicate=True),
    Output('refresh-database', 'n_clicks', allow_duplicate=True),
    Input('submit-yes-close', 'n_clicks'),
    State('table', 'selectedRows'),
    State('payload', 'data'),
    State('submit-crate', 'value'),
    State('submit-capacity', 'value'),
    prevent_initial_call=True,
)
def submit_pipeline(yes_clicks, selected_rows, payload, crate_calc, capacity):
    if not yes_clicks:
        return no_update, 0
    # capacity_Ah: float | 'areal','mass','nominal'
    if crate_calc == 'custom':
        capacity_Ah = capacity/1000
    else:
        capacity_Ah = crate_calc
    if not isinstance(capacity_Ah, float):
        if capacity_Ah not in ['areal','mass','nominal']:
            print(f"Invalid capacity calculation method: {capacity_Ah}")
            return no_update, 0
    for row in selected_rows:
        print(f"Submitting payload {payload} to sample {row['Sample ID']} with capacity_Ah {capacity_Ah}")
        # TODO gracefully handle errors here
        sm.submit(row['Sample ID'], payload, capacity_Ah)
    return no_update, 1

# Cancel button pop up
@app.callback(
    Output("cancel-modal", "is_open"),
    Input('cancel-button', 'n_clicks'),
    Input('cancel-yes-close', 'n_clicks'),
    Input('cancel-no-close', 'n_clicks'),
    State('cancel-modal', 'is_open'),
)
def cancel_job_button(cancel_clicks, yes_clicks, no_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'cancel-button':
        return not is_open
    elif button_id == 'cancel-yes-close' and yes_clicks:
        return False
    elif button_id == 'cancel-no-close' and no_clicks:
        return False
    return is_open, no_update, no_update, no_update
# When cancel confirmed, cancel the jobs and refresh the database
@app.callback(
    Output('refresh-database', 'n_clicks', allow_duplicate=True),
    Output('loading-database', 'children', allow_duplicate=True),
    Input('cancel-yes-close', 'n_clicks'),
    State('table', 'selectedRows'),
    prevent_initial_call=True,
)
def cancel_job(yes_clicks, selected_rows):
    if not yes_clicks:
        return no_update, 0
    for row in selected_rows:
        print(f"Cancelling job {row['Job ID']}")
        sm.cancel(row['Job ID'])
    return no_update, 1

#----------------------------- SAMPLES CALLBACKS ------------------------------#

# Update the samples data store
@app.callback(
    Output('samples-data-store', 'data'),
    Output('samples-time-y', 'options'),
    Output('samples-cycles-y', 'options'),
    Input('samples-dropdown', 'value'),
    Input('samples-data-store', 'data'),
)
def update_sample_data(samples, data):
    # Get rid of samples that are no longer selected
    for sample in list(data['data_sample_time'].keys()):
        if sample not in samples:
            data['data_sample_time'].pop(sample)
            if sample in data['data_sample_cycle'].keys():
                data['data_sample_cycle'].pop(sample)

    for sample in samples:
        # Check if already in data store
        if sample in data['data_sample_time'].keys():
            continue

        # Otherwise import the data
        run_id = _run_from_sample(sample)
        data_folder = config['Processed snapshots folder path']
        file_location = os.path.join(data_folder,run_id,sample)

        # Get raw data
        try:
            files = os.listdir(file_location)
        except FileNotFoundError:
            continue
        if any(f.startswith('full') and f.endswith('.json.gz') for f in files):
            filepath = next(f for f in files if f.startswith('full') and f.endswith('.json.gz'))
            with gzip.open(f'{file_location}/{filepath}', 'rb') as f:
                data_dict = json.load(f)['data']
            data['data_sample_time'][sample] = data_dict
        elif any(f.startswith('full') and f.endswith('.h5') for f in files):
            filepath = next(f for f in files if f.startswith('full') and f.endswith('.h5'))
            df = pd.read_hdf(f'{file_location}/{filepath}')
            data['data_sample_time'][sample] = df.to_dict(orient='list')
        else:
            cycling_files = [
                os.path.join(file_location,f) for f in files
                if (f.startswith('snapshot') and f.endswith('.json.gz'))
            ]
            if not cycling_files:
                cycling_files = [
                    os.path.join(file_location,f) for f in files
                    if (f.startswith('snapshot') and f.endswith('.h5'))
                ]
                if not cycling_files:
                    print(f"No cycling files found in {file_location}")
                    continue
            df, metadata = combine_jobs(cycling_files)
            data['data_sample_time'][sample] = df.to_dict(orient='list')

        # Get the analysed file
        try:
            analysed_file = next(f for f in files if (f.startswith('cycles') and f.endswith('.json')))
        except StopIteration:
            continue
        with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
            cycle_dict = json.load(f)["data"]
        if not cycle_dict or 'Cycle' not in cycle_dict.keys():
            continue
        data['data_sample_cycle'][sample] = cycle_dict
    
    # Update the y-axis options
    time_y_vars = set(['V (V)'])
    for sample, data_dict in data['data_sample_time'].items():
        time_y_vars.update(data_dict.keys())
    time_y_vars = list(time_y_vars)

    cycles_y_vars = set(['Specific discharge capacity (mAh/g)', 'Normalised discharge capacity (%)', 'Efficiency (%)'])
    for sample, data_dict in data['data_sample_cycle'].items():
        cycles_y_vars.update([k for k,v in data_dict.items() if isinstance(v,list)])
    cycles_y_vars = list(cycles_y_vars)
    
    return data, time_y_vars, cycles_y_vars

# Update the time graph
@app.callback(
    Output('time-graph', 'figure'),
    Input('samples-data-store', 'data'),
    Input('samples-time-x', 'value'),
    Input('samples-time-units', 'value'),
    Input('samples-time-y', 'value'),
)
def update_time_graph(data, xvar, xunits, yvar):
    fig = px.scatter().update_layout(title='No data...', xaxis_title=f'Time ({xunits.lower()})', yaxis_title=yvar,showlegend=False)
    fig.update_layout(template = graph_template)
    if not data['data_sample_time'] or not xvar or not yvar or not xunits:
        return fig

    multiplier = {'Seconds': 1, 'Minutes': 60, 'Hours': 3600, 'Days': 86400}[xunits]
    for sample, data_dict in data['data_sample_time'].items():
        uts = np.array(data_dict['uts'])
        if xvar == 'From start':
            offset=uts[0]
        elif xvar == 'From formation':
            offset=uts[next(i for i, x in enumerate(data_dict['Cycle']) if x >= 1)]
        elif xvar == 'From cycling':
            offset=uts[next(i for i, x in enumerate(data_dict['Cycle']) if x >= 4)]
        else:
            offset=0

        trace = go.Scatter(
            x=(np.array(data_dict['uts']) - offset) / multiplier,
            y=data_dict[yvar],
            mode='lines',
            name=sample,
            hovertemplate=f'{sample}<br>Time: %{{x}}<br>{yvar}: %{{y}}<extra></extra>',
        )
        fig.add_trace(trace)

    fig.update_layout(
        title=f'{yvar} vs time',
    )

    return fig


# Update the cycles graph
@app.callback(
    Output('cycles-graph', 'figure'),
    Input('samples-data-store', 'data'),
    Input('samples-cycles-y', 'value'),
)
def update_cycles_graph(data, yvar):
    fig = px.scatter().update_layout(title='No data...', xaxis_title='Cycle', yaxis_title=yvar,showlegend=False)
    fig.update_layout(template = graph_template)
    if not data['data_sample_cycle'] or not yvar:
        return fig
    for sample, cycle_dict in data['data_sample_cycle'].items():
        trace = go.Scatter(
            x=cycle_dict['Cycle'],
            y=cycle_dict[yvar],
            mode='lines+markers',
            name=sample,
            hovertemplate=f'{sample}<br>Cycle: %{{x}}<br>{yvar}: %{{y}}<extra></extra>',
        )
        fig.add_trace(trace)
    fig.update_layout(title=f'{yvar} vs cycle')
    return fig

# Update the one cycle graph
@app.callback(
    Output('cycle-graph', 'figure'),
    Input('cycles-graph', 'clickData'),
    Input('samples-data-store', 'data'),
    Input('samples-cycle-x', 'value'),
    Input('samples-cycle-y', 'value'),
)
def update_cycle_graph(clickData,data,xvar,yvar):
    fig = px.scatter().update_layout(title='No data...', xaxis_title=xvar, yaxis_title=yvar,showlegend=False)
    fig.update_layout(template = graph_template)
    if not data['data_sample_cycle'] or not xvar or not yvar:
        return fig
    
    if not clickData:
        cycle = 1
    else:
        # clickData is a dict with keys 'points' and 'event'
        # 'points' is a list of dicts with keys 'curveNumber', 'pointNumber', 'pointIndex', 'x', 'y', 'text'
        point = clickData['points'][0]
        cycle = point['x']
    traces = []
    for sample, data_dict in data['data_sample_time'].items():
        # find where the cycle = cycle
        mask = np.array(data_dict['Cycle']) == cycle
        if not any(mask):
            continue
        mask_dict = {}
        mask_dict['V (V)'] = np.array(data_dict['V (V)'])[mask]
        mask_dict['Q (mAh)'] = np.array(data_dict['dQ (mAh)'])[mask].cumsum()
        mask_dict['dQdV (mAh/V)'] = smoothed_derivative(mask_dict['V (V)'], mask_dict['Q (mAh)'])
        trace = go.Scatter(
            x=mask_dict[xvar],
            y=mask_dict[yvar],
            mode='lines',
            name=sample,
            hovertemplate=f'{sample}<br>{xvar}: %{{x}}<br>{yvar}: %{{y}}<extra></extra>',
        )
        fig.add_trace(trace)

    fig.update_layout(title=f'{yvar} vs {xvar} for cycle {cycle}')
    return fig

#----------------------------- BATCHES CALLBACKS ------------------------------#

# Update the batches data store
@app.callback(
    Output('batches-data-store', 'data'),
    Output('batch-cycle-y', 'options'),
    Output('batch-cycle-color', 'options'),
    Output('batch-cycle-style', 'options'),
    Output('batch-correlation-color', 'options'),
    Input('batches-dropdown', 'value'),
    Input('batches-data-store', 'data'),
)
def update_batch_data(batches, data):
    # Remove batches no longer in dropdown
    for batch in list(data['data_batch_cycle'].keys()):
        if batch not in batches:
            data['data_batch_cycle'].pop(batch)

    data_folder = config['Batches folder path']

    for batch in batches:
        if batch in data['data_batch_cycle'].keys():
            continue
        file_location = os.path.join(data_folder, batch)
        files = os.listdir(file_location)
        try:
            analysed_file = next(f for f in files if (f.startswith('batch') and f.endswith('.json')))
        except StopIteration:
            continue
        with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data['data_batch_cycle'][batch] = json_data["data"]
    
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data_list

    data['data_all_samples'] = data_list
    
    # Update the y-axis options
    y_vars = set(['Normalised discharge capacity (%)'])
    for data_dict in data_list:
        y_vars.update([k for k,v in data_dict.items() if isinstance(v,list)])
    y_vars = list(y_vars)

    color_vars = set(['Max voltage (V)', 'Actual N:P ratio', '1/Formation C', 'Electrolyte name'])
    for data_dict in data['data_all_samples']:
        color_vars.update([k for k,v in data_dict.items() if not isinstance(v,list)])
    color_vars = list(color_vars)

    return data, y_vars, color_vars, color_vars, color_vars


# Update the batch cycle graph
@app.callback(
    Output('batch-cycle-graph', 'figure'),
    Output('batch-cycle-y', 'value'),
    Input('batches-data-store', 'data'),
    Input('batch-cycle-y', 'value'),
    Input('batch-cycle-color', 'value'),
    Input('batch-cycle-colormap', 'value'),
    Input('batch-cycle-style', 'value'),
)
def update_batch_cycle_graph(data, variable, color, colormap, style):
    fig = px.scatter().update_layout(title='No data...', xaxis_title='Cycle', yaxis_title='')
    fig.update_layout(template = graph_template)
    if not data or not data['data_batch_cycle']:
        return fig, variable

    # data['data_batch_cycle'] is a dict with keys as batch names and values as dicts
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data_list
    df = pd.concat(pd.DataFrame(d) for d in data_list)

    if df.empty:
        return fig, variable
    
    # Use Plotly Express to create the scatter plot
    # TODO copy the stuff from other plots here
    if 'Formation C' in df.columns:
        df['1/Formation C'] = 1 / df['Formation C']
    
    if not variable:
        if 'Specific discharge capacity (mAh/g)' in df.columns:
            variable = 'Specific discharge capacity (mAh/g)'
        elif 'Discharge capacity (mAh)' in df.columns:
            variable = 'Discharge capacity (mAh)'
    fig.update_layout(
        title=f'{variable} vs cycle',
    )

    fig = px.scatter(
        df,
        x='Cycle',
        y=variable,
        color=color,
        color_continuous_scale=colormap,
        symbol=style,
        hover_name='Sample ID'
    )
    fig.update_coloraxes(colorbar_title_side='right')

    return fig, variable

# Update the correlation map
@app.callback(
    Output('batch-correlation-map', 'figure'),
    Output('batch-correlation-x', 'options'),
    Output('batch-correlation-y', 'options'),
    Input('batches-data-store', 'data'),
)
def update_correlation_map(data):
    # data is a list of dicts
    if not data['data_batch_cycle']:
        fig = px.imshow([[0]], color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(template = graph_template)
        return fig, [], []
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data = [{k: v for k, v in d.items() if not isinstance(v, list) and v} for d in data_list]
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    if not dfs:
        fig = px.imshow([[0]], color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(template = graph_template)
        return fig, [], []
    df = pd.concat(dfs, ignore_index=True)

    if 'Formation C' in df.columns:
        df['1/Formation C'] = 1 / df['Formation C']

    # remove columns where all values are the same
    df = df.loc[:, df.nunique() > 1]

    # remove other unnecessary columns
    columns_not_needed = [
        'Sample ID',
        'Last efficiency (%)',
        'Last specific discharge capacity (mAh/g)',
        'Capacity loss (%)',
    ]
    df = df.drop(columns=columns_not_needed)

    # sort columns reverse alphabetically
    df = df.reindex(sorted(df.columns), axis=1)

    def customwrap(s,width=30):
        return "<br>".join(textwrap.wrap(s,width=width))

    options = df.columns
    df.columns = [customwrap(col) for col in df.columns]

    # Calculate the correlation matrix
    corr = correlation_matrix(df)

    # Use Plotly Express to create the heatmap
    fig = px.imshow(corr, color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)

    fig.update_layout(
        coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=0, b=0),
        template = graph_template,
    )

    return fig, options, options

# On clicking the correlation map, update the X-axis and Y-axis dropdowns
@app.callback(
    Output('batch-correlation-x', 'value'),
    Output('batch-correlation-y', 'value'),
    Input('batch-correlation-map', 'clickData'),
)
def update_correlation_vars(clickData):
    if not clickData:
        return no_update
    point = clickData['points'][0]
    xvar = point['x'].replace('<br>', ' ')
    yvar = point['y'].replace('<br>', ' ')
    return xvar, yvar

# On changing x and y axes, update the correlation graph
@app.callback(
    Output('batch-correlation-graph', 'figure'),
    Input('batches-data-store', 'data'),
    Input('batch-correlation-x', 'value'),
    Input('batch-correlation-y', 'value'),
    Input('batch-correlation-color', 'value'),
    Input('batch-correlation-colorscale', 'value'),
)
def update_correlation_graph(data, xvar, yvar, color, colormap):
    if not xvar or not yvar:
        fig = px.scatter().update_layout(xaxis_title='X-axis Title', yaxis_title='Y-axis Title')
        fig.update_layout(template = graph_template)
        return fig
    data_list = []
    for key, value in data['data_batch_cycle'].items():
        data_list += value
    data = [{k: v for k, v in d.items() if not isinstance(v, list) and v} for d in data_list]
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    if not dfs:
        fig = px.scatter().update_layout(xaxis_title='X-axis Title', yaxis_title='Y-axis Title')
        fig.update_layout(template = graph_template)
        return fig
    df = pd.concat(dfs, ignore_index=True)
    if 'Formation C' in df.columns:
        df['1/Formation C'] = 1 / df['Formation C']

    hover_columns = [
        'Max voltage (V)',
        'Anode type',
        'Cathode type',
        'Anode active material mass (mg)',
        'Cathode active material mass (mg)',
        'Actual N:P ratio',
        # 'Electrolyte name',
        'Electrolyte description'
        # 'Electrolyte amount (uL)',
        'First formation efficiency (%)',
        'First formation specific discharge capacity (mAh/g)',
        # 'Initial specific discharge capacity (mAh/g)',
        # 'Initial efficiency (%)',
    ]
    # remove columns which are not in the data
    hover_columns = [col for col in hover_columns if col in df.columns]
    hover_data = {col: True for col in hover_columns}

    fig = px.scatter(
        df,
        x=xvar,
        y=yvar,
        color=color,
        color_continuous_scale=colormap,
        custom_data=df[hover_columns],
        hover_name="Sample ID",
        hover_data=hover_data,
    )
    fig.update_traces(
        marker=dict(size=10,line=dict(color='black', width=1)),
    )
    fig.update_coloraxes(colorbar_title_side='right')
    fig.update_layout(
        xaxis_title=xvar,
        yaxis_title=yvar,
        template = graph_template,
    )
    return fig

if __name__ == '__main__':
    def start_dash_server():
        app.run_server(debug=True, use_reloader=False)
    def start_web_browser():
        webbrowser.open_new("http://localhost:8050")
    Thread(target=start_web_browser).start()
    start_dash_server()
