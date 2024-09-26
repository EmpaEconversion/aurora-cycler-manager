""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Web-based visualiser for the Aurora cycler manager based on Dash and Plotly.

Allows users to rapidly view and compare data from the Aurora robot and cycler
systems, both of individual samples and of batches of samples.
"""

import os
import sys
import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table
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
import pandas as pd
from datetime import datetime
from scipy import stats
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.analysis import combine_jobs, _run_from_sample

app = dash.Dash(__name__)
app.title = "Aurora Visualiser"

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

samples_menu = html.Div(
    style = {'overflow': 'scroll', 'height': '100%'},
    children = [
        html.H4("Select samples to plot:"),
        dcc.Dropdown(
            id='samples-dropdown',
            options=[
                {'label': name, 'value': name} for name in get_sample_names()
            ],
            value=[],
            multi=True,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H4("Time graph"),
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
        html.H4("Cycles graph"),
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
        html.H4("One cycle graph"),
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

batches_menu = html.Div(
    style = {'overflow': 'scroll', 'height': '100%'},
    children = [
        html.H4("Select batches to plot:"),
        dcc.Dropdown(
            id='batches-dropdown',
            options=[
                {'label': name, 'value': name} for name in get_batch_names()
            ],
            value=[],
            multi=True,
        ),
        html.Div(style={'margin-top': '50px'}),
        html.H4("Cycles graph"),
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
        html.H4("Correlation graph"),
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
                                            defaultSizePercentage=20,
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
                                                                                dcc.Graph(id='cycles-graph',figure={'data': [],'layout': go.Layout(template=graph_template,title='vs cycle',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'},showlegend=False)}, config={'scrollZoom':True, 'displaylogo':False},style={'height': '100%'}), # TODO this doesn't work
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
                                        defaultSizePercentage=20,
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
                                    overlay_style={"visibility":"visible", "filter": "blur(2px)"},
                                    style={'height': '100%'},
                                    children=[
                                        html.Div(
                                            style={'height': '100%'},
                                            children = [
                                                dcc.Store(id='table-data-store', data=get_database()),
                                                # Button to update the database
                                                html.P(children = f"Last refreshed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}", id='last-refreshed'),
                                                html.P(children = f"Last updated: unknown", id='last-updated'),
                                                html.Button("Refresh database", id='refresh-database'),
                                                # html.Button("Force update database", id='update-database'), # TODO
                                                html.Div(style={'margin-top': '10px'}),
                                                # Buttons to select which table to display
                                                dcc.RadioItems(
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
                                                dag.AgGrid(
                                                    id='table',
                                                    dashGridOptions = {"enableCellTextSelection": False, "tooltipShowDelay": 1000, 'rowSelection': 'multiple'},
                                                    style={"height": "calc(90vh - 200px)", "width": "100%", "minHeight": "400px"},
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

#======================================================================================================================#
#===================================================== CALLBACKS ======================================================#
#======================================================================================================================#

#----------------------------- DATABASE CALLBACKS ------------------------------#
@app.callback(
    Output('table', 'rowData'),
    Output('table', 'columnDefs'),
    Input('table-select', 'value'),
    Input('table-data-store', 'data'),
)
def update_table(table, data):
    return data['data'][table], data['column_defs'][table]



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

#----------------------------- SAMPLES CALLBACKS ------------------------------#

# TODO fix errors when data does not include any Formation C values

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
        data_folder = "K:/Aurora/cucumber/snapshots" # hardcoded for now
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
    # TODO fix error when data has no cycles
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

    data_folder = "K:/Aurora/cucumber/batches" # Hardcoded for now

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
    app.run_server(debug=True)
