import os
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import textwrap
import yaml
import numpy as np
import json
import sqlite3
import pandas as pd

app = dash.Dash(__name__)

#======================================================================================================================#
#===================================================== FUNCTIONS ======================================================#
#======================================================================================================================#

def get_sample_names() -> list:
    db_path = "K:/Aurora/cucumber/database/database.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM results")
        samples = cursor.fetchall()
    return [sample[0] for sample in samples]

def get_batch_names() -> list:
    graph_config_path = "K:/Aurora/cucumber/graph_config.yml"
    with open(graph_config_path, 'r') as f:
        graph_config = yaml.safe_load(f)
    return list(graph_config.keys())

#======================================================================================================================#
#======================================================= LAYOUT =======================================================#
#======================================================================================================================#

colorscales = px.colors.named_colorscales()

app.layout = html.Div([
    dcc.Store(id='samples-data-store', data={'data_top': [], 'data_bottom': []}),
    dcc.Store(id='batches-data-store', data={'data_top': [], 'data_bottom': []}),
    html.Div(
        [
            dcc.Tabs(id="tabs", value='tab-1', children=[
                #################### SAMPLES TAB ####################
                dcc.Tab(
                    label='Samples',
                    value='tab-1',
                    children=[
                        html.Div(
                            [
                                html.P("Select samples to plot:"),
                                dcc.Dropdown(
                                    id='samples-dropdown',
                                    options=[
                                        {'label': name, 'value': name} for name in get_sample_names()
                                    ],
                                    value=[],
                                    multi=True,
                                ),
                                html.Div(style={'margin-top': '20px'}),
                                html.P("Time graph"),
                                html.Label('X-axis', htmlFor='samples-time-y'),
                                dcc.Dropdown(
                                    id='samples-time-x',
                                    options=['uts','From protection','From formation','From cycling'],
                                    value='uts',
                                    multi=False,
                                ),
                                html.Label('Y-axis', htmlFor='samples-time-y'),
                                dcc.Dropdown(
                                    id='samples-time-y',
                                    options=['Ewe'],
                                    value='Ewe',
                                    multi=False,
                                ),
                                html.P("Cycles graph"),
                                html.P("X-axis: Cycle"),
                                html.Label('Y-axis', htmlFor='samples-cycle-y'),
                                dcc.Dropdown(
                                    id='samples-cycle-y',
                                    options=[
                                        'Specific discharge capacity (mAh/g)',
                                        'Normalised discharge capacity (%)',
                                        'Efficiency (%)',
                                    ],
                                    value='Specific discharge capacity (mAh/g)',
                                    multi=False,
                                ),
                            ],
                        style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'horizontalAlign': 'left', 'height': '90vh'}
                        ),
                        html.Div(
                            [
                                dcc.Graph(id='time-graph',figure={'data': [],'layout': go.Layout(title='vs time',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'})}, config={'scrollZoom':True, 'displaylogo':False}, style={'height': '45vh'}),
                                dcc.Graph(id='cycles-graph',figure={'data': [],'layout': go.Layout(title='vs cycle',xaxis={'title': 'X-axis Title'},yaxis={'title': 'Y-axis Title'})}, config={'scrollZoom':True, 'displaylogo':False}, style={'height': '45vh'}),
                            ],
                        style={'width': '75%', 'display': 'inline-block', 'paddingLeft': '20px', 'horizontalAlign': 'right', 'height': '90vh'}
                        ),
                    ],
                ),
                #################### BATCHES TAB ####################
                dcc.Tab(
                    label='Batches',
                    value='tab-2',
                    children=[
                        html.Div(
                            [
                                html.P("Select batches to plot:"),
                                dcc.Dropdown(
                                    id='batches-dropdown',
                                    options=[
                                        {'label': name, 'value': name} for name in get_batch_names()
                                    ],
                                    value=[],
                                    multi=True,
                                ),
                                html.Div(style={'margin-top': '20px'}),
                                html.P("Select variable for top graph"),
                                dcc.Dropdown(
                                    id='batch-cycle-y',
                                    options=['Normalised discharge capacity (%)'],
                                    value='Normalised discharge capacity (%)',
                                    multi=False,
                                ),
                                html.P("Colormap of top graph"),
                                dcc.Dropdown(
                                    id='batch-cycle-color',
                                    options=[
                                        'Max voltage (V)',
                                        'Actual N:P ratio',
                                        '1/Formation C',
                                        'Electrolyte name',
                                    ],
                                    value='1/Formation C',
                                    multi=False,
                                ),
                                dcc.Dropdown(
                                    id='batch-cycle-colormap',
                                    options=colorscales,
                                    value='viridis'
                                )
                            ],
                        style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'horizontalAlign': 'left', 'height': '40vh'}
                        ),
                        html.Div(
                            [
                                # Top graph
                                dcc.Graph(
                                    id='batch-cycle-graph',
                                    figure={
                                        'data': [],
                                        'layout': go.Layout(title='vs cycle', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                    }, 
                                    config={'scrollZoom': True, 'displaylogo':False}
                                ),
                            ],
                        style={'width': '75%', 'display': 'inline-block', 'paddingLeft': '20px', 'horizontalAlign': 'right', 'height': '40vh'}
                        ),
                        # Div for two graphs side by side
                        html.Div(
                            [
                                # First graph on the left
                                html.Div(
                                    dcc.Graph(
                                        id='batch-correlation-map',
                                        figure={
                                            'data': [],
                                            'layout': go.Layout(title='heatmap', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                        }, 
                                        config={'scrollZoom': False, 'displayModeBar': False},
                                        style={'height': '50vh'},
                                    ),
                                    style={'width': '50%', 'display': 'inline-block', 'height': '45vh'}
                                ),
                                # Second graph on the right
                                html.Div(
                                    [
                                        html.Div(
                                            html.P("Choose a colormap"),
                                            style={'width': '25%', 'display': 'inline-block', 'paddingLeft': '50px', 'verticalAlign': 'middle'},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id='batch-correlation-color',
                                                    options=[
                                                        'Max voltage (V)',
                                                        'Actual N:P ratio',
                                                        '1/Formation C',
                                                        'Electrolyte name',
                                                    ],
                                                    value='Actual N:P ratio',
                                                    multi=False,
                                                ),
                                            ],
                                            style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Dropdown(
                                                    id='batch-correlation-colorscale',
                                                    options=colorscales,
                                                    value='viridis',
                                                    multi=False,
                                                ),
                                            ],
                                            style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'middle'},
                                        ),
                                        dcc.Graph(
                                            id='batch-correlation-graph',
                                            figure={
                                                'data': [],
                                                'layout': go.Layout(title='params', xaxis={'title': 'X-axis Title'}, yaxis={'title': 'Y-axis Title'})
                                            },
                                            config={'scrollZoom': True, 'displaylogo':False},
                                            style={'height': '40vh'},
                                        ),
                                    ],
                                    style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}
                                ),
                            ],
                            style={'height': '45vh'}
                        ),
                    ],
                ),
            ]),
        ],
        style={'height': '100vh'},
    ),
])

#======================================================================================================================#
#===================================================== CALLBACKS ======================================================#
#======================================================================================================================#

#----------------------------- SAMPLES CALLBACKS ------------------------------#

# Update the samples data store
@app.callback(
    Output('samples-data-store', 'data'),
    Input('samples-dropdown', 'value'),
)
def update_sample_data(samples):
    data_sample_time = {}
    data_sample_cycle = {}
    for sample in samples:
        # Get the raw file
        run_id = sample.rsplit('_',1)[0]
        data_folder = "K:/Aurora/cucumber/snapshots" # hardcoded for now
        file_location = os.path.join(data_folder,run_id,sample)

        # Get raw data
        files = os.listdir(file_location)
        cycling_files = [f for f in files if (f.startswith('snapshot') and f.endswith('.h5'))]
        if not cycling_files:
            print(f"No cycling files found in {file_location}")
            continue
        dfs = [pd.read_hdf(f'{file_location}/{f}') for f in cycling_files]
        dfs = [df for df in dfs if 'uts' in df.columns and df['uts'].nunique() > 1]
        dfs.sort(key=lambda df: df['uts'].iloc[0])
        df = pd.concat(dfs)
        data_sample_time[sample] = df.to_dict(orient='list')

        # Get the analysed file
        try:
            analysed_file = next(f for f in files if (f.startswith('cycles') and f.endswith('.json')))
        except StopIteration:
            continue
        with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
            cycle_dict = json.load(f)
        if not cycle_dict or 'Cycle' not in cycle_dict.keys():
            continue
        data_sample_cycle[sample] = cycle_dict
    return {'data_sample_time': data_sample_time, 'data_sample_cycle': data_sample_cycle}


# Update the time graph
@app.callback(
    Output('time-graph', 'figure'),
    Input('samples-data-store', 'data'),
    Input('samples-time-x', 'value'),
    Input('samples-time-y', 'value'),
)
def update_time_graph(data, xvar, yvar):
    if not data['data_sample_time']:
        return {'data': [], 'layout': go.Layout(title=f'No data...', xaxis={'title': 'Time (s)'},)}
    traces = []
    for sample, data_dict in data['data_sample_time'].items():
        uts = np.array(data_dict['uts'])
        if xvar == 'From protection':
            offset=uts[0]
        if xvar == 'From formation':
            offset=uts[0]
        else:
            offset=0
        traces.append(go.Scatter(x=np.array(data_dict['uts'])-offset, y=data_dict[yvar], mode='lines', name=sample))
    return {'data': traces, 'layout': go.Layout(title=f'{yvar} vs time', xaxis={'title': 'Time (s)'}, yaxis={'title': yvar})}


# Update the cycles graph
@app.callback(
    Output('cycles-graph', 'figure'),
    Input('samples-data-store', 'data'),
    Input('samples-cycle-y', 'value'),
)
def update_cycles_graph(data, yvar):
    traces = []
    if not data['data_sample_cycle']:
        return {'data': traces, 'layout': go.Layout(title=f'No data...', xaxis={'title': 'Cycle'}, yaxis={'title': yvar})}
    for sample, cycle_dict in data['data_sample_cycle'].items():
        traces.append(go.Scatter(x=cycle_dict['Cycle'], y=cycle_dict[yvar], mode='lines+markers', name=sample))
    return {'data': traces, 'layout': go.Layout(title=f'{yvar} vs cycle', xaxis={'title': 'Cycle'}, yaxis={'title': yvar})}

#----------------------------- BATCHES CALLBACKS ------------------------------#

# Update the batches data store
@app.callback(
    Output('batches-data-store', 'data'),
    Input('batches-dropdown', 'value'),
)
def update_batch_data(batches):
    data_folder = "K:/Aurora/cucumber/batches" # Hardcoded for now
    data = []
    for batch in batches:
        file_location = os.path.join(data_folder, batch)
        files = os.listdir(file_location)
        try:
            analysed_file = next(f for f in files if (f.startswith('batch') and f.endswith('.json')))
        except StopIteration:
            continue
        with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data += json_data
    return {'data_batch_cycle': data}


# Update the batch cycle graph
@app.callback(
    Output('batch-cycle-graph', 'figure'),
    Input('batches-data-store', 'data'),
    Input('batch-cycle-y', 'value'),
    Input('batch-cycle-color', 'value'),
    Input('batch-cycle-colormap', 'value'),
)
def update_batch_cycle_graph(data, variable, color, colormap):
    if not data['data_batch_cycle']:
        fig = px.scatter().update_layout(title=f'{variable} vs cycle', xaxis_title='Cycle', yaxis_title=variable)
        fig.update_layout(template = 'ggplot2')
        return fig

    df = pd.concat([pd.DataFrame(sample) for sample in data['data_batch_cycle']])
    
    if df.empty:
        return px.scatter().update_layout(title=f'{variable} vs cycle', xaxis_title='Cycle', yaxis_title=variable)
    
    # Use Plotly Express to create the scatter plot
    # TODO copy the stuff from other plots here
    df['1/Formation C'] = 1 / df['Formation C']

    fig = px.scatter(df, x='Cycle', y=variable, title=f'{variable} vs cycle', color=color, color_continuous_scale=colormap)
    fig.update_layout(scattermode="group", scattergap=0.75, template = 'ggplot2')
    # Plotly Express returns a complete figure, so you can directly return it
    return fig

# Update the correlation map
@app.callback(
    Output('batch-correlation-map', 'figure'),
    # Output('batch-correlation-vars', 'options'),
    Input('batches-data-store', 'data'),
)
def update_correlation_map(data):
    # data is a list of dicts
    if not data['data_batch_cycle']:
        fig = px.imshow([[0]], color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)
        fig.update_layout(template = 'ggplot2')
        return fig
    data = [{k: v for k, v in d.items() if not isinstance(v, list) and v} for d in data['data_batch_cycle']]
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    df = pd.concat(dfs, ignore_index=True)
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

    # TEMPORARY: remove anything non-numeric
    df = df.select_dtypes(include=[np.number])

    def customwrap(s,width=30):
        return "<br>".join(textwrap.wrap(s,width=width))

    df.columns = [customwrap(col) for col in df.columns]

    # Calculate the correlation matrix
    corr = df.corr()

    # Use Plotly Express to create the heatmap
    fig = px.imshow(corr, color_continuous_scale='balance', aspect="auto", zmin=-1, zmax=1)

    fig.update_layout(
        coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1']),
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=0, b=0),
        template = 'ggplot2',
    )
    return fig

@app.callback(
    Output('batch-correlation-graph', 'figure'),
    Input('batch-correlation-map', 'clickData'),
    Input('batches-data-store', 'data'),
    Input('batch-correlation-color', 'value'),
    Input('batch-correlation-colorscale', 'value'),
)
def update_correlation_graph(clickData, data, color, colormap):
    if not clickData:
        fig = px.scatter().update_layout(xaxis_title='X-axis Title', yaxis_title='Y-axis Title')
        fig.update_layout(template = 'ggplot2')
        return fig
    # clickData is a dict with keys 'points' and 'event'
    # 'points' is a list of dicts with keys 'curveNumber', 'pointNumber', 'pointIndex', 'x', 'y', 'text'

    data = [{k: v for k, v in d.items() if not isinstance(v, list) and v} for d in data['data_batch_cycle']]
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    df = pd.concat(dfs, ignore_index=True)

    df['1/Formation C'] = 1 / df['Formation C']

    point = clickData['points'][0]
    xvar = point['x']
    yvar = point['y']

    xvar = xvar.replace('<br>', ' ')
    yvar = yvar.replace('<br>', ' ')

    hover_columns = [
        'Sample ID',
        'Formation C',
    ]

    fig = px.scatter(
        df,
        x=xvar,
        y=yvar,
        color=color,
        color_continuous_scale=colormap,
        custom_data=df[hover_columns],
        hover_name="Sample ID",
        hover_data={
            'Formation C': True,
        },
    )
    fig.update_traces(
        marker=dict(size=10)
    )
    fig.update_layout(
        xaxis_title=xvar,
        yaxis_title=yvar,
        template = 'ggplot2',
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)