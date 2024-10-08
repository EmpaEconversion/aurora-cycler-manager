import os
import json
import pandas as pd
import textwrap
from dash import Dash, dcc, html, Input, Output, no_update
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
import plotly.graph_objs as go
import plotly.express as px
from aurora_cycler_manager.visualiser.funcs import get_batch_names, correlation_matrix

graph_template = 'seaborn'
colorscales = px.colors.named_colorscales()

def batches_menu(config: dict) -> html.Div:
    return html.Div(
        style = {'overflow': 'scroll', 'height': '100%'},
        children = [
            html.H5("Select batches to plot:"),
            dcc.Dropdown(
                id='batches-dropdown',
                options=[
                    {'label': name, 'value': name} for name in get_batch_names(config)
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

def batches_layout(config: dict) -> html.Div:
    return html.Div(
        style={'height': '100%'}, 
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
            PanelGroup(
                id="batches-panel-group",
                children=[
                    Panel(
                        id="batches-menu",
                        className="menu-panel",
                        children=batches_menu(config),
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
    )

#----------------------------- BATCHES CALLBACKS ------------------------------#

def register_batches_callbacks(app: Dash, config: dict) -> None:
    """ Register all callbacks for the batches tab. """
    
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