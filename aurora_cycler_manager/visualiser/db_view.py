""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Database view tab layout and callbacks for the visualiser app.
"""
import json
from datetime import datetime
import base64
import paramiko
from dash import Dash, dcc, html, Input, Output, State, callback_context as ctx, no_update, ALL
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from aurora_cycler_manager.visualiser.funcs import get_sample_names, get_database
from aurora_cycler_manager.server_manager import ServerManager

# Server manager
# If user cannot ssh connect then disable features that require it
try:
    sm = ServerManager()
    print("Successfully connected to the servers. You have permissions to alter everything.")
    permissions = True
except (paramiko.SSHException, FileNotFoundError):
    print("You do not have permission to write to the servers. Disabling these features.")
    sm = None
    permissions = False


#-------------------------------------- Database view layout --------------------------------------#
def db_view_layout(config: dict) -> html.Div:
    # Get the database data when loading layout
    db_data = get_database(config)
    
    # Layout
    return html.Div(
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
                            dbc.Tabs(
                                [
                                    dbc.Tab(label = 'Pipelines', tab_id = 'pipelines', active_label_style={"background-color": "#F7F8F8"}, activeTabClassName="fw-bold"),
                                    dbc.Tab(label = 'Samples', tab_id = 'samples', active_label_style={"background-color": "#F7F8F8"}, activeTabClassName="fw-bold"),
                                    dbc.Tab(label = 'Jobs', tab_id = 'jobs', active_label_style={"background-color": "#F7F8F8"}, activeTabClassName="fw-bold"),
                                    dbc.Tab(label = 'Results', tab_id = 'results', active_label_style={"background-color": "#F7F8F8"}, activeTabClassName="fw-bold"),
                                ],
                                id='table-select',
                                active_tab='pipelines',
                            ),
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
                            dbc.Button("View data", id='view-button', color='primary', outline=True, className='me-1'),
                            dbc.Button("Snapshot", id='snapshot-button', color='primary', outline=True, className='me-1'),
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
                                            {'label': name, 'value': name} for name in get_sample_names(config)
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
                                        "Auto-increment", id="load-incrememt", className="ms-auto", color='light', n_clicks=0
                                    ),
                                    dbc.Button(
                                        "Clear all", id="load-clear", className="ms-auto", color='light', n_clicks=0
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
                            dbc.ModalBody(
                                id='ready-modal-body',
                                children="""
                                    Are you sure you want ready the selected pipelines?
                                    You must force update the database afterwards to check if tomato has started the job(s).
                                """
                            ),
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
                                            dcc.Input(id='submit-capacity', type='number', min=0, max=10),
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
                    # Snapshot
                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("Snapshot")),
                            dbc.ModalBody(
                                id='snapshot-modal-body',
                                children="""
                                    Do you want to snapshot the selected samples?
                                    This could take minutes per sample depending on data size.
                                """
                            ),
                            dbc.ModalFooter(
                                [
                                    dbc.Button(
                                        "Snapshot", id="snapshot-yes-close", className="ms-auto", n_clicks=0, color='warning'
                                    ),
                                    dbc.Button(
                                        "Go back", id="snapshot-no-close", className="ms-auto", n_clicks=0, color='secondary'
                                    ),
                                ]
                            ),
                        ],
                        id="snapshot-modal",
                        is_open=False,
                    ),
                ]
            )
        ]
    )

#------------------------------------- Database view callbacks ------------------------------------#

def register_db_view_callbacks(app: Dash, config: dict) -> None:
    """ Register callbacks for the database view layout. """

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
        Output('view-button', 'style'),
        Output('snapshot-button', 'style'),
        Input('table-select', 'active_tab'),
        Input('table-data-store', 'data'),
    )
    def update_table(table, data):
        load = {'display': 'none'}
        eject = {'display': 'none'}
        ready = {'display': 'none'}
        unready = {'display': 'none'}
        cancel = {'display': 'none'}
        submit = {'display': 'none'}
        view = {'display': 'none'}
        snapshot = {'display': 'none'}
        if table == 'pipelines':
            load = {'display': 'inline-block'}
            eject = {'display': 'inline-block'}
            ready = {'display': 'inline-block'}
            unready = {'display': 'inline-block'}
            cancel = {'display': 'inline-block'}
            submit = {'display': 'inline-block'}
            view = {'display': 'inline-block'}
            snapshot = {'display': 'inline-block'}
        elif table == 'jobs':
            cancel = {'display': 'inline-block'}
        elif table == 'samples' or table == 'results':
            view = {'display': 'inline-block'}
            snapshot = {'display': 'inline-block'}
        return data['data'][table], data['column_defs'][table], load, eject, ready, unready, submit, cancel, view, snapshot

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
        db_data = get_database(config)
        dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_checked = db_data['data']['pipelines'][0]['Last checked']
        return db_data, f"Last refreshed: {dt_string}", f"Last updated: {last_checked}"

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
        Output('view-button', 'disabled'),
        Output('snapshot-button', 'disabled'),
        Input('table', 'selectedRows'),
        State('table-select', 'active_tab'),
    )
    def enable_buttons(selected_rows, table):
        load, eject, ready, unready, submit, cancel, view, snapshot = True,True,True,True,True,True,True,True
        if selected_rows:  # Must have something selected
            if permissions:  # Must have permissions to do anything except view
                if table == 'pipelines':
                    if all([s['Sample ID'] is not None for s in selected_rows]):
                        submit, snapshot = False, False
                        if all([s['Job ID'] is None for s in selected_rows]):
                            eject, ready, unready = False, False, False
                        elif all([s['Job ID'] is not None for s in selected_rows]):
                            cancel = False
                    elif all([s['Sample ID'] is None for s in selected_rows]):
                        load = False
                elif table == 'jobs':
                    if all([s['Status'] in ['r','q','qw'] for s in selected_rows]):
                        cancel = False
                elif table == 'results' or table == 'samples':
                    if all([s['Sample ID'] is not None for s in selected_rows]):
                        snapshot = False
            if any([s['Sample ID'] is not None for s in selected_rows]):
                view = False
        return load, eject, ready, unready, submit, cancel, view, snapshot

    # Eject button pop up
    @app.callback(
        Output("eject-modal", "is_open"),
        Input('eject-button', 'n_clicks'),
        Input('eject-yes-close', 'n_clicks'),
        Input('eject-no-close', 'n_clicks'),
        State('eject-modal', 'is_open'),
    )
    def eject_sample_button(eject_clicks, yes_clicks, no_clicks, is_open):
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
        Output("load-incrememt", "style"),
        Input('load-button', 'n_clicks'),
        Input('load-yes-close', 'n_clicks'),
        Input('load-no-close', 'n_clicks'),
        State('load-modal', 'is_open'),
        State('table', 'selectedRows'),
        State('table-data-store', 'data'),
    )
    def load_sample_button(load_clicks, yes_clicks, no_clicks, is_open, selected_rows, db_data):
        if not selected_rows or not ctx.triggered:
            return is_open, no_update, no_update
        possible_samples = [s.get('Sample ID', None) for s in db_data['data']['samples']]
        options = [{'label': s, 'value': s} for s in possible_samples if s]
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
        if len(selected_rows) > 1:
            increment = {'display': 'inline-block'}
        else:
            increment = {'display': 'none'}
        if button_id == 'load-button':
            return not is_open, children, increment
        elif button_id == 'load-yes-close' and yes_clicks:
            return False, no_update, increment
        elif button_id == 'load-no-close' and no_clicks:
            return False, no_update, increment
        return is_open, no_update, increment
    
    # When auto-increment is pressed, increment the sample ID for each selected pipeline
    @app.callback(
        Output({'type':'load-dropdown','index':ALL}, 'value'),
        Input('load-incrememt', 'n_clicks'),
        Input('load-clear', 'n_clicks'),
        State({'type':'load-dropdown','index':ALL}, 'value'),
        State('table-data-store', 'data'),
    )
    def update_load_selection(inc_clicks, clear_clicks, selected_samples, db_data):
        if not ctx.triggered:
            return selected_samples
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # If clear, clear all selected samples
        if button_id == 'load-clear':
            return [[] for _ in selected_samples]
        
        # If auto-increment, go through the list, if the sample is empty increment the previous sample
        if button_id == 'load-incrememt':
            possible_samples = [s.get('Sample ID', None) for s in db_data['data']['samples']]
            for i in range(1,len(selected_samples)):
                if not selected_samples[i]:
                    prev_sample = selected_samples[i-1]
                    if prev_sample:
                        prev_sample_number = prev_sample.split('_')[-1]
                        #convert to int, increment, convert back to string with same padding
                        new_sample_number = str(int(prev_sample_number)+1).zfill(len(prev_sample_number))
                        new_sample = '_'.join(prev_sample.split('_')[:-1]) + '_' + new_sample_number
                        if new_sample in possible_samples:
                            selected_samples[i] = new_sample
        return selected_samples

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
        Output('refresh-database', 'n_clicks', allow_duplicate=True),
        Input('ready-yes-close', 'n_clicks'),
        State('table', 'selectedRows'),
        prevent_initial_call=True,
    )
    def ready_pipeline(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            print(f"Readying {row['Pipeline']}")
            output = sm.ready(row['Pipeline'])
        return no_update, 1

    # Unready button pop up
    @app.callback(
        Output("unready-modal", "is_open"),
        Input('unready-button', 'n_clicks'),
        Input('unready-yes-close', 'n_clicks'),
        Input('unready-no-close', 'n_clicks'),
        State('unready-modal', 'is_open'),
    )
    def unready_pipeline_button(unready_clicks, yes_clicks, no_clicks, is_open):
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
        Output('refresh-database', 'n_clicks', allow_duplicate=True),
        Input('unready-yes-close', 'n_clicks'),
        State('table', 'selectedRows'),
        prevent_initial_call=True,
    )
    def unready_pipeline(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            print(f"Unreadying {row['Pipeline']}")
            output = sm.unready(row['Pipeline'])
        return no_update, 1

    # Submit button pop up
    @app.callback(
        Output("submit-modal", "is_open"),
        Input('submit-button', 'n_clicks'),
        Input('submit-yes-close', 'n_clicks'),
        Input('submit-no-close', 'n_clicks'),
        State('submit-modal', 'is_open'),
    )
    def submit_pipeline_button(submit_clicks, yes_clicks, no_clicks, is_open):
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
        Output('loading-database', 'children', allow_duplicate=True),
        Output('refresh-database', 'n_clicks', allow_duplicate=True),
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

    # View data
    @app.callback(
        Output('tabs', 'value'),
        Output('samples-dropdown', 'value'),
        Input('view-button', 'n_clicks'),
        State('table', 'selectedRows'),
        prevent_initial_call=True,
    )
    def view_data(n_clicks, selected_rows):
        if not n_clicks or not selected_rows:
            return no_update, no_update
        sample_id = [s['Sample ID'] for s in selected_rows]
        return 'tab-1', sample_id

    # Snapshot button pop up
    @app.callback(
        Output("snapshot-modal", "is_open"),
        Input('snapshot-button', 'n_clicks'),
        Input('snapshot-yes-close', 'n_clicks'),
        Input('snapshot-no-close', 'n_clicks'),
        State('snapshot-modal', 'is_open'),
    )
    def snapshot_sample_button(snapshot_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'snapshot-button':
            return not is_open
        elif button_id == 'snapshot-yes-close' and yes_clicks:
            return False
        elif button_id == 'snapshot-no-close' and no_clicks:
            return False
        return is_open, no_update, no_update, no_update
    # When snapshot confirmed, snapshot the samples and refresh the database
    @app.callback(
        Output('loading-database', 'children', allow_duplicate=True),
        Input('snapshot-yes-close', 'n_clicks'),
        State('table', 'selectedRows'),
        prevent_initial_call=True,
    )
    def snapshot_sample(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update
        for row in selected_rows:
            print(f"Snapshotting {row['Sample ID']}")
            sm.snapshot(row['Sample ID'])
        return no_update
