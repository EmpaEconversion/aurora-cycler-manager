"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Database view tab layout and callbacks for the visualiser app.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import paramiko
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
from dash import callback_context as ctx
from dash_mantine_components import Notification, TextInput
from obvibe.vibing import push_exp

from aurora_cycler_manager.analysis import update_sample_metadata
from aurora_cycler_manager.config import CONFIG
from aurora_cycler_manager.database_funcs import (
    add_samples_from_object,
    delete_samples,
    get_batch_details,
    update_sample_label,
)
from aurora_cycler_manager.server_manager import ServerManager
from aurora_cycler_manager.utils import run_from_sample
from aurora_cycler_manager.visualiser.db_batch_edit import (
    batch_edit_layout,
    register_batch_edit_callbacks,
)
from aurora_cycler_manager.visualiser.funcs import (
    get_database,
    get_db_last_update,
    make_pipelines_comparable,
)
from aurora_cycler_manager.visualiser.notifications import active_time, idle_time, queue_notification

# Server manager
# If user cannot ssh connect then disable features that require it
accessible_servers = []
database_access = False
sm: ServerManager | None = None
try:
    sm = ServerManager()
    accessible_servers = [s.label for s in sm.servers]
    database_access = bool(accessible_servers)
except (paramiko.SSHException, FileNotFoundError, ValueError) as e:
    print(e)
    print("You cannot access any servers. Running in view-only mode.")
openbis_path = CONFIG.get("OpenBIS PAT")
OPENBIS_DISABLED = Path(openbis_path).exists() if openbis_path else True
#-------------------------------------- Database view layout --------------------------------------#

# Define visibility settings for buttons and divs when switching between tabs
CONTAINERS = [
    "table-container", "batch-container",
]
BUTTONS = [
    "load-button", "eject-button", "ready-button", "unready-button",
    "submit-button", "cancel-button", "view-button", "snapshot-button",
    "openbis-button", "create-batch-button", "delete-button",
    "add-samples-button", "label-button",
]
visibility_settings = {
    "batches": {
        "batch-container",
    },
    "pipelines": {
        "table-container",
        "load-button",
        "eject-button",
        "ready-button",
        "unready-button",
        "submit-button",
        "cancel-button",
        "view-button",
        "snapshot-button",
    },
    "jobs": {
        "table-container",
        "cancel-button",
        "snapshot-button",
    },
    "results": {
        "table-container",
        "view-button",
        "label-button",
    },
    "samples": {
        "table-container",
        "view-button",
        "openbis-button",
        "batch-button",
        "delete-button",
        "add-samples-button",
        "label-button",
    },
}

button_layout = html.Div(
    style={
        "display": "flex",
        "justify-content": "space-between",
        "flex-wrap": "wrap",
        "margin-top": "5px",
        "margin-bottom": "20px",
    },
    children = [
        # Left aligned buttons
        html.Div(
            style={"display": "flex", "flex-wrap": "wrap"},
            children = [
                dbc.Button(
                    [html.I(className="bi bi-clipboard me-2"), "Copy"],
                    id="copy-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-arrow-90deg-down me-2"),"Load"],
                    id="load-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-arrow-90deg-right me-2"),"Eject"],
                    id="eject-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-play me-2"),"Ready"],
                    id="ready-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-slash-circle me-2"),"Unready"],
                    id="unready-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-upload me-2"),"Submit"],
                    id="submit-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-x-circle me-2"),"Cancel"],
                    id="cancel-button", color="danger", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-graph-down me-2"),"View data"],
                    id="view-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-camera me-2"),"Snapshot"],
                    id="snapshot-button", color="primary", className="me-1",
                ),
                dcc.Upload(
                    children=dbc.Button(
                        [html.I(className="bi bi-database-add me-2"), "Add samples"],
                        id="add-samples-button", color="primary", className="me-1",
                    ),
                    id="sample-upload", accept=".json", max_size= 2*1024*1024, multiple=False,
                ),
                dbc.Tooltip(
                    children = (
                        "You do not have an OpenBIS personal access token in the config. "
                        "Need key 'OpenBIS PAT' pointing to your PAT.txt file path.",
                    ),
                    placement="top",
                    delay={"show": 1000},
                    target="openbis-button",
                    style={"whiteSpace": "pre-wrap"},
                ) if OPENBIS_DISABLED else None,
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem(
                            [html.I(className="bi bi-cloud-upload me-2"),"Automatic upload"],
                            id="openbis-auto-button", disabled=OPENBIS_DISABLED,
                        ),
                        dbc.DropdownMenuItem(
                            [html.I(className="bi bi-cloud-upload me-2"),"Custom upload"],
                            id="openbis-custom-button", disabled=OPENBIS_DISABLED,
                        ),
                    ],
                    label=html.Span([
                        html.Img(
                            src="/assets/openbis.svg",
                            style={"height": "20px", "width": "20px", "vertical-align": "middle"},
                        ),
                        " OpenBIS",
                    ]),
                    id="openbis-button", color="primary", className="bi me-1",
                    direction="up", disabled=OPENBIS_DISABLED,
                ),
                dbc.Button(
                    [html.I(className="bi bi-tag me-2"),"Add label"],
                    id="label-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-grid-3x2-gap-fill me-2"),"Create batch"],
                    id="create-batch-button", color="primary", className="me-1",
                ),
                dbc.Button(
                    [html.I(className="bi bi-trash3 me-2"),"Delete"],
                    id="delete-button", color="danger", className="me-1",
                ),
            ],
        ),
        # Right aligned buttons
        html.Div(
            style={"display": "flex", "flex-wrap": "wrap", "align-items": "center"},
            children=[
                html.Div(
                    "Loading...", id="table-info", className="me-1",
                    style={"display": "inline-block", "opacity": "0.5"},
                ),
                dbc.Button(
                    className="bi bi-arrow-clockwise me-2 large-icon",
                    id="refresh-database", color="primary",
                ),
                dbc.Button(
                    className="bi bi-database-down me-2 large-icon",
                    id="update-database", color="warning", disabled = not accessible_servers,
                ),
                dbc.Tooltip(
                    children = "Refresh database",
                    id="last-refreshed",
                    target="refresh-database",
                    style={"whiteSpace": "pre-wrap"},
                    placement="top",
                ),
                dbc.Tooltip(
                    children = "Update database from cyclers",
                    id="last-updated",
                    target="update-database",
                    style={"whiteSpace": "pre-wrap"},
                    placement="top",
                ),
            ],
        ),
    ],
)

db_view_layout = html.Div(
    style={"height": "100%", "padding": "10px"},
    children = [
        # invisible div just to make the loading spinner work when no outputs are changed
        html.Div(
            id="loading-database",
            style={"display": "none"},
        ),
        html.Div(
            style={"height": "100%", "overflow": "scroll"},
            children = [
                # Buttons to select which table to display
                dbc.Tabs(
                    [
                        dbc.Tab(label = "Samples", tab_id = "samples", activeTabClassName="fw-bold"),
                        dbc.Tab(label = "Pipelines", tab_id = "pipelines", activeTabClassName="fw-bold"),
                        dbc.Tab(label = "Jobs", tab_id = "jobs", activeTabClassName="fw-bold"),
                        dbc.Tab(label = "Results", tab_id = "results", activeTabClassName="fw-bold"),
                        dbc.Tab(label = "Batches", tab_id = "batches", activeTabClassName="fw-bold"),
                    ],
                    id="table-select",
                    active_tab="samples",
                ),
                # Main table for displaying info from database
                html.Div(
                    id="table-container",
                    children = [
                        dcc.Clipboard(id="clipboard", style={"display": "none"}),
                        dag.AgGrid(
                            id="table",
                            dashGridOptions = {
                                "enableCellTextSelection": False,
                                "ensureDomOrder": True,
                                "tooltipShowDelay": 1000,
                                "rowSelection": "multiple",
                            },
                            defaultColDef={"filter": True, "sortable": True, "floatingFilter": True},
                            style={"height": "calc(100vh - 240px)", "width": "100%", "minHeight": "300px"},
                        ),
                        button_layout,
                    ],
                ),
                batch_edit_layout,
            ],
        ),
        # Pop up modals for interacting with the database after clicking buttons
        # Eject
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Eject")),
                dbc.ModalBody(id="eject-modal-body",children="Are you sure you want eject the selected samples?"),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Eject", id="eject-yes-close", className="ms-auto", n_clicks=0, color="primary",
                        ),
                        dbc.Button(
                            "Go back", id="eject-no-close", className="ms-auto", n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="eject-modal",
            centered=True,
            is_open=False,
        ),
        # Load
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Load")),
                dbc.ModalBody(
                    id="load-modal-body",
                    children=[],
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Load", id="load-yes-close", className="ms-auto", color="primary", n_clicks=0,
                        ),
                        dbc.Button(
                            "Auto-increment", id="load-incrememt", className="ms-auto", color="light", n_clicks=0,
                        ),
                        dbc.Button(
                            "Clear all", id="load-clear", className="ms-auto", color="light", n_clicks=0,
                        ),
                        dbc.Button(
                            "Go back", id="load-no-close", className="ms-auto", color="secondary", n_clicks=0,
                        ),
                    ],
                ),
                dcc.Store(id="load-modal-store", data={}),
            ],
            id="load-modal",
            centered=True,
            is_open=False,
        ),
        # Ready
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Ready")),
                dbc.ModalBody(
                    id="ready-modal-body",
                    children="""
                        Are you sure you want ready the selected pipelines?
                        You must force update the database afterwards to check if tomato has started the job(s).
                    """,
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Ready", id="ready-yes-close", className="ms-auto", n_clicks=0, color="primary",
                        ),
                        dbc.Button(
                            "Go back", id="ready-no-close", className="ms-auto", n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="ready-modal",
            centered=True,
            is_open=False,
        ),
        # Unready
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Unready")),
                dbc.ModalBody(
                    id="unready-modal-body",
                    children="Are you sure you want un-ready the selected pipelines?",
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Unready", id="unready-yes-close", className="ms-auto", n_clicks=0, color="primary",
                        ),
                        dbc.Button(
                            "Go back", id="unready-no-close", className="ms-auto", n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="unready-modal",
            centered=True,
            is_open=False,
        ),
        # Submit
        dbc.Modal(
            [
                dcc.Store(id="payload", data={}),
                dbc.ModalHeader(dbc.ModalTitle("Submit")),
                dbc.ModalBody(
                    id="submit-modal-body",
                    style={"width": "100%"},
                    children=[
                        "Select a tomato .json payload to submit",
                        dcc.Upload(
                            id="submit-upload",
                            children=html.Div([
                                "Drag and Drop or ",
                                html.A("Select Files"),
                            ]),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "8px",
                                "textAlign": "center",
                            },
                            accept=".json",
                            multiple=False,
                        ),
                        html.P(children="No file selected", id="validator"),
                        html.Div(style={"margin-top": "10px"}),
                        html.Div([
                            html.Label("Calculate C-rate by:", htmlFor="submit-crate"),
                            dcc.Dropdown(
                                id="submit-crate",
                                options=[
                                    {"value": "areal", "label": "areal capacity x area from db"},
                                    {"value": "mass", "label": "specific capacity x mass  from db"},
                                    {"value": "nominal", "label": "nominal capacity from db"},
                                    {"value": "custom", "label": "custom capacity value"},
                                ],
                            ),
                        ]),
                        html.Div(
                            id="submit-capacity-div",
                            children=[
                                "Capacity = ",
                                dcc.Input(id="submit-capacity", type="number", min=0, max=10),
                                " mAh",
                            ],
                            style={"display": "none"},
                        ),
                    ],
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Submit", id="submit-yes-close", className="ms-auto",
                            n_clicks=0, color="primary", disabled=True,
                        ),
                        dbc.Button(
                            "Go back", id="submit-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="submit-modal",
            centered=True,
            is_open=False,
        ),
        # Cancel
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Cancel")),
                dbc.ModalBody(
                    id="cancel-modal-body",
                    children="Are you sure you want to cancel the selected jobs?",
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Cancel", id="cancel-yes-close", className="ms-auto",
                            n_clicks=0, color="danger",
                        ),
                        dbc.Button(
                            "Go back", id="cancel-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="cancel-modal",
            centered=True,
            is_open=False,
        ),
        # Snapshot
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Snapshot")),
                dbc.ModalBody(
                    id="snapshot-modal-body",
                    children="""
                        Do you want to snapshot the selected samples?
                        This could take minutes per sample depending on data size.
                    """,
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Snapshot", id="snapshot-yes-close", className="ms-auto",
                            n_clicks=0, color="warning",
                        ),
                        dbc.Button(
                            "Go back", id="snapshot-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="snapshot-modal",
            centered=True,
            is_open=False,
        ),
        # OpenBis
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("OpenBis upload")),
                dbc.ModalBody(
                    id="openbis-auto-modal-body",
                    children=[
                        "Upload the selected samples to OpenBis?",
                        html.Div(style={"margin-top": "10px"}),
                        html.P("",id="openbis-auto-sample-list",style={"whiteSpace": "pre-wrap"}),
                    ],
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Upload", id="openbis-auto-yes-close", className="ms-auto",
                            n_clicks=0, color="primary",
                        ),
                        dbc.Button(
                            "Go back", id="openbis-auto-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="openbis-auto-modal",
            centered=True,
            is_open=False,
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Custom OpenBis upload")),
                dbc.ModalBody(
                    id="openbis-custom-modal-body",
                    children=[
                        "Select the BattINFO .xlsx.",
                        html.Div(style={"margin-top": "10px"}),
                        "Fields in the template will overwrite the fields found from the database",
                        html.Div(style={"margin-top": "10px"}),
                        dcc.Upload(
                            id="openbis-template-upload",
                            children=html.Div([
                                "Drag and Drop or ",
                                html.A("Select Files"),
                            ]),
                            style={
                                "width": "100%",
                                "height": "60px",
                                "lineHeight": "60px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "8px",
                                "textAlign": "center",
                            },
                            accept=".xlsx",
                            multiple=False,
                        ),
                        html.P(children="No file selected", id="openbis-validator"),
                        html.Div(style={"margin-top": "10px"}),
                        html.P("Samples to upload:"),
                        html.Div(style={"margin-top": "10px"}),
                        html.P("",id="openbis-custom-sample-list",style={"whiteSpace": "pre-wrap"}),
                    ],
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Upload", id="openbis-custom-yes-close", className="ms-auto",
                            n_clicks=0, color="primary", disabled=True,
                        ),
                        dbc.Button(
                            "Go back", id="openbis-custom-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="openbis-custom-modal",
            centered=True,
            is_open=False,
        ),
        # Batch
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Create batch")),
                dbc.ModalBody(
                    id="batch-save-modal-body",
                    children=[
                        "Create a batch from the selected samples?",
                        html.Div(style={"margin-top": "10px"}),
                        dcc.Input(id="batch-name", type="text", placeholder="Batch name"),
                        html.Div(style={"margin-top": "10px"}),
                        dcc.Textarea(
                            id="batch-description",
                            placeholder="Batch description",
                            style={"width": "100%", "height": "200px"},
                        ),
                    ],
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Create", id="batch-save-yes-close", className="ms-auto",
                            n_clicks=0, disabled=True, color="primary",
                        ),
                        dbc.Button(
                            "Go back", id="batch-save-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="batch-save-modal",
            centered=True,
            is_open=False,
        ),
        # Delete
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Delete")),
                dbc.ModalBody(
                    id="delete-modal-body",
                    children="""
                        Are you sure you want to delete the selected samples?
                        Samples will only stay deleted if the .csv files in
                        the data folder are also deleted.
                    """,
                    ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Delete", id="delete-yes-close", className="ms-auto",
                            n_clicks=0, color="danger",
                        ),
                        dbc.Button(
                            "Go back", id="delete-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="delete-modal",
            centered=True,
            is_open=False,
        ),
        # Label
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Label samples:")),
                dbc.ModalBody(
                    id="label-modal-body",
                    children=TextInput(
                        id="label-input",
                        placeholder="This overwrites any existing label",
                        label="Label",
                    ),
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button(
                            "Add label", id="label-yes-close", className="ms-auto",
                            n_clicks=0, color="primary",
                        ),
                        dbc.Button(
                            "Go back", id="label-no-close", className="ms-auto",
                            n_clicks=0, color="secondary",
                        ),
                    ],
                ),
            ],
            id="label-modal",
            centered=True,
            is_open=False,
        ),
    ],
)

#------------------------------------- Database view callbacks ------------------------------------#

def register_db_view_callbacks(app: Dash) -> None:
    """Register callbacks for the database view layout."""
    register_batch_edit_callbacks(app, database_access)

    # Update the buttons displayed depending on the table selected
    @app.callback(
        Output("table", "rowData"),
        Output("table", "columnDefs"),
        [Output(element, "style") for element in CONTAINERS+BUTTONS],
        Input("table-select", "active_tab"),
        Input("table-data-store", "data"),
    )
    def update_table(table, data):
        settings = visibility_settings.get(table, {})
        show = {}
        hide = {"display": "none"}
        visibilities = [
            show if element in settings else hide for element in CONTAINERS+BUTTONS
        ]
        return (
            data["data"].get(table, no_update),
            data["column_defs"].get(table, no_update),
            *visibilities,
        )

    # Refresh the local data from the database
    @app.callback(
        Output("table-data-store", "data"),
        Output("last-refreshed", "children"),
        Output("last-updated", "children"),
        Output("samples-store", "data"),
        Output("batches-store", "data"),
        Input("refresh-database", "n_clicks"),
        Input("db-update-interval", "n_intervals"),
    )
    def refresh_database(n_clicks, n_intervals):
        db_data = get_database()
        dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_checked = get_db_last_update()
        samples = [s["Sample ID"] for s in db_data["data"]["samples"]]
        batches = get_batch_details()
        return (
            db_data,
            f"Refresh database\nLast refreshed: {dt_string}",
            f"Update database\nLast updated: {last_checked}",
            samples,
            batches,
        )

    # Update the database i.e. connect to servers and grab new info, then refresh the local data
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("update-database", "n_clicks"),
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
        [Output(b, "disabled") for b in BUTTONS],
        Input("table", "selectedRows"),
        State("table-select", "active_tab"),
    )
    def enable_buttons(selected_rows, table):
        enabled = {"add-samples-button"}  # These buttons are always enabled
        # Add more buttons to enabled set with union operator |=
        if selected_rows:
            enabled |= {"copy-button"}
            if accessible_servers:  # Need cycler permissions to do anything except copy, view or upload
                if table == "pipelines":
                    if all(s["Server label"] in accessible_servers for s in selected_rows):
                        if all(s["Sample ID"] is not None for s in selected_rows):
                            enabled |= {"submit-button","snapshot-button"}
                            if all(s["Job ID"] is None for s in selected_rows):
                                enabled |= {"eject-button","ready-button","unready-button"}
                            elif all(s["Job ID"] is not None for s in selected_rows):
                                enabled |= {"cancel-button"}
                        elif all(s["Sample ID"] is None for s in selected_rows):
                            enabled |= {"load-button"}
                elif table == "jobs":
                    if all(s["Server label"] in accessible_servers for s in selected_rows):
                        enabled |= {"snapshot-button"}
                        if all(s["Status"] in ["r","q","qw"] for s in selected_rows):
                            enabled |= {"cancel-button"}
                elif table == "samples":  # noqa: SIM102
                    if all(s["Sample ID"] is not None for s in selected_rows):
                        enabled |= {"delete-button","label-button"}
                        if len(selected_rows) > 1:
                            enabled |= {"create-batch-button"}
            if table == "samples" and not OPENBIS_DISABLED:
                enabled |= {"openbis-button"}
            if any(s["Sample ID"] is not None for s in selected_rows):
                enabled |= {"view-button"}
        # False = enabled (not my choice), so this returns True if button is NOT in enabled set
        return tuple(b not in enabled for b in BUTTONS)

    @app.callback(
        Output("table-info", "children"),
        Input("table", "selectedRows"),
        Input("table", "rowData"),
        Input("table", "virtualRowData"),
    )
    def update_table_info(selected_rows, row_data, filtered_row_data):
        if not row_data:
            return "..."
        total_rows = len(row_data)
        filtered_rows_count = len(filtered_row_data) if (filtered_row_data is not None) else total_rows
        selected_rows_count = len(selected_rows) if selected_rows else 0

        return f"Selected: {selected_rows_count}/{filtered_rows_count}"

    # Copy button copies current selected rows to clipboard
    @app.callback(
        Output("clipboard", "content"),
        Output("clipboard", "n_clicks"),
        Input("copy-button", "n_clicks"),
        State("table", "selectedRows"),
        State("clipboard", "n_clicks"),
    )
    def copy_button(n, selected_rows, nclip):
        if selected_rows:
            tsv_header = "\t".join(selected_rows[0].keys())
            tsv_data = "\n".join(["\t".join(str(value) for value in row.values()) for row in selected_rows])
            nclip = 1 if nclip is None else nclip + 1
            return f"{tsv_header}\n{tsv_data}", nclip
        return no_update, no_update

    # Eject button pop up
    @app.callback(
        Output("eject-modal", "is_open"),
        Input("eject-button", "n_clicks"),
        Input("eject-yes-close", "n_clicks"),
        Input("eject-no-close", "n_clicks"),
        State("eject-modal", "is_open"),
    )
    def eject_sample_button(eject_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "eject-button":
            return not is_open
        if (button_id == "eject-yes-close" and yes_clicks) or (button_id == "eject-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When eject button confirmed, eject samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("eject-yes-close", "n_clicks"),
        State("table-data-store", "data"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def eject_sample(yes_clicks, data, selected_rows):
        if not yes_clicks:
            return no_update,0
        for row in selected_rows:
            print(f"Ejecting {row['Pipeline']}")
            sm.eject(row["Pipeline"])
        return no_update,1

    # Load button pop up, includes dynamic dropdowns for selecting samples to load
    @app.callback(
        Output("load-modal", "is_open"),
        Output("load-modal-body", "children"),
        Output("load-incrememt", "style"),
        Input("load-button", "n_clicks"),
        Input("load-yes-close", "n_clicks"),
        Input("load-no-close", "n_clicks"),
        State("load-modal", "is_open"),
        State("table", "selectedRows"),
        State("samples-store", "data"),
    )
    def load_sample_button(load_clicks, yes_clicks, no_clicks, is_open, selected_rows, possible_samples):
        if not selected_rows or not ctx.triggered:
            return is_open, no_update, no_update
        options = [{"label": s, "value": s} for s in possible_samples if s]
        # sort the selected rows by their pipeline with the same sorting as the AG grid
        pipelines = [s["Pipeline"] for s in selected_rows]
        selected_rows = [s for _,s in sorted(zip(make_pipelines_comparable(pipelines),selected_rows))]
        dropdowns = [
            html.Div(
                children=[
                    html.Label(
                        f"{s['Pipeline']}",
                        htmlFor=f"dropdown-{s['Pipeline']}",
                        style={"margin-right": "10px"},
                    ),
                    dcc.Dropdown(
                        id={"type":"load-dropdown","index":i},
                        options=options,
                        value=[],
                        multi=False,
                        style={"width": "100%"},
                    ),
                ],
                style={"display": "flex", "align-items": "center", "padding": "5px"},
            )
            for i,s in enumerate(selected_rows)
        ]
        children = ["Select the samples you want to load"] + dropdowns
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        increment = {"display": "inline-block"} if len(selected_rows) > 1 else {"display": "none"}
        if button_id == "load-button":
            return not is_open, children, increment
        if (button_id == "load-yes-close" and yes_clicks) or (button_id == "load-no-close" and no_clicks):
            return False, no_update, increment
        return is_open, no_update, increment

    # When auto-increment is pressed, increment the sample ID for each selected pipeline
    @app.callback(
        Output({"type":"load-dropdown","index":ALL}, "value"),
        Input("load-incrememt", "n_clicks"),
        Input("load-clear", "n_clicks"),
        State({"type":"load-dropdown","index":ALL}, "value"),
        State("samples-store", "data"),
    )
    def update_load_selection(inc_clicks, clear_clicks, selected_samples, possible_samples):
        if not ctx.triggered:
            return selected_samples
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # If clear, clear all selected samples
        if button_id == "load-clear":
            return [[] for _ in selected_samples]

        # If auto-increment, go through the list, if the sample is empty increment the previous sample
        if button_id == "load-incrememt":
            for i in range(1,len(selected_samples)):
                if not selected_samples[i]:
                    prev_sample = selected_samples[i-1]
                    if prev_sample:
                        prev_sample_number = prev_sample.split("_")[-1]
                        #convert to int, increment, convert back to string with same padding
                        new_sample_number = str(int(prev_sample_number)+1).zfill(len(prev_sample_number))
                        new_sample = "_".join(prev_sample.split("_")[:-1]) + "_" + new_sample_number
                        if new_sample in possible_samples:
                            selected_samples[i] = new_sample
        return selected_samples

    # When load is pressed, load samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("load-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        State({"type": "load-dropdown", "index": ALL}, "value"),
        prevent_initial_call=True,
    )
    def load_sample(yes_clicks, selected_rows, selected_samples):
        if not yes_clicks:
            return no_update,0
        pipelines = [s["Pipeline"] for s in selected_rows]
        pipelines = [s for _,s in sorted(zip(make_pipelines_comparable(pipelines),pipelines))]
        for sample, pipeline in zip(selected_samples, pipelines):
            if not sample:
                continue
            print(f"Loading {sample} to {pipeline}")
            sm.load(sample, pipeline)
        return no_update, 1

    # Ready button pop up
    @app.callback(
        Output("ready-modal", "is_open"),
        Input("ready-button", "n_clicks"),
        Input("ready-yes-close", "n_clicks"),
        Input("ready-no-close", "n_clicks"),
        State("ready-modal", "is_open"),
    )
    def ready_pipeline_button(ready_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "ready-button":
            return not is_open
        if (button_id == "ready-yes-close" and yes_clicks) or (button_id == "ready-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When ready button confirmed, ready pipelines and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("ready-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def ready_pipeline(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            print(f"Readying {row['Pipeline']}")
            output = sm.ready(row["Pipeline"])
        return no_update, 1

    # Unready button pop up
    @app.callback(
        Output("unready-modal", "is_open"),
        Input("unready-button", "n_clicks"),
        Input("unready-yes-close", "n_clicks"),
        Input("unready-no-close", "n_clicks"),
        State("unready-modal", "is_open"),
    )
    def unready_pipeline_button(unready_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "unready-button":
            return not is_open
        if (button_id == "unready-yes-close" and yes_clicks) or (button_id == "unready-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When unready button confirmed, unready pipelines and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("unready-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def unready_pipeline(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            print(f"Unreadying {row['Pipeline']}")
            output = sm.unready(row["Pipeline"])
        return no_update, 1

    # Submit button pop up
    @app.callback(
        Output("submit-modal", "is_open"),
        Input("submit-button", "n_clicks"),
        Input("submit-yes-close", "n_clicks"),
        Input("submit-no-close", "n_clicks"),
        State("submit-modal", "is_open"),
    )
    def submit_pipeline_button(submit_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "submit-button":
            return not is_open
        if (button_id == "submit-yes-close" and yes_clicks) or (button_id == "submit-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # Submit pop up - check that the json file is valid
    @app.callback(
        Output("validator", "children"),
        Output("payload", "data"),
        Input("submit-upload", "contents"),
        State("submit-upload", "filename"),
        prevent_initial_call=True,
    )
    def check_json(contents, filename):
        if not contents:
            return "No file selected", {}
        content_type, content_string = contents.split(",")
        try:
            decoded = base64.b64decode(content_string).decode("utf-8")
        except UnicodeDecodeError:
            return f"ERROR: {filename} had decoding error", {}
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError:
            return f"ERROR: {filename} is invalid json file", {}
        # TODO use proper tomato schemas to validate the json
        missing_keys = [key for key in ["version","method","tomato"] if key not in payload.keys()]
        if missing_keys:
            msg = f"ERROR: {filename} is missing keys: {', '.join(missing_keys)}"
            return msg, {}
        return f"{filename} loaded", payload
    # Submit pop up - show custom capacity input if custom capacity is selected
    @app.callback(
        Output("submit-capacity-div", "style"),
        Input("submit-crate", "value"),
        prevent_initial_call=True,
    )
    def submit_custom_crate(crate):
        if crate == "custom":
            return {"display": "block"}
        return {"display": "none"}
    # Submit pop up - enable submit button if json valid and a capacity is given
    @app.callback(
        Output("submit-yes-close", "disabled"),
        Input("payload", "data"),
        Input("submit-crate", "value"),
        Input("submit-capacity", "value"),
        prevent_initial_call=True,
    )
    def enable_submit(payload, crate, capacity):
        if not payload or not crate:
            return True  # disabled
        if crate == "custom" and (not capacity or capacity < 0 or capacity > 10):  # noqa: SIM103
            return True  # disabled
        return False  # enabled
    # When submit button confirmed, submit the payload with sample and capacity, refresh database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("submit-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        State("payload", "data"),
        State("submit-crate", "value"),
        State("submit-capacity", "value"),
        prevent_initial_call=True,
    )
    def submit_pipeline(yes_clicks, selected_rows, payload, crate_calc, capacity):
        if not yes_clicks:
            return no_update, 0
        # capacity_Ah: float | 'areal','mass','nominal'
        capacity_Ah = capacity / 1000 if crate_calc == "custom" else crate_calc
        if not isinstance(capacity_Ah, float) and capacity_Ah not in ["areal","mass","nominal"]:
            print(f"Invalid capacity calculation method: {capacity_Ah}")
            return no_update, 0
        for row in selected_rows:
            print(f"Submitting payload {payload} to sample {row['Sample ID']} with capacity_Ah {capacity_Ah}")
            # TODO gracefully handle errors here
            sm.submit(row["Sample ID"], payload, capacity_Ah)
        return no_update, 1

    # OpenBIS auto
    @app.callback(
        Output("openbis-auto-modal", "is_open"),
        Output("openbis-auto-sample-list", "children"),
        Output("notify-interval", "interval", allow_duplicate=True),
        Input("openbis-auto-button", "n_clicks"),
        Input("openbis-auto-yes-close", "n_clicks"),
        Input("openbis-auto-no-close", "n_clicks"),
        State("openbis-auto-modal", "is_open"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def openbis_auto_button(n_clicks, yes_clicks, no_clicks, is_open, selected_rows):
        if not ctx.triggered:
            return is_open, "", idle_time
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "openbis-auto-button":
            sample_list = "\n".join([f"{s['Sample ID']}" for s in selected_rows])
            return not is_open, sample_list, idle_time
        if (button_id == "openbis-auto-yes-close" and yes_clicks):
            return False, "", active_time
        if (button_id == "openbis-auto-no-close" and no_clicks):
            return False, "", idle_time
        return False, "", idle_time

    # OpenBIS custom
    @app.callback(
        Output("openbis-custom-modal", "is_open"),
        Output("openbis-custom-sample-list", "children"),
        Output("notify-interval", "interval", allow_duplicate=True),
        Input("openbis-custom-button", "n_clicks"),
        Input("openbis-custom-yes-close", "n_clicks"),
        Input("openbis-custom-no-close", "n_clicks"),
        State("openbis-custom-modal", "is_open"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def openbis_custom_button(n_clicks, yes_clicks, no_clicks, is_open, selected_rows):
        if not ctx.triggered:
            return is_open, "", idle_time
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "openbis-custom-button":
            sample_list = "\n".join([f"{s['Sample ID']}" for s in selected_rows])
            return not is_open, sample_list, idle_time
        if (button_id == "openbis-custom-yes-close" and yes_clicks):
            return False, "", active_time
        if (button_id == "openbis-custom-no-close" and no_clicks):
            return False, "", idle_time
        return False, "", idle_time
    # When an xlsx file is uploaded, check if it is valid
    @app.callback(
        Output("openbis-validator", "children"),
        Output("openbis-custom-yes-close", "disabled"),
        Input("openbis-template-upload", "contents"),
        Input("openbis-template-upload", "filename"),
        prevent_initial_call=True,
    )
    def check_xlsx(contents, filename):
        if not contents:
            return "No file selected", True
        if not filename.endswith(".xlsx"):
            return f"ERROR: {filename} is not an .xlsx file", True
        # Validation can go here
        return f"{filename} loaded", False
    # When openbis auto OR custom confirmed, upload the samples to OpenBis and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("notify-interval", "interval", allow_duplicate=True),
        Input("openbis-custom-yes-close", "n_clicks"),
        Input("openbis-auto-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        State("openbis-template-upload", "contents"),
        State("openbis-template-upload", "filename"),
        prevent_initial_call=True,
    )
    def openbis_upload(custom_clicks, auto_clicks, selected_rows, contents, filename):
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "openbis-custom-yes-close" and custom_clicks:
            custom = True
        elif button_id == "openbis-auto-yes-close" and auto_clicks:
            custom = False
        else:
            return no_update, idle_time
        personal_access_token_path = CONFIG.get("OpenBIS PAT")
        user_mapping = CONFIG.get("User mapping")
        if not personal_access_token_path:
            print("Error: missing OpenBIS personal access token")
        for sample_id in [s["Sample ID"] for s in selected_rows]:
            run_id = run_from_sample(sample_id)
            sample_folder = Path(CONFIG["Processed snapshots folder path"])/run_id/sample_id
            if not sample_folder.exists():
                print(f"Error: {sample_folder} does not exist")
                queue_notification(Notification(
                    title="Oh no!",
                    message=f"{sample_id} doesn't exist in the processed snapshots folder",
                    action="show",
                    color="orange",
                    icon=html.I(className="bi bi-exclamation-triangle"),
                ))
                continue
            try:
                if custom:  # Put the custom template the user uploaded into the sample folder
                    print(f"Copying {filename} to {sample_folder}")
                    with (sample_folder/f"{sample_id}_custom_metadata.xlsx").open("wb") as f:
                        f.write(base64.b64decode(contents.split(",")[1]))
                print(f"Uploading {sample_id} to OpenBis with template {filename}")
                push_exp(
                    str(personal_access_token_path),
                    str(sample_folder),
                    user_mapping,
                )
                queue_notification(Notification(
                    title="Success!",
                    message=f"{sample_id} uploaded to OpenBIS",
                    action="show",
                    color="green",
                    icon=html.I(className="bi bi-check-circle"),
                ))
            except Exception as e:
                print(f"Error: {sample_id} encountered an error: {e}")
                queue_notification(Notification(
                    title="Error",
                    message=f"{sample_id} encountered an error: {e}",
                    action="show",
                    color="red",
                    icon=html.I(className="bi bi-x-circle"),
                ))
        return no_update, idle_time

    # When selecting create batch, switch to batch sub-tab with samples selected
    @app.callback(
        Output("table-select", "active_tab"),
        Output("create-batch-store", "data", allow_duplicate=True),
        Input("create-batch-button", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def create_batch(n_clicks, selected_rows):
        return "batches", [s.get("Sample ID") for s in selected_rows]

    # Cancel button pop up
    @app.callback(
        Output("cancel-modal", "is_open"),
        Input("cancel-button", "n_clicks"),
        Input("cancel-yes-close", "n_clicks"),
        Input("cancel-no-close", "n_clicks"),
        State("cancel-modal", "is_open"),
    )
    def cancel_job_button(cancel_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "cancel-button":
            return not is_open
        if (button_id == "cancel-yes-close" and yes_clicks):
            return False
        if (button_id == "cancel-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When cancel confirmed, cancel the jobs and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("cancel-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def cancel_job(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            print(f"Cancelling job {row['Job ID']}")
            sm.cancel(row["Job ID"])
        return no_update, 1

    # View data
    @app.callback(
        Output("tabs", "value"),
        Output("samples-dropdown", "value"),
        Output("batch-samples-dropdown", "value"),
        Output("batch-yes-close", "n_clicks"),
        Input("view-button", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def view_data(n_clicks, selected_rows):
        if not n_clicks or not selected_rows:
            return no_update, no_update
        sample_id = [s["Sample ID"] for s in selected_rows]
        if len(sample_id) > 1:
            return "tab-2", no_update, sample_id, 1
        return "tab-1", sample_id, no_update, no_update

    # Snapshot button pop up
    @app.callback(
        Output("snapshot-modal", "is_open"),
        Input("snapshot-button", "n_clicks"),
        Input("snapshot-yes-close", "n_clicks"),
        Input("snapshot-no-close", "n_clicks"),
        State("snapshot-modal", "is_open"),
    )
    def snapshot_sample_button(snapshot_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "snapshot-button":
            return not is_open
        if (button_id == "snapshot-yes-close" and yes_clicks):
            return False
        if (button_id == "snapshot-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When snapshot confirmed, snapshot the samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Input("snapshot-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def snapshot_sample(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update
        for row in selected_rows:
            if row.get("Job ID"):
                print(f"Snapshotting {row['Job ID']}")
                sm.snapshot(row["Job ID"])
            else:
                print(f"Snapshotting {row['Sample ID']}")
                sm.snapshot(row["Sample ID"])
        return no_update

    # Delete button pop up
    @app.callback(
        Output("delete-modal", "is_open"),
        Input("delete-button", "n_clicks"),
        Input("delete-yes-close", "n_clicks"),
        Input("delete-no-close", "n_clicks"),
        State("delete-modal", "is_open"),
    )
    def delete_sample_button(delete_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "delete-button":
            return not is_open
        if (button_id == "delete-yes-close" and yes_clicks):
            return False
        if (button_id == "delete-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When delete confirmed, delete the samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("delete-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        prevent_initial_call=True,
    )
    def delete_sample(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        sample_ids = [s["Sample ID"] for s in selected_rows]
        print(f"Deleting {sample_ids}")
        delete_samples(sample_ids)
        return no_update, 1

    # Add samples button pop up
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Output("add-samples-confirm", "displayed"),
        Input("sample-upload", "contents"),
        State("sample-upload", "filename"),
        prevent_initial_call=True,
    )
    def upload_samples(contents, filename):
        if contents:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            samples = json.loads(decoded)
            print(f"Adding samples {filename}")
            try:
                add_samples_from_object(samples)
            except ValueError as e:
                if "already exist" in str(e):
                    return no_update, True # Open confirm dialog
            return 1, no_update  # Refresh the database
        return no_update, no_update

    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("add-samples-confirm", "submit_n_clicks"),
        State("sample-upload", "contents"),
        State("sample-upload", "filename"),
        prevent_initial_call=True,
    )
    def upload_overwrite_samples(submit_n_clicks,contents,filename):
        if submit_n_clicks and contents:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            samples = json.loads(decoded)
            print(f"Adding samples {filename}")
            add_samples_from_object(samples, overwrite=True)
            return 1
        return no_update

    # Label button pop up
    @app.callback(
        Output("label-modal", "is_open"),
        Input("label-button", "n_clicks"),
        Input("label-yes-close", "n_clicks"),
        Input("label-no-close", "n_clicks"),
        State("label-modal", "is_open"),
    )
    def label_sample_button(label_clicks, yes_clicks, no_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "label-button":
            return not is_open
        if (button_id == "label-yes-close" and yes_clicks):
            return False
        if (button_id == "label-no-close" and no_clicks):
            return False
        return is_open, no_update, no_update, no_update
    # When label confirmed, label the samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("label-yes-close", "n_clicks"),
        State("table", "selectedRows"),
        State("label-input", "value"),
        prevent_initial_call=True,
    )
    def label_sample(yes_clicks, selected_rows, label):
        if not yes_clicks:
            return no_update, 0
        sample_ids = [s["Sample ID"] for s in selected_rows]
        print(f"Labelling {sample_ids} with '{label}'")
        update_sample_label(sample_ids, label)
        print("Updating metadata in cycles.*.json and full.*.h5 files")
        update_sample_metadata(sample_ids)
        return no_update, 1
