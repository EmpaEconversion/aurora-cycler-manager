"""Copyright © 2025-2026, Empa.

Database view tab layout and callbacks for the visualiser app.
"""

import json
import logging
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import dash_ag_grid as dag
import dash_mantine_components as dmc
import dash_uploader as du
import paramiko
from dash import ALL, Dash, Input, NoUpdate, Output, State, callback, clientside_callback, dcc, html, no_update
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from flask import abort, send_file
from flask.typing import ResponseReturnValue

import aurora_cycler_manager.battinfo_utils as bu
from aurora_cycler_manager.analysis import update_sample_metadata
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import (
    delete_samples,
    get_batch_details,
    update_sample_label,
)
from aurora_cycler_manager.server_manager import ServerManager, _Sample
from aurora_cycler_manager.visualiser import file_io
from aurora_cycler_manager.visualiser.db_batch_edit import (
    batch_edit_layout,
    register_batch_edit_callbacks,
)
from aurora_cycler_manager.visualiser.db_protocol_edit import (
    protocol_edit_layout,
    register_protocol_edit_callbacks,
)
from aurora_cycler_manager.visualiser.funcs import (
    get_database,
    get_db_last_update,
    make_pipelines_comparable,
)
from aurora_cycler_manager.visualiser.notifications import (
    active_time,
    error_notification,
    idle_time,
    success_notification,
)

# ------------------------ Initialize server manager ------------------------- #

# If user cannot ssh connect then disable features that require it
logger = logging.getLogger(__name__)
CONFIG = get_config()
sm: ServerManager | None = None
try:
    sm = ServerManager()
except (paramiko.SSHException, FileNotFoundError, ValueError) as e:
    logger.warning(e)
    logger.warning("You cannot access any servers. Running in view-only mode.")

# ---------------------------- Utility functions ----------------------------- #

DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "aurora_download_tmp"
DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "aurora_upload_tmp"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


def cleanup_temp_folder() -> None:
    """Remove temp files older than an hour in temp download dir. Only keep last 5 files."""
    files = {}
    for f in DOWNLOAD_DIR.iterdir():
        if f.is_file():
            creation_uts = f.stat().st_birthtime
            now_uts = datetime.now(timezone.utc).timestamp()
            age = now_uts - creation_uts
            if age > 3600:
                f.unlink()
            else:
                files[f] = age
    if len(files) > 5:
        files = dict(sorted(files.items(), key=lambda x: x[1]))
        for f in list(files.keys())[5:]:
            f.unlink()


# ----------------------------- Layout - tables ------------------------------ #

DEFAULT_TABLE_OPTIONS: dict[str, str | dict] = {
    "dashGridOptions": {
        "enableCellTextSelection": False,
        "ensureDomOrder": True,
        "rowSelection": "multiple",
    },
    "defaultColDef": {
        "filter": True,
        "sortable": True,
        "floatingFilter": True,
        "cellRenderer": "agAnimateShowChangeCellRenderer",
    },
    "className": "ag-theme-quartz",
}
TABLES = [
    "samples-table",
    "pipelines-table",
    "jobs-table",
    "results-table",
]
samples_table = dag.AgGrid(
    id="samples-table",
    selectedRows=[],
    getRowId="params.data['Sample ID']",
    **DEFAULT_TABLE_OPTIONS,
)
pipelines_table = dag.AgGrid(
    id="pipelines-table",
    selectedRows=[],
    getRowId="params.data['Pipeline']",
    **DEFAULT_TABLE_OPTIONS,
)
jobs_table = dag.AgGrid(
    id="jobs-table",
    selectedRows=[],
    getRowId="params.data['Job ID']",
    **DEFAULT_TABLE_OPTIONS,
)
results_table = dag.AgGrid(
    id="results-table",
    selectedRows=[],
    getRowId="params.data['Sample ID']",
    **DEFAULT_TABLE_OPTIONS,
)

# ----------------------------- Layout - buttons ----------------------------- #

# Define visibility settings for buttons and divs when switching between tabs
CONTAINERS = [
    "table-container",
    "batch-container",
    "protocol-container",
]
BUTTONS = [
    "copy-button",
    "load-button",
    "eject-button",
    "submit-button",
    "cancel-button",
    "view-button",
    "snapshot-button",
    "create-batch-button",
    "delete-button",
    "label-button",
    "download-button",
    "upload-button",
]
SHOW_CONTAINER_STYLE = {
    "flex": "1",
    "minHeight": 0,
    "display": "flex",
    "flexDirection": "column",
}
visibility_settings = {
    "batches": {
        "batch-container",
    },
    "protocols": {
        "protocol-container",
    },
    "pipelines": {
        "table-container",
        "copy-button",
        "load-button",
        "eject-button",
        "submit-button",
        "cancel-button",
        "view-button",
        "snapshot-button",
        "label-button",
        "create-batch-button",
        "download-button",
        "upload-button",
    },
    "jobs": {
        "table-container",
        "copy-button",
        "cancel-button",
        "snapshot-button",
        "upload-button",
    },
    "results": {
        "table-container",
        "copy-button",
        "view-button",
        "label-button",
        "create-batch-button",
        "download-button",
        "upload-button",
    },
    "samples": {
        "table-container",
        "copy-button",
        "view-button",
        "batch-button",
        "delete-button",
        "label-button",
        "create-batch-button",
        "download-button",
        "upload-button",
    },
}

# Buttons under tables
button_layout = dmc.Flex(
    justify="space-between",
    children=[
        # Left aligned buttons
        dmc.Group(
            justify="flex-start",
            gap="xs",
            children=[
                dmc.Button(
                    "Copy",
                    leftSection=html.I(className="bi bi-clipboard"),
                    id="copy-button",
                    disabled=True,
                ),
                dmc.Button(
                    "Load",
                    leftSection=html.I(className="bi bi-arrow-90deg-down"),
                    id="load-button",
                ),
                dmc.Button(
                    "Eject",
                    leftSection=html.I(className="bi bi-arrow-90deg-right"),
                    id="eject-button",
                ),
                dmc.Button(
                    "Submit",
                    leftSection=html.I(className="bi bi-upload"),
                    id="submit-button",
                ),
                dmc.Button(
                    "Cancel",
                    leftSection=html.I(className="bi bi-x-circle"),
                    id="cancel-button",
                    color="red",
                ),
                dmc.Button(
                    "View",
                    leftSection=html.I(className="bi bi-graph-down"),
                    id="view-button",
                ),
                dmc.Button(
                    "Snapshot",
                    leftSection=html.I(className="bi bi-camera"),
                    id="snapshot-button",
                ),
                dmc.Button(
                    "Label",
                    leftSection=html.I(className="bi bi-tag"),
                    id="label-button",
                    className="me-1",
                ),
                dmc.Button(
                    "Batch",
                    leftSection=html.I(className="bi bi-grid-3x2-gap-fill"),
                    id="create-batch-button",
                ),
                dmc.Button(
                    "Delete",
                    leftSection=html.I(className="bi bi-trash3"),
                    id="delete-button",
                    color="red",
                ),
                dmc.Button(
                    "Upload",
                    leftSection=html.I(className="bi bi-upload"),
                    id="upload-button",
                    className="me-1",
                ),
                dmc.Button(
                    "Download",
                    leftSection=html.I(className="bi bi-download"),
                    id="download-button",
                    className="me-1",
                ),
            ],
        ),
        # Right aligned buttons
        dmc.Group(
            justify="flex-end",
            gap="xs",
            children=[
                html.Div(
                    "Loading...",
                    id="table-info",
                    className="me-1",
                    style={"display": "inline-block", "opacity": "0.5"},
                ),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        html.I(className="bi bi-arrow-clockwise"),
                        id="refresh-database",
                        size="lg",
                    ),
                    label="Refresh database",
                    id="last-refreshed",
                    multiline=True,
                    openDelay=500,
                ),
                dmc.Tooltip(
                    dmc.ActionIcon(
                        html.I(className="bi bi-database-down"),
                        id="update-database",
                        size="lg",
                    ),
                    label="Update database by querying cyclers",
                    id="last-updated",
                    multiline=True,
                    openDelay=500,
                ),
            ],
        ),
    ],
)

# Tabs for tables, batch/protocol edit
tabs_layout = dmc.Tabs(
    [
        dmc.TabsList(
            [
                dmc.TabsTab("Samples", value="samples"),
                dmc.TabsTab("Pipelines", value="pipelines"),
                dmc.TabsTab("Jobs", value="jobs"),
                dmc.TabsTab("Results", value="results"),
                dmc.TabsTab("Batches", value="batches"),
                dmc.TabsTab("Protocols", value="protocols"),
            ],
        ),
    ],
    id="table-select",
    value="samples",
)

# ------------------------- Layout - Table container ------------------------- #

table_container_layout = html.Div(
    id="table-container",
    style={
        "display": "flex",
        "flexDirection": "column",
        "flex": "1",
        "minHeight": 0,
    },
    children=[
        # Tables - grows
        html.Div(
            style={
                "flex": "1",
                "overflow": "auto",
                "minHeight": 0,
            },
            children=[
                samples_table,
                pipelines_table,
                jobs_table,
                results_table,
            ],
        ),
        # Buttons - minimum space
        html.Div(
            style={
                "padding": "10px",
                "flexShrink": 0,
            },
            children=[button_layout],
        ),
    ],
)

# ------------------------------ Layout - modals ----------------------------- #


eject_modal = dmc.Modal(
    title="Eject",
    children=[
        dmc.Text("Are you sure you want eject the selected samples?"),
        dmc.Button(
            "Eject",
            id="eject-yes-close",
        ),
    ],
    id="eject-modal",
    centered=True,
)

load_modal = dmc.Modal(
    title="Load",
    children=[
        dmc.Text(
            "Load samples?",
            id="load-modal-text",
        ),
        dmc.Space(h="md"),
        dmc.Group(
            [
                dmc.Button(
                    "Load",
                    id="load-yes-close",
                ),
                dmc.Button(
                    "Auto-increment",
                    id="load-increment",
                    color="gray",
                ),
                dmc.Button(
                    "Clear all",
                    id="load-clear",
                    color="gray",
                ),
            ],
        ),
        dcc.Store(id="load-modal-store", data={}),
    ],
    id="load-modal",
    centered=True,
)

submit_modal = dmc.Modal(
    title="Submit",
    id="submit-modal",
    opened=False,
    centered=True,
    children=dmc.Stack(
        [
            dcc.Store(id="payload", data={}),
            dmc.Select(
                label="Select protocol to submit",
                id="submit-select-payload",
                data=[],
                placeholder="Select protocol",
                searchable=True,
                clearable=True,
            ),
            dmc.Text("No file selected", id="validator", size="sm"),
            dmc.Select(
                label="Calculate C-rate by:",
                id="submit-crate",
                data=[
                    {"value": "areal", "label": "areal capacity x area from db"},
                    {"value": "mass", "label": "specific capacity x mass from db"},
                    {"value": "nominal", "label": "nominal capacity from db"},
                    {"value": "custom", "label": "custom capacity value"},
                ],
                value="mass",
            ),
            dcc.Store("submit-crate-vals", data={}),
            dmc.NumberInput(
                id="submit-capacity",
                label="Custom capacity value",
                placeholder="Enter capacity in mAh",
                min=0,
                max=10,
                suffix=" mAh",
                style={"display": "none"},  # Hidden by default, shown when custom is selected
            ),
            dmc.Text(
                id="submit-capacity-display",
                size="sm",
                style={"whiteSpace": "pre-line"},
            ),
            dmc.Button(
                "Submit",
                id="submit-yes-close",
                disabled=True,
            ),
        ],
    ),
)

cancel_modal = dmc.Modal(
    title="Cancel",
    children=dmc.Stack(
        [
            dmc.Text("Are you sure you want to cancel the selected jobs?"),
            dmc.Button(
                "Cancel",
                id="cancel-yes-close",
                color="red",
            ),
        ],
    ),
    id="cancel-modal",
    centered=True,
)

snapshot_modal = dmc.Modal(
    title="Snapshot",
    children=dmc.Stack(
        [
            dmc.Text("Do you want to snapshot the selected samples?"),
            dmc.Text("This could take minutes per sample depending on data size."),
            dmc.Button(
                "Snapshot",
                id="snapshot-yes-close",
                color="orange",
            ),
        ],
    ),
    id="snapshot-modal",
    centered=True,
)

create_batch_modal = dmc.Modal(
    title="Create batch",
    children=dmc.Stack(
        [
            dmc.Text("Create a batch from the selected samples?"),
            dmc.TextInput(
                id="batch-name",
                placeholder="Batch name",
            ),
            dmc.Textarea(
                id="batch-description",
                placeholder="Batch description",
            ),
            dmc.Button(
                "Create",
                id="batch-save-yes-close",
            ),
        ],
    ),
    id="batch-save-modal",
    centered=True,
)

delete_modal = dmc.Modal(
    title="Delete samples",
    children=dmc.Stack(
        [
            dmc.Text("Are you sure you want to remove the selected samples from the database?"),
            dmc.Text("This will not delete the same files or any experimental data."),
            dmc.Button(
                "Delete",
                id="delete-yes-close",
                color="red",
            ),
        ],
    ),
    id="delete-modal",
    centered=True,
)

label_modal = dmc.Modal(
    title="Label samples",
    children=dmc.Stack(
        [
            dmc.Text("Add a label to the selected samples."),
            dmc.TextInput(
                id="label-input",
                placeholder="This overwrites any existing label",
                label="Label",
            ),
            dmc.Button(
                "Label",
                id="label-yes-close",
            ),
        ],
    ),
    id="label-modal",
    centered=True,
)

download_modal = dmc.Modal(
    title="Download",
    children=dmc.Stack(
        [
            dmc.Fieldset(
                legend="Select files to download",
                children=dmc.Stack(
                    [
                        dmc.Checkbox(label="BDF CSV time-series", id="download-bdf-csv"),
                        dmc.Checkbox(label="BDF parquet time-series", id="download-bdf-parquet"),
                        dmc.Checkbox(label="CSV cycling summary", id="download-cycles-csv"),
                        dmc.Checkbox(label="Parquet cycling summary", id="download-cycles-parquet"),
                        dmc.Checkbox(label="JSON-LD ontologised metadata", id="download-jsonld"),
                    ]
                ),
            ),
            html.Div(
                children=[
                    dmc.Group(
                        children=[
                            dmc.Button(
                                "Download Zenodo xlsx template",
                                id="download-zenodo-info-button-element",
                                leftSection=html.I(className="bi bi-download"),
                            ),
                            dcc.Upload(
                                dmc.Button(
                                    "Add Zenodo xlsx info",
                                    id="upload-zenodo-info-button-element",
                                    leftSection=html.I(className="bi bi-upload"),
                                    fullWidth=True,
                                    style={"width": "100%"},
                                ),
                                id="upload-zenodo-info-button",
                                accept=".json,.xlsx",
                                max_size=512 * 1024 * 1024,
                                multiple=False,
                                style={"display": "block", "width": "100%"},
                            ),
                        ],
                        grow=True,
                    ),
                    dmc.Text(
                        "Fill and upload xlsx template to add extra metadata",
                        id="upload-zenodo-status",
                    ),
                    dcc.Download(
                        id="download-zenodo-info-link",
                    ),
                ],
                id="upload-zenodo-group",
                style={"display": "none"},
            ),
            dmc.Button(
                "Process",
                id="download-process-button",
                leftSection=html.I(className="bi bi-arrow-repeat"),
                disabled=True,
            ),
            dmc.Progress(value=0, id="process-progress"),
            dmc.Alert(
                title="Process status",
                children="Waiting...",
                color="gray",
                id="download-alert",
                style={
                    "white-space": "pre-wrap",
                    "minHeight": "200px",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                },
            ),
            dmc.NavLink(
                label="Download data",
                id="download-yes-close",
                href=None,
                disabled=True,
                target="_blank",
                variant="filled",
                color="blue",
                leftSection=html.I(className="bi bi-download"),
                active=True,
            ),
        ],
    ),
    id="download-modal",
    centered=True,
    size="lg",
)

upload_modal = dmc.Modal(
    title="Upload",
    children=dmc.Stack(
        [
            dmc.Text("You can upload:"),
            dmc.List(
                [
                    dmc.ListItem("Samples as a .json file"),
                    dmc.ListItem("Data as a .zip - subfolders must be Sample ID"),
                    dmc.ListItem("BattINFO .xlsx and .jsonld and auxiliary .jsonld files"),
                    dmc.ListItem("Unicycler protocols as a .json file"),
                ]
            ),
            du.Upload(id="dash-uploader", max_file_size=2048),
            dcc.Store(id="upload-filepath", data=""),
            dcc.Store(id="upload-store", data={"file": None, "data": None}),
            dmc.Alert(
                title="File status",
                children="You haven't uploaded anything",
                color="gray",
                id="upload-alert",
                style={"white-space": "pre-wrap"},
            ),
            dmc.Button(
                "Confirm",
                id="upload-data-confirm-button",
                disabled=True,
            ),
        ],
    ),
    id="upload-modal",
    centered=True,
    size="lg",
)

# ------------------------------- Main layout -------------------------------- #

db_view_layout = html.Div(
    style={
        "flex": 1,
        "display": "flex",
        "flexDirection": "column",
        "minHeight": 0,
    },
    children=[
        # Sub-tabs
        html.Div(
            style={
                "padding": "10px",
                "flexShrink": 0,
            },
            children=[tabs_layout],
        ),
        # Content
        html.Div(
            style={
                "flex": "1",
                "minHeight": 0,
                "display": "flex",
                "flexDirection": "column",
            },
            children=[
                table_container_layout,
                batch_edit_layout,
                protocol_edit_layout,
            ],
        ),
        # Invisible stuff
        dcc.Clipboard(id="clipboard", style={"display": "none"}),
        dcc.Store(id="selected-rows-store", data={}),
        dcc.Store(id="len-store", data={}),
        eject_modal,
        load_modal,
        submit_modal,
        cancel_modal,
        snapshot_modal,
        create_batch_modal,
        delete_modal,
        label_modal,
        download_modal,
        upload_modal,
    ],
)

# -------------------------------- Callbacks --------------------------------- #


def register_db_view_callbacks(app: Dash) -> None:
    """Register callbacks for the database view layout."""
    register_batch_edit_callbacks(app)
    register_protocol_edit_callbacks(app)
    du.configure_upload(app, UPLOAD_DIR)

    # Update data in tables when it changes
    @app.callback(
        Output("samples-table", "rowData"),
        Output("samples-table", "columnDefs"),
        Output("pipelines-table", "rowData"),
        Output("pipelines-table", "columnDefs"),
        Output("jobs-table", "rowData"),
        Output("jobs-table", "columnDefs"),
        Output("results-table", "rowData"),
        Output("results-table", "columnDefs"),
        Output("len-store", "data"),
        Input("table-data-store", "data"),
        running=[(Output("loading-message-store", "data"), "Updating tables...", "")],
        prevent_initial_call=True,
    )
    def update_data(data: dict[str, dict]) -> tuple:
        return (
            data["data"].get("samples", no_update),
            data["column_defs"].get("samples", no_update),
            data["data"].get("pipelines", no_update),
            data["column_defs"].get("pipelines", no_update),
            data["data"].get("jobs", no_update),
            data["column_defs"].get("jobs", no_update),
            data["data"].get("results", no_update),
            data["column_defs"].get("results", no_update),
            {
                "samples": len(data["data"].get("samples", [])),
                "pipelines": len(data["data"].get("pipelines", [])),
                "jobs": len(data["data"].get("jobs", [])),
                "results": len(data["data"].get("results", [])),
            },
        )

    # Update the buttons displayed depending on the table selected
    @app.callback(
        [Output(element, "style") for element in CONTAINERS],
        [Output(element, "style") for element in BUTTONS],
        [Output(element, "style") for element in TABLES],
        Input("table-select", "value"),
    )
    def update_table(table: str) -> tuple:
        settings: set = visibility_settings.get(table, set())
        show_container = SHOW_CONTAINER_STYLE
        show_table = {"height": "100%"}
        show_button: dict = {}
        hide = {"display": "none"}
        container_visibilities = [show_container if element in settings else hide for element in CONTAINERS]
        button_visibilities = [show_button if element in settings else hide for element in BUTTONS]
        table_visibilities = [show_table if element == f"{table}-table" else hide for element in TABLES]
        return (
            *container_visibilities,
            *button_visibilities,
            *table_visibilities,
        )

    # Update data store with the selected rows, and update the message below the table
    # Triggers when table or selection changes
    @app.callback(
        Output("selected-rows-store", "data"),
        Output("table-info", "children"),
        Input("samples-table", "selectedRows"),
        Input("pipelines-table", "selectedRows"),
        Input("jobs-table", "selectedRows"),
        Input("results-table", "selectedRows"),
        Input("table-select", "value"),
        Input("len-store", "data"),
        prevent_initial_call=True,
    )
    def update_selected_rows(
        samples: list,
        pipelines: list,
        jobs: list,
        results: list,
        table: str,
        lens: dict,
    ) -> tuple[list, str]:
        message_dict = {
            "samples": (samples, f"{len(samples) if samples else 0}/{lens['samples'] if lens else 0}"),
            "pipelines": (pipelines, f"{len(pipelines) if pipelines else 0}/{lens['pipelines'] if lens else 0}"),
            "jobs": (jobs, f"{len(jobs) if jobs else 0}/{lens['jobs'] if lens else 0}"),
            "results": (results, f"{len(results) if results else 0}/{lens['results'] if lens else 0}"),
        }
        return message_dict.get(table, ([], "..."))

    # Refresh the local data from the database
    @app.callback(
        Output("table-data-store", "data"),
        Output("last-refreshed", "label"),
        Output("last-updated", "label"),
        Output("samples-store", "data"),
        Output("batches-store", "data"),
        Input("refresh-database", "n_clicks"),
        Input("db-update-interval", "n_intervals"),
        running=[(Output("loading-message-store", "data"), "Reading database...", "")],
    )
    def refresh_database(_n_clicks: int, _n_intervals: int) -> tuple:
        db_data = get_database()
        dt_string = datetime.now(CONFIG["tz"]).strftime("%Y-%m-%d %H:%M:%S %z")
        last_checked = get_db_last_update().astimezone(CONFIG["tz"]).strftime("%Y-%m-%d %H:%M:%S %z")
        samples = [s["Sample ID"] for s in db_data["data"]["samples"]]
        batches = get_batch_details()
        return (
            db_data,
            f"Refresh database, last refreshed: {dt_string}",
            f"Update database, last updated: {last_checked}",
            samples,
            batches,
        )

    # Update the database i.e. connect to servers and grab new info, then refresh the local data
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Output("notifications-container", "sendNotifications", allow_duplicate=True),
        Input("update-database", "n_clicks"),
        running=[(Output("loading-message-store", "data"), "Updating database - querying servers...", "")],
        prevent_initial_call=True,
    )
    def update_database(n_clicks: int) -> tuple:
        if n_clicks is None:
            raise PreventUpdate
        if not sm or not sm.servers:
            return 0, [error_notification("Error", "You do not have access to any cycling servers.")]
        try:
            sm.update_db()
        except Exception as e:
            return 0, [error_notification("Error", str(e))]
        else:
            return 1, NoUpdate

    # Enable or disable buttons (load, eject, etc.) depending on what is selected in the table
    @app.callback(
        [Output(b, "disabled") for b in BUTTONS],
        Input("selected-rows-store", "data"),
        State("table-select", "value"),
        prevent_initial_call=True,
    )
    def enable_buttons(selected_rows: list, table: str) -> tuple[bool, ...]:
        enabled = set()
        # Add buttons to enabled set with union operator |=
        if sm is not None:
            enabled |= {"upload-button"}
        if selected_rows:
            enabled |= {"copy-button"}
            if len(selected_rows) <= 200:  # To avoid enormous zip files being stored
                enabled |= {"download-button"}
            if sm is not None:
                if table == "samples":
                    if all(s.get("Sample ID") is not None for s in selected_rows):
                        enabled |= {"delete-button", "label-button", "create-batch-button"}
                elif table == "pipelines":
                    all_samples = all(s.get("Sample ID") is not None for s in selected_rows)
                    all_servers = all(s.get("Server label") in sm.servers for s in selected_rows)
                    no_samples = all(s.get("Sample ID") is None for s in selected_rows)
                    if all_samples:
                        enabled |= {"label-button", "create-batch-button"}
                        if all_servers:
                            enabled |= {"submit-button", "snapshot-button"}
                            if all(s["Job ID"] is None for s in selected_rows):
                                enabled |= {"eject-button"}
                            elif all(s.get("Job ID") is not None for s in selected_rows):
                                enabled |= {"cancel-button"}
                    elif all_servers and no_samples:
                        enabled |= {"load-button"}
                elif table == "jobs":
                    if all(s.get("Server label") in sm.servers for s in selected_rows):
                        enabled |= {"snapshot-button"}
                        if all(s.get("Job ID") for s in selected_rows):
                            enabled |= {"cancel-button"}
                elif table == "results" and all(s.get("Sample ID") is not None for s in selected_rows):
                    enabled |= {"label-button", "create-batch-button"}
            if any(s["Sample ID"] is not None for s in selected_rows):
                enabled |= {"view-button"}
        # False = enabled (not my choice), so this returns True if button is NOT in enabled set
        return tuple(b not in enabled for b in BUTTONS)

    # Copy button copies current selected rows to clipboard
    @app.callback(
        Output("clipboard", "content"),
        Output("clipboard", "n_clicks"),
        Input("copy-button", "n_clicks"),
        State("selected-rows-store", "data"),
        State("clipboard", "n_clicks"),
        prevent_initial_call=True,
    )
    def copy_button(_n: int, selected_rows: list, nclip: int) -> tuple[str, int]:
        if selected_rows:
            tsv_header = "\t".join(selected_rows[0].keys())
            tsv_data = "\n".join(["\t".join(str(value) for value in row.values()) for row in selected_rows])
            nclip = 1 if nclip is None else nclip + 1
            return f"{tsv_header}\n{tsv_data}", nclip
        raise PreventUpdate

    # Eject button pop up
    @app.callback(
        Output("eject-modal", "opened"),
        Input("eject-button", "n_clicks"),
        Input("eject-yes-close", "n_clicks"),
        State("eject-modal", "opened"),
        prevent_initial_call=True,
    )
    def eject_sample_button(_eject_clicks: int, yes_clicks: int, is_open: bool) -> bool:
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "eject-button":
            return not is_open
        if button_id == "eject-yes-close" and yes_clicks:
            return False
        return is_open

    # When eject button confirmed, eject samples and refresh the database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("eject-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        running=[(Output("loading-message-store", "data"), "Ejecting samples...", "")],
        prevent_initial_call=True,
    )
    def eject_sample(yes_clicks: int, selected_rows: list) -> int:
        if not yes_clicks:
            return 0
        for row in selected_rows:
            logger.info("Ejecting Sample %s from the Pipeline %s", row["Sample ID"], row["Pipeline"])
            sm.eject(row["Pipeline"], row["Sample ID"])
        return 1

    # Load button pop up, includes dynamic dropdowns for selecting samples to load
    @app.callback(
        Output("load-modal", "opened"),
        Output("load-modal-text", "children"),
        Output("load-increment", "style"),
        Input("load-button", "n_clicks"),
        Input("load-yes-close", "n_clicks"),
        State("load-modal", "opened"),
        State("selected-rows-store", "data"),
        State("samples-store", "data"),
        prevent_initial_call=True,
    )
    def load_sample_button(
        _load_clicks: int,
        yes_clicks: int,
        is_open: bool,
        selected_rows: list,
        possible_samples: list,
    ) -> tuple[bool, list | NoUpdate, dict | NoUpdate]:
        if not selected_rows or not ctx.triggered:
            return is_open, no_update, no_update
        options = [{"label": s, "value": s} for s in possible_samples if s]
        # sort the selected rows by their pipeline with the same sorting as the AG grid
        pipelines = [s["Pipeline"] for s in selected_rows]
        selected_rows = [s for _, s in sorted(zip(make_pipelines_comparable(pipelines), selected_rows, strict=True))]
        dropdowns = [
            dmc.Select(
                label=f"{s['Pipeline']}",
                id={"type": "load-dropdown", "index": i},
                data=options,
                searchable=True,
                clearable=True,
                placeholder="Select sample",
            )
            for i, s in enumerate(selected_rows)
        ]
        children = ["Select the samples you want to load", *dropdowns]
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        increment = {"display": "inline-block"} if len(selected_rows) > 1 else {"display": "none"}
        if button_id == "load-button":
            return not is_open, children, increment
        if button_id == "load-yes-close" and yes_clicks:
            return False, no_update, increment
        return is_open, no_update, increment

    # When auto-increment is pressed, increment the sample ID for each selected pipeline
    @app.callback(
        Output({"type": "load-dropdown", "index": ALL}, "value"),
        Input("load-increment", "n_clicks"),
        Input("load-clear", "n_clicks"),
        State({"type": "load-dropdown", "index": ALL}, "value"),
        State("samples-store", "data"),
        prevent_initial_call=True,
    )
    def update_load_selection(
        _inc_clicks: int,
        _clear_clicks: int,
        selected_samples: list,
        possible_samples: list,
    ) -> list:
        if not ctx.triggered:
            return selected_samples
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # If clear, clear all selected samples
        if button_id == "load-clear":
            return [None for _ in selected_samples]

        # If auto-increment, go through the list, if the sample is empty increment the previous sample
        if button_id == "load-increment":
            for i in range(1, len(selected_samples)):
                if not selected_samples[i]:
                    prev_sample = selected_samples[i - 1]
                    if prev_sample:
                        prev_sample_number = prev_sample.split("_")[-1]
                        # convert to int, increment, convert back to string with same padding
                        new_sample_number = str(int(prev_sample_number) + 1).zfill(len(prev_sample_number))
                        new_sample = "_".join(prev_sample.split("_")[:-1]) + "_" + new_sample_number
                        if new_sample in possible_samples:
                            selected_samples[i] = new_sample
        return selected_samples

    # When load is pressed, load samples and refresh the database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("load-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        State({"type": "load-dropdown", "index": ALL}, "value"),
        running=[(Output("loading-message-store", "data"), "Loading samples...", "")],
        prevent_initial_call=True,
    )
    def load_sample(yes_clicks: int, selected_rows: list, selected_samples: list) -> int:
        if not yes_clicks:
            return 0
        pipelines = [s["Pipeline"] for s in selected_rows]
        pipelines = [s for _, s in sorted(zip(make_pipelines_comparable(pipelines), pipelines, strict=True))]
        for sample, pipeline in zip(selected_samples, pipelines, strict=True):
            if not sample:
                continue
            logger.info("Loading %s to %s", sample, pipeline)
            sm.load(pipeline, sample)
        return 1

    # Submit button pop up
    @app.callback(
        Output("submit-modal", "opened"),
        Output("submit-select-payload", "data"),
        Output("submit-crate-vals", "data"),
        Input("submit-button", "n_clicks"),
        Input("submit-yes-close", "n_clicks"),
        State("submit-modal", "opened"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def submit_pipeline_button(
        _submit_clicks: int,
        yes_clicks: int,
        _is_open: bool,
        selected_rows: list,
    ) -> tuple[bool | NoUpdate, list | NoUpdate, dict | NoUpdate]:
        if not ctx.triggered:
            return no_update, no_update, no_update
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "submit-button":
            samples = [s.get("Sample ID") for s in selected_rows]
            capacities = {
                mode: {s: _Sample.from_id(s).safe_get_sample_capacity(mode) for s in samples}
                for mode in ["areal", "mass", "nominal"]
            }
            folder = CONFIG.get("Protocols folder path")
            if folder:
                filenames = [p.name for p in folder.glob("*.json")] + [p.name for p in folder.glob("*.xml")]
                return True, filenames, capacities
            return True, [], no_update
        if button_id == "submit-yes-close" and yes_clicks:
            return False, no_update, no_update
        return no_update, no_update, no_update

    # Submit pop up - check that the json file is valid
    @app.callback(
        Output("validator", "children"),
        Output("payload", "data"),
        Input("submit-modal", "opened"),
        Input("submit-select-payload", "value"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def check_payload(opened: bool, filename: str, selected_rows: list) -> tuple[str, dict]:
        if not opened:
            return no_update, no_update
        if not filename:
            return "No file selected", {}
        folder = CONFIG.get("Protocols folder path")

        if filename.endswith(".json"):
            try:
                with (folder / filename).open(encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                return f"ERROR: {filename} is invalid json file", {}
        elif filename.endswith(".xml"):
            try:
                with (folder / filename).open("r", encoding="utf-8") as f:
                    payload = f.read()  # Store XML as string
            except Exception as e:
                return f"❌ {filename} couldn't be read as xml file: {e}", {}
        else:
            return f"❌ {filename} is not a valid file type", {}

        if any(s["Server type"] == "neware" for s in selected_rows):
            # Should be an XML file or universal JSON file
            if isinstance(payload, str):
                if (
                    payload.startswith('<?xml version="1.0"')
                    and "BTS Client" in payload
                    and 'config type="Step File"' in payload
                ):
                    return f"{filename} loaded", payload
                return f"❌ {filename} is not a Neware xml file", {}

            # It's a json dict
            if "unicycler" not in payload:
                msg = f"❌ {filename} is not a unicycler json file"
                return msg, {}

        # Passed all the checks, should be a valid payload
        return f"✅ {filename} loaded", payload

    # Submit pop up - show custom capacity input if custom capacity is selected
    @app.callback(
        Output("submit-capacity", "style"),
        Output("submit-capacity-display", "children"),
        Input("submit-crate", "value"),
        Input("submit-crate-vals", "data"),
        prevent_initial_call=True,
    )
    def submit_custom_crate(crate: str, capacities: dict) -> tuple[dict, str]:
        if crate == "custom":
            return {}, ""
        capacity_vals = capacities.get(crate, {})  # sample: capacity
        capacity_text = "\n".join(
            f"✅ {s}: {c * 1000:.3f} mAh" if c is not None else f"❌ {s}: N/A " for s, c in capacity_vals.items()
        )
        return {"display": "none"}, capacity_text

    # Submit pop up - enable submit button if json valid and a capacity is given
    @app.callback(
        Output("submit-yes-close", "disabled"),
        Input("payload", "data"),
        Input("submit-crate", "value"),
        Input("submit-capacity", "value"),
        Input("submit-crate-vals", "data"),
        prevent_initial_call=True,
    )
    def enable_submit(payload: dict, crate: str, capacity: float, capacity_vals: dict) -> bool:
        if not payload or not crate:
            return True  # Disable
        # Capacity limited to 100 mAh for safety
        if crate == "custom":
            # Disable (True) if custom capacity is None or not a valid number
            return not isinstance(capacity, (int, float)) or capacity < 0 or capacity > 100
        # Disable (True) if any capacities are not valid
        return any(c is None or c < 0 or c > 0.1 for c in capacity_vals[crate].values())

    # When submit button confirmed, submit the payload with sample and capacity, refresh database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("submit-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        State("payload", "data"),
        State("submit-crate", "value"),
        State("submit-capacity", "value"),
        running=[
            (Output("loading-message-store", "data"), "Submitting protocols...", ""),
            (Output("notify-interval", "interval"), active_time, idle_time),
        ],
        prevent_initial_call=True,
    )
    def submit_pipeline(
        yes_clicks: int,
        selected_rows: list,
        payload: dict,
        crate_calc: Literal["custom", "areal", "mass", "nominal"],
        capacity: float,
    ) -> int:
        if not yes_clicks:
            return 0
        # capacity_Ah: float | 'areal','mass','nominal'
        capacity_Ah = capacity / 1000 if crate_calc == "custom" else crate_calc
        if not isinstance(capacity_Ah, float) and capacity_Ah not in ["areal", "mass", "nominal"]:
            logger.error("Invalid capacity calculation method: %s", capacity_Ah)
            return 0
        for row in selected_rows:
            try:
                sm.submit(row["Sample ID"], payload, capacity_Ah)
                success_notification("", f"Sample {row['Sample ID']} submitted", queue=True)
            except Exception as e:
                error_notification("", f"Error submitting sample {row['Sample ID']}: {e}", queue=True)
        return 1

    # When selecting create batch, switch to batch sub-tab with samples selected
    @app.callback(
        Output("table-select", "value"),
        Output("create-batch-store", "data", allow_duplicate=True),
        Input("create-batch-button", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def create_batch(_n_clicks: int, selected_rows: list) -> tuple[str, list]:
        return "batches", [s.get("Sample ID") for s in selected_rows]

    # Cancel button pop up
    @app.callback(
        Output("cancel-modal", "opened"),
        Input("cancel-button", "n_clicks"),
        Input("cancel-yes-close", "n_clicks"),
        State("cancel-modal", "opened"),
        prevent_initial_call=True,
    )
    def cancel_job_button(_cancel_clicks: int, yes_clicks: int, is_open: bool) -> bool:
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "cancel-button":
            return not is_open
        if button_id == "cancel-yes-close" and yes_clicks:
            return False
        return is_open

    # When cancel confirmed, cancel the jobs and refresh the database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("cancel-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        running=[(Output("loading-message-store", "data"), "Cancelling jobs...", "")],
        prevent_initial_call=True,
    )
    def cancel_job(yes_clicks: int, selected_rows: list) -> int:
        if not yes_clicks:
            return 0
        for row in selected_rows:
            logger.info("Cancelling job %s", row["Job ID"])
            sm.cancel(row["Job ID"])
        return 1

    # View data
    @app.callback(
        Output("tabs", "value"),
        Output("samples-dropdown", "value"),
        Output("batch-samples-dropdown", "value"),
        Output("batch-yes-close", "n_clicks"),
        Input("view-button", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def view_data(n_clicks: int, selected_rows: list) -> tuple[str, list | NoUpdate, list | NoUpdate, int | NoUpdate]:
        if not n_clicks or not selected_rows:
            raise PreventUpdate
        sample_id = [s["Sample ID"] for s in selected_rows if s.get("Sample ID")]
        if len(sample_id) > 1:
            return "tab-2", no_update, sample_id, 1
        return "tab-1", sample_id, no_update, no_update

    # Snapshot button pop up
    @app.callback(
        Output("snapshot-modal", "opened"),
        Input("snapshot-button", "n_clicks"),
        Input("snapshot-yes-close", "n_clicks"),
        State("snapshot-modal", "opened"),
        prevent_initial_call=True,
    )
    def snapshot_sample_button(_snapshot_clicks: int, yes_clicks: int, is_open: bool) -> bool:
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "snapshot-button":
            return not is_open
        if button_id == "snapshot-yes-close" and yes_clicks:
            return False
        return is_open

    # When snapshot confirmed, snapshot the samples and refresh the database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("snapshot-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        running=[(Output("loading-message-store", "data"), "Snapshotting data...", "")],
        prevent_initial_call=True,
    )
    def snapshot_sample(yes_clicks: int, selected_rows: list) -> NoUpdate:
        if yes_clicks:
            for row in selected_rows:
                if row:
                    if row.get("Job ID"):
                        logger.info("Snapshotting %s", row["Job ID"])
                        sm.snapshot(row["Job ID"])
                    else:
                        logger.info("Snapshotting %s", row["Sample ID"])
                        sm.snapshot(row["Sample ID"])
        return no_update  # Needs any output to trigger loading spinner

    # Delete button pop up
    @app.callback(
        Output("delete-modal", "opened"),
        Input("delete-button", "n_clicks"),
        Input("delete-yes-close", "n_clicks"),
        State("delete-modal", "opened"),
        prevent_initial_call=True,
    )
    def delete_sample_button(_delete_clicks: int, yes_clicks: int, is_open: bool) -> bool:
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "delete-button":
            return not is_open
        if button_id == "delete-yes-close" and yes_clicks:
            return False
        return is_open

    # When delete confirmed, delete the samples and refresh the database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("delete-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
        running=[(Output("loading-message-store", "data"), "Deleting samples...", "")],
    )
    def delete_sample(yes_clicks: int, selected_rows: list) -> int:
        if not yes_clicks:
            return 0
        sample_ids = [s["Sample ID"] for s in selected_rows]
        logger.info("Deleting [%s]", ", ".join(sample_ids))
        delete_samples(sample_ids)
        return 1

    # Label button pop up
    @app.callback(
        Output("label-modal", "opened"),
        Input("label-button", "n_clicks"),
        Input("label-yes-close", "n_clicks"),
        State("label-modal", "opened"),
        prevent_initial_call=True,
    )
    def label_sample_button(_label_clicks: int, yes_clicks: int, is_open: bool) -> bool:
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "label-button":
            return not is_open
        if button_id == "label-yes-close" and yes_clicks:
            return False
        return is_open

    # When label confirmed, label the samples and refresh the database
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("label-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        State("label-input", "value"),
        prevent_initial_call=True,
        running=[(Output("loading-message-store", "data"), "Labelling samples...", "")],
    )
    def label_sample(yes_clicks: int, selected_rows: list, label: str) -> int:
        if not yes_clicks:
            return 0
        sample_ids = [s["Sample ID"] for s in selected_rows]
        logger.info("Labelling [%s] with '%s'", ", ".join(sample_ids), label)
        update_sample_label(sample_ids, label)
        logger.info("Updating metadata in data files")
        update_sample_metadata(sample_ids)
        return 1

    # When download button is pressed, open the modal
    @app.callback(
        Output("download-modal", "opened"),
        Output("download-yes-close", "disabled", allow_duplicate=True),
        Output("process-progress", "value", allow_duplicate=True),
        Output("download-alert", "children", allow_duplicate=True),
        Output("download-alert", "color", allow_duplicate=True),
        Input("download-button", "n_clicks"),
        Input("download-yes-close", "n_clicks"),
        State("download-modal", "opened"),
        prevent_initial_call=True,
    )
    def download_data_open_modal(
        _download_clicks: int, _yes_clicks: int, is_open: bool
    ) -> tuple[bool, bool, int | NoUpdate, str | NoUpdate, str | NoUpdate]:
        if not ctx.triggered:
            raise PreventUpdate
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "download-button":
            return True, True, 0, "Waiting...", "grey"
        if button_id == "download-yes-close":
            return False, True, 0, "Waiting...", "grey"
        return is_open, True, no_update, no_update, no_update

    clientside_callback(
        """
        function updateLoadingState(n_clicks) {
            return true
        }
        """,
        Output("download-process-button", "loading", allow_duplicate=True),
        Input("download-process-button", "n_clicks"),
        prevent_initial_call=True,
    )

    # If the download template button is pressed, generate a xlsx template for zenodo info and send it
    @app.callback(
        Output("download-zenodo-info-link", "data"),
        Input("download-zenodo-info-button-element", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def generate_zenodo_info_template(_n_clicks: int, selected_rows: list) -> dict:
        """Generate a zenodo info xlsx template and return as data uri."""
        template_bytes = bu.generate_zenodo_info_xlsx_template(
            sample_ids=[s.get("Sample ID") for s in selected_rows],
            ccids=[s.get("Barcode") for s in selected_rows],
        )
        return dcc.send_bytes(template_bytes.getvalue(), "aurora_zenodo_template.xlsx")

    # If the file type is changed, allow reprocessing
    @app.callback(
        Output("download-process-button", "disabled", allow_duplicate=True),
        Output("download-yes-close", "disabled", allow_duplicate=True),
        Output("process-progress", "value", allow_duplicate=True),
        Output("download-alert", "children", allow_duplicate=True),
        Output("download-alert", "color", allow_duplicate=True),
        Output("upload-zenodo-group", "style"),
        Input("download-button", "n_clicks"),
        Input("upload-zenodo-info-button", "contents"),
        Input("download-jsonld", "checked"),
        Input("download-cycles-csv", "checked"),
        Input("download-cycles-parquet", "checked"),
        Input("download-bdf-parquet", "checked"),
        Input("download-bdf-csv", "checked"),
        prevent_initial_call=True,
    )
    def reset_process(
        _n_clicks: int, _zenodo_info: str, jsonld: bool, *inputs: tuple
    ) -> tuple[bool, bool, int, str, str, dict]:
        """If filetypes change, disable download, enable process, reset progress bar."""
        if jsonld or any(inputs):
            return False, True, 0, "Waiting...", "grey", {"display": "inline-block"} if jsonld else {"display": "none"}
        return True, True, 0, "Waiting...", "grey", {"display": "none"}

    # If a zenodo info file is uploaded, check if it is valid and show message
    @app.callback(
        Output("upload-zenodo-status", "children"),
        Input("upload-zenodo-info-button", "contents"),
        prevent_initial_call=True,
    )
    def check_zenodo_info(contents: str | None) -> str:
        """Check if uploaded zenodo info file is valid."""
        if not contents:
            return "Fill and upload xlsx template to add extra metadata"
        try:
            bu.parse_zenodo_info_xlsx(contents)
        except Exception as e:
            logger.error("Error parsing zenodo info file: %s", e)
            return f"❌ Error parsing zenodo info file: {e}"
        return "✅ Zenodo info file loaded"

    # When process is confirmed, process the data and store zip in temp folder
    @callback(
        output=[
            Output("download-yes-close", "href", allow_duplicate=True),
            Output("download-yes-close", "disabled", allow_duplicate=True),
            Output("download-process-button", "disabled", allow_duplicate=True),
        ],
        inputs=[
            Input("download-process-button", "n_clicks"),
        ],
        state=[
            State("selected-rows-store", "data"),
            State("download-cycles-parquet", "checked"),
            State("download-cycles-csv", "checked"),
            State("download-bdf-parquet", "checked"),
            State("download-bdf-csv", "checked"),
            State("download-jsonld", "checked"),
            State("upload-zenodo-info-button", "contents"),
        ],
        running=[
            (Output("download-cycles-parquet", "disabled"), True, False),
            (Output("download-cycles-csv", "disabled"), True, False),
            (Output("download-bdf-parquet", "disabled"), True, False),
            (Output("download-bdf-csv", "disabled"), True, False),
            (Output("download-jsonld", "disabled"), True, False),
            (Output("download-process-button", "disabled"), True, False),
            (Output("download-process-button", "loading"), True, False),
        ],
        cancel=Input("download-modal", "opened"),
        progress=[
            Output("process-progress", "value", allow_duplicate=True),
            Output("download-alert", "children", allow_duplicate=True),
            Output("download-alert", "color", allow_duplicate=True),
        ],
        background=True,
        prevent_initial_call=True,
    )
    def batch_process_data(
        set_progress: Callable,
        _yes_clicks: int,
        selected_rows: list,
        download_cycles_parquet: bool | None,
        download_cycles_csv: bool | None,
        download_bdf_parquet: bool | None,
        download_bdf_csv: bool | None,
        download_jsonld: bool | None,
        zenodo_info: str | None,
    ) -> tuple[str, bool, bool]:
        """Process the requested data and store in zip in temp folder on server."""
        cleanup_temp_folder()
        sample_ids = [s["Sample ID"] for s in selected_rows]
        filetypes = {
            "bdf-parquet": download_bdf_parquet,
            "bdf-csv": download_bdf_csv,
            "cycles-csv": download_cycles_csv,
            "cycles-parquet": download_cycles_parquet,
            "metadata-jsonld": download_jsonld,
        }
        filetype_set = {ft for ft, enabled in filetypes.items() if enabled}  # Convert to set
        temp_zip_path = DOWNLOAD_DIR / f"aurora_{uuid.uuid4().hex}.zip"
        try:
            file_io.create_rocrate(sample_ids, filetype_set, temp_zip_path, zenodo_info, set_progress)
        except ValueError:
            return "", True, True
        else:
            return f"/download-temp/{temp_zip_path.name}", False, True

    # This lets users download files from a URL
    @app.server.route("/download-temp/<path:filename>")
    def download_temp_file(filename: str) -> ResponseReturnValue:
        """Route to download files from the server."""
        try:
            file_path = (DOWNLOAD_DIR / filename).resolve()
        except Exception:
            abort(400)
        if not str(file_path).startswith(str(DOWNLOAD_DIR)):  # Traversal attack
            abort(403)
        if file_path.exists() and file_path.is_file():  # Download the file
            return send_file(str(file_path), as_attachment=True, download_name="aurora-data.zip")
        abort(404)
        return None

    # When upload button is pressed, open the modal
    @app.callback(
        Output("upload-modal", "opened", allow_duplicate=True),
        Input("upload-button", "n_clicks"),
        Input("upload-data-confirm-button", "n_clicks"),
        State("upload-modal", "opened"),
        prevent_initial_call=True,
    )
    def upload_data_open_modal(_n_clicks: int, _n_yes_clicks: int, is_open: bool) -> bool:
        if not ctx.triggered:
            raise PreventUpdate
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id in {"upload-button", "upload-data-confirm-button"}:
            return not is_open
        raise PreventUpdate

    # Figure out what was just uploaded and tell the user
    @du.callback(
        output=Output("upload-filepath", "data"),
        id="dash-uploader",
    )
    def callback_on_completion(status: du.UploadStatus) -> str:
        """Update filepath when upload finished."""
        return str(status.uploaded_files[0])

    @app.callback(
        Output("upload-alert", "children"),
        Output("upload-alert", "color"),
        Output("upload-data-confirm-button", "disabled"),
        Output("upload-store", "data"),
        Input("upload-filepath", "data"),
        State("selected-rows-store", "data"),
        running=[
            (Output("upload-data-button-element", "loading"), True, False),
        ],
        prevent_initial_call=True,
    )
    def figure_out_files(filepath: str, selected_rows: list) -> tuple[str, str, bool, dict]:
        return file_io.determine_file(filepath, selected_rows)

    # When hitting confirm, process the file
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("upload-data-confirm-button", "n_clicks"),
        State("upload-store", "data"),
        State("upload-filepath", "data"),
        State("selected-rows-store", "data"),
        running=[
            (Output("loading-message-store", "data"), "Processing data...", ""),
            (Output("notify-interval", "interval"), active_time, idle_time),
        ],
        prevent_initial_call=True,
    )
    def process_file(n_clicks: int, data: dict, filepath: str, selected_rows: list) -> int:
        if not n_clicks:
            raise PreventUpdate
        return file_io.process_file(data, filepath, selected_rows)
