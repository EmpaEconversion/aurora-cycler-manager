"""Copyright © 2025, Empa.

Database view tab layout and callbacks for the visualiser app.
"""

import base64
import io
import json
import logging
import tempfile
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import paramiko
from aurora_unicycler import Protocol
from battinfoconverter_backend.json_convert import convert_excel_to_jsonld
from dash import ALL, Dash, Input, NoUpdate, Output, State, callback, clientside_callback, dcc, html, no_update
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from flask import abort, send_file
from flask.typing import ResponseReturnValue

from aurora_cycler_manager.analysis import analyse_sample, update_sample_metadata
from aurora_cycler_manager.battinfo_utils import (
    generate_battery_test,
    make_test_object,
    merge_battinfo_with_db_data,
    merge_jsonld_on_type,
)
from aurora_cycler_manager.bdf_converter import aurora_to_bdf
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import (
    add_protocol_to_job,
    add_samples_from_object,
    delete_samples,
    get_all_sampleids,
    get_batch_details,
    get_sample_data,
    get_unicycler_protocols,
    update_sample_label,
)
from aurora_cycler_manager.eclab_harvester import convert_mpr
from aurora_cycler_manager.server_manager import ServerManager, _Sample
from aurora_cycler_manager.utils import run_from_sample
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

DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "aurora_tmp"
DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)


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
    "style": {
        "height": "calc(100vh - 200px)",
        "width": "100%",
        "minHeight": "300px",
        "display": "none",
        "padding-top": "10px",
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

button_layout = dmc.Flex(
    pt="xs",
    justify="space-between",
    align="center",
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
                        dmc.Checkbox(label="HDF5 time-series", id="download-hdf"),
                        dmc.Checkbox(label="BDF parquet time-series", id="download-bdf-parquet"),
                        dmc.Checkbox(label="JSON cycling summary", id="download-json-summary"),
                        dmc.Checkbox(label="JSON-LD ontologised metadata", id="download-jsonld"),
                    ]
                ),
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
            dcc.Store(id="upload-store", data={"file": None, "data": None}),
            dmc.Text("You can upload:"),
            dmc.List(
                [
                    dmc.ListItem("Samples as a .json file"),
                    dmc.ListItem("Data as a .zip - subfolders must be Sample ID"),
                    dmc.ListItem("BattINFO .xlsx and .jsonld and auxiliary .jsonld files"),
                    dmc.ListItem("Unicycler protocols as a .json file"),
                ]
            ),
            dcc.Upload(
                dmc.Button(
                    "Drag & drop or select file",
                    id="upload-data-button-element",
                    fullWidth=True,
                    style={"width": "100%"},
                ),
                id="upload-data-button",
                accept=".json,.zip,.xlsx,.jsonld",
                max_size=512 * 1024 * 1024,
                multiple=False,
                style={"display": "block", "width": "100%"},
            ),
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
    style={"height": "100%", "padding": "10px"},
    children=[
        html.Div(
            style={"height": "100%", "overflow": "auto"},
            children=[
                # Buttons to select which table to display
                dmc.Tabs(
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
                ),
                # Main table for displaying info from database
                html.Div(
                    id="table-container",
                    children=[
                        dcc.Clipboard(id="clipboard", style={"display": "none"}),
                        dcc.Store(id="selected-rows-store", data={}),
                        dcc.Store(id="len-store", data={}),
                        samples_table,
                        pipelines_table,
                        jobs_table,
                        results_table,
                        button_layout,  # Buttons along bottom of table
                    ],
                ),
                batch_edit_layout,  # When viewing 'batches' tab
                protocol_edit_layout,  # When viewing 'protocols' tab
            ],
        ),
        # Pop ups after clicking buttons
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
        [Output(element, "style") for element in CONTAINERS + BUTTONS],
        [Output(element, "style") for element in TABLES],
        Input("table-select", "value"),
    )
    def update_table(table: str) -> tuple:
        settings: set = visibility_settings.get(table, set())
        show: dict = {}
        hide = {"display": "none"}
        visibilities = [show if element in settings else hide for element in CONTAINERS + BUTTONS]
        table_style: dict = DEFAULT_TABLE_OPTIONS["style"]
        show_table = DEFAULT_TABLE_OPTIONS["style"].copy()
        show_table["display"] = "block"
        table_visibilities = [show_table if element == f"{table}-table" else table_style for element in TABLES]
        return (
            *visibilities,
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
        if sm and sm.servers:
            enabled |= {"upload-button"}
        if selected_rows:
            enabled |= {"copy-button"}
            if len(selected_rows) <= 200:  # To avoid enormous zip files being stored
                enabled |= {"download-button"}
            if sm and sm.servers:  # Need cycler permissions to do anything except copy, view, upload, download
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
            sm.eject(row["Sample ID"], row["Pipeline"])
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
            sm.load(sample, pipeline)
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
        logger.info("Updating metadata in cycles.*.json and full.*.h5 files")
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

    # If the file type is changed, allow reprocessing
    @app.callback(
        Output("download-process-button", "disabled", allow_duplicate=True),
        Output("download-yes-close", "disabled", allow_duplicate=True),
        Output("process-progress", "value", allow_duplicate=True),
        Output("download-alert", "children", allow_duplicate=True),
        Output("download-alert", "color", allow_duplicate=True),
        Input("download-button", "n_clicks"),
        Input("download-hdf", "checked"),
        Input("download-json-summary", "checked"),
        Input("download-bdf-parquet", "checked"),
        Input("download-jsonld", "checked"),
        prevent_initial_call=True,
    )
    def reset_process(_n_clicks: int, *inputs: tuple) -> tuple[bool, bool, int, str, str]:
        """If filetypes change, disable download, enable process, reset progress bar."""
        if any(inputs):
            return False, True, 0, "Waiting...", "grey"
        return True, True, 0, "Waiting...", "grey"

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
            State("download-hdf", "checked"),
            State("download-json-summary", "checked"),
            State("download-bdf-parquet", "checked"),
            State("download-jsonld", "checked"),
        ],
        running=[
            (Output("download-hdf", "disabled"), True, False),
            (Output("download-json-summary", "disabled"), True, False),
            (Output("download-bdf-parquet", "disabled"), True, False),
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
        download_hdf: bool | None,
        download_json: bool | None,
        download_bdf: bool | None,
        download_jsonld: bool | None,
    ) -> tuple[str, bool, bool]:
        """Batch process data from selected samples and store in zip in temp folder on server."""
        sample_ids = [s["Sample ID"] for s in selected_rows]
        ccids = [s["Barcode"] for s in selected_rows]

        # Remove old download files
        cleanup_temp_folder()

        rocrate = {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {
                    "@type": "CreativeWork",
                    "@id": "ro-crate-metadata.json",
                    "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                    "about": {"@id": "./"},
                },
                {
                    "@id": "./",
                    "@type": "Dataset",
                    "name": "Aurora Battery Assembly & Cycling Experiments",
                    "description": (
                        "A collection of battery assembly and cycling experiments. "
                        "Data processing, analysis, export, and ro-crate generation completed with "
                        "aurora-cycler-manager (https://github.com/empaeconversion/aurora-cycler-manager)"
                    ),
                    "dateCreated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "hasPart": [],
                },
            ],
        }

        # Number of files
        n_files = len(sample_ids) * (
            int(download_hdf or 0) + int(download_json or 0) + int(download_bdf or 0) + int(download_jsonld or 0)
        )
        i = 0
        set_progress((100 * i / n_files, "Initializing...", "Grey"))
        messages = ""
        color = "green"
        # Create a new zip archive to populate
        temp_zip = tempfile.NamedTemporaryFile(dir=DOWNLOAD_DIR, delete=False, suffix=".zip")  # noqa: SIM115
        with zipfile.ZipFile(temp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for sample, ccid in zip(sample_ids, ccids, strict=True):
                run_id = run_from_sample(sample)
                data_folder = CONFIG["Processed snapshots folder path"]
                sample_folder = str(data_folder / run_id / sample)
                messages += f"{sample} - "
                warnings = []
                errors = []
                if download_hdf:
                    i += 1
                    try:
                        hdf5_file = next(Path(sample_folder).glob("full.*.h5"))
                        with hdf5_file.open("rb") as f:
                            hdf5_bytes = f.read()
                        zf.writestr(sample + "/" + hdf5_file.name, hdf5_bytes)
                        rocrate["@graph"][1]["hasPart"].append({"@id": sample + "/" + hdf5_file.name})
                        rocrate["@graph"].append(
                            {
                                "@id": sample + "/" + hdf5_file.name,
                                "@type": "File",
                                "about": {"@id": ccid if ccid else sample},
                                "encodingFormat": "application/octet-stream",
                                "description": (
                                    f"Time-series battery cycling data for sample: '{sample}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "Data is in HDF5 format, keys are 'data' and 'metadata'. "
                                    "'data' contains an export from a Python Pandas dataframe. "
                                    "'metadata' contains a JSON-string with info about the sample and experiment."
                                ),
                            }
                        )
                        messages += "✅"
                        set_progress((100 * i / n_files, messages, color))
                    except StopIteration:
                        logger.warning("No HDF5 file found for %s", sample)
                        messages += "⚠️"
                        warnings.append("HDF5")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))
                    except Exception:
                        logger.exception("Unexpected error processing HDF5 for sample %s", sample)
                        messages += "⁉️"
                        errors.append("HDF5")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))

                if download_json:
                    i += 1
                    try:
                        json_file = next(Path(sample_folder).glob("cycles.*.json"))
                        with json_file.open("rb") as f:
                            json_bytes = f.read()
                        zf.writestr(sample + "/" + json_file.name, json_bytes)
                        messages += "✅"
                        rocrate["@graph"][1]["hasPart"].append({"@id": sample + "/" + json_file.name})
                        rocrate["@graph"].append(
                            {
                                "@id": sample + "/" + json_file.name,
                                "@type": "File",
                                "encodingFormat": "text/json",
                                "about": {"@id": ccid if ccid else sample},
                                "description": (
                                    f"Summary data from battery cycling for sample: '{sample}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "File is in JSON format, top level keys are 'data' and 'metadata'. "
                                    "'data' contains lists with per-cycle summary stastics, e.g. discharge capacity. "
                                    "'metadata' contains sample and experiment information."
                                ),
                            }
                        )
                        set_progress((100 * i / n_files, messages, color))
                    except StopIteration:
                        logger.warning("No JSON summary file found for %s", sample)
                        messages += "⚠️"
                        warnings.append("JSON summary")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))
                    except Exception:
                        logger.exception("Unexpected error processing JSON summary for sample %s", sample)
                        messages += "⁉️"
                        errors.append("JSON summary")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))

                if download_bdf:
                    i += 1
                    try:
                        hdf5_file = next(Path(sample_folder).glob("full.*.h5"))
                        df = pd.read_hdf(hdf5_file)
                        df = aurora_to_bdf(pd.DataFrame(df))
                        # convert to parquet file and write to zip
                        buffer = io.BytesIO()
                        df.to_parquet(buffer, index=False)  # or index=True, depending on your needs
                        buffer.seek(0)
                        parquet_name = hdf5_file.with_suffix(".bdf.parquet").name
                        zf.writestr(sample + "/" + parquet_name, buffer.read())
                        messages += "✅"
                        rocrate["@graph"][1]["hasPart"].append({"@id": sample + "/" + parquet_name})
                        rocrate["@graph"].append(
                            {
                                "@id": sample + "/" + parquet_name,
                                "@type": "File",
                                "encodingFormat": "application/octet-stream",
                                "about": {"@id": ccid if ccid else sample},
                                "description": (
                                    f"Time-series battery cycling data for sample: '{sample}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "Data is parquet format, columns are 'battery data format' (BDF) compliant."
                                ),
                            }
                        )
                        set_progress((100 * i / n_files, messages, color))
                    except StopIteration:
                        logger.warning("No HDF5 file found for %s", sample)
                        messages += "⚠️"
                        warnings.append("BDF")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))
                    except Exception:
                        logger.exception("Unexpected error processing BDF file for sample %s", sample)
                        messages += "⁉️"
                        errors.append("BDF")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))

                if download_jsonld:
                    i += 1
                    try:
                        battinfo_file = next(Path(sample_folder).glob("battinfo.*.jsonld"))
                        with battinfo_file.open("r") as f:
                            battinfo_json = json.load(f)
                        battinfo_json = make_test_object(battinfo_json)
                        aux_json = None
                        try:
                            aux_file = next(Path(sample_folder).glob("aux.*.jsonld"))
                            with aux_file.open("r") as f:
                                aux_json = json.load(f)
                        except StopIteration:
                            pass
                        if aux_json:
                            try:
                                merge_jsonld_on_type(battinfo_json, aux_json, target_type="BatteryTest")
                            except ValueError:
                                merge_jsonld_on_type(battinfo_json["hasTestObject"], aux_json, target_type="CoinCell")
                        db_jobs = get_unicycler_protocols(sample)
                        if db_jobs:
                            ontologized_protocols = []
                            for db_job in db_jobs:
                                protocol = Protocol.from_dict(json.loads(db_job["Unicycler protocol"]))
                                ontologized_protocols.append(
                                    protocol.to_battinfo_jsonld(capacity_mAh=db_job["Capacity (mAh)"])
                                )
                            test_jsonld = generate_battery_test(ontologized_protocols)
                            battinfo_json = merge_jsonld_on_type(battinfo_json, test_jsonld, target_type="BatteryTest")
                        jsonld_name = f"metadata.{sample}.jsonld"
                        zf.writestr(sample + "/" + jsonld_name, json.dumps(battinfo_json, indent=4))
                        messages += "✅"
                        rocrate["@graph"][1]["hasPart"].append({"@id": sample + "/" + jsonld_name})
                        rocrate["@graph"].append(
                            {
                                "@id": sample + "/" + jsonld_name,
                                "@type": "File",
                                "encodingFormat": "text/json",
                                "about": {"@id": ccid if ccid else sample},
                                "description": (
                                    f"Metadata for sample: '{sample}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "File is a BattINFO JSON-LD, describing the sample and experiment."
                                ),
                            }
                        )
                        set_progress((100 * i / n_files, messages, color))
                    except StopIteration:
                        logger.warning("No BattINFO JSON-LD file found for %s", sample)
                        messages += "⚠️"
                        warnings.append("JSON-LD")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))
                    except Exception:
                        logger.exception("Unexpected error processing JSON-LD file for sample %s", sample)
                        messages += "⁉️"
                        errors.append("JSON-LD")
                        color = "orange"
                        set_progress((100 * i / n_files, messages, color))
                messages += "\n"
                if warnings:
                    messages += "No data for " + ", ".join(warnings) + "\n"
                if errors:
                    messages += "Error processing " + ", ".join(errors) + "\n"
                set_progress((100 * i / n_files, messages, color))

            # Check if zipfile contains anything
            if zf.filelist:
                zf.writestr("ro-crate-metadata.json", json.dumps(rocrate, indent=4))
                logger.info("Saved zip file in server temp folder: %s", temp_zip.name)
                messages += "\n✅ ZIP file ready to download"
                set_progress((100 * i / n_files, messages, color))
                return (f"/download-temp/{temp_zip.name}", False, True)
            messages += "\n❌ No ZIP file created"
            set_progress((100 * i / n_files, messages, "red"))
            return ("", True, True)

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
    @app.callback(
        Output("upload-alert", "children"),
        Output("upload-alert", "color"),
        Output("upload-data-confirm-button", "disabled"),
        Output("upload-store", "data"),
        Input("upload-data-button", "contents"),
        State("upload-data-button", "filename"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def figure_out_files(contents: str, filename: str, selected_rows: list) -> tuple[str, str, bool, dict]:
        _content_type, content_string = contents.split(",")
        if not content_string:
            return "Nothing uploaded", "grey", True, {"file": None, "data": None}

        if filename.endswith((".jsonld", ".json")):
            # It could be ontology or samples
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded)
            if is_samples_json(data):
                samples = [d.get("Sample ID") for d in data]
                known_samples = set(get_all_sampleids())
                overwriting_samples = [s for s in samples if s in known_samples]
                if overwriting_samples:
                    return (
                        f"Got a samples json\nContains{len(samples)} samples\n"
                        f"WARNING - it will overwrite {len(overwriting_samples)} samples:\n"
                        + "\n".join(overwriting_samples),
                        "orange",
                        False,
                        {"file": "samples-json", "data": data},
                    )
                return (
                    f"Got a samples json\nContains {len(samples)} samples",
                    "green",
                    False,
                    {"file": "samples-json", "data": data},
                )

            samples = [s.get("Sample ID") for s in selected_rows if s.get("Sample ID")]
            if is_battinfo_jsonld(data):
                if not samples:
                    return (
                        "Got a BattINFO json-ld, but you must select samples.",
                        "red",
                        True,
                        {"file": None, "data": None},
                    )
                return (
                    "Got a BattINFO json-ld\n"
                    "The metadata will be merged with info from the database\n"
                    f"It will be applied to {len(samples)} samples:\n" + "\n".join(samples),
                    "green",
                    False,
                    {"file": "battinfo-jsonld", "data": data},
                )
            if is_aux_jsonld(data):
                if not samples:
                    return (
                        "Got an auxiliary json-ld, but you must select samples.",
                        "red",
                        True,
                        {"file": None, "data": None},
                    )
                return (
                    "Got an auxiliary json-ld\n"
                    "Each sample can have one auxiliary file that is merged when outputting\n"
                    f"I will apply it to {len(samples)} samples:\n" + "\n".join(samples),
                    "green",
                    False,
                    {"file": "aux-jsonld", "data": data},
                )
            if is_unicycler_protocol(data):
                jobs = [s.get("Job ID") for s in selected_rows if s.get("Job ID")]
                if not jobs:
                    return (
                        "Got a unicycler protocol, but you must select jobs.",
                        "red",
                        True,
                        {"file": None, "data": None},
                    )
                protocols = [s.get("Unicycler protocol") for s in selected_rows if s.get("Unicycler protocol")]
                if protocols:
                    return (
                        "Got a unicycler protocol.\nWARNING - this will overwrite data",
                        "orange",
                        False,
                        {"file": "unicycler-json", "data": data},
                    )
                return (
                    "Got a unicycler protocol.",
                    "green",
                    False,
                    {"file": "unicycler-json", "data": data},
                )

        elif filename.endswith(".xlsx"):
            # It is probably a battinfo xlsx file
            decoded = base64.b64decode(content_string)
            excel_file = pd.ExcelFile(io.BytesIO(decoded))
            sheet_names = [str(s) for s in excel_file.sheet_names]
            expected_sheets = ["Schema", "@context-TopLevel", "@context-Connector", "Ontology - Unit", "Unique ID"]
            if not all(sheet in expected_sheets for sheet in sheet_names):
                return (
                    "Excel file does not have the expected sheets"
                    "Found: " + ", ".join(sheet_names) + "\n"
                    "Expected: " + ", ".join(expected_sheets),
                    "red",
                    True,
                    {"file": None, "data": None},
                )
            samples = [s.get("Sample ID") for s in selected_rows]
            if not samples:
                return "Got a BattINFO xlsx, but you must select samples.", "red", True, {"file": None, "data": None}
            return (
                "Got a BattINFO xlsx\n"
                "The metadata will be merged with info from the database\n"
                f"It will be applied to {len(samples)} samples:\n" + "\n".join(samples),
                "green",
                False,
                {"file": "battinfo-xlsx", "data": None},  # Don't copy the content_string
            )

        elif filename.endswith(".zip"):
            # It is probably data, look for mpr and ndax files
            # Decode the base64 string
            decoded = base64.b64decode(content_string)

            # Wrap in BytesIO so we can use it like a file
            zip_buffer = io.BytesIO(decoded)

            # Open the zip archive
            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                # List the contents
                valid_files = {}
                invalid_files = {}
                file_list = zip_file.namelist()
                known_samples = set(get_all_sampleids())
                for file in file_list:
                    logger.info("Checking file %s", file)
                    if file.endswith("/"):
                        # Just a folder
                        continue
                    parts = file.split("/")
                    if len(parts) < 2:
                        invalid_files[file] = "File must be inside a Sample ID folder"
                        continue
                    folder = parts[-2]
                    if folder not in known_samples:
                        invalid_files[file] = f"Folder {folder} is not a Sample ID in the database"
                        continue
                    filetype = file.split(".")[-1]
                    if filetype not in ["mpr"]:
                        invalid_files[file] = f"Filetype {filetype} is not supported"
                        continue
                    if filetype in ["mpl"]:  # silently ignore - sidecar file
                        continue
                    valid_files[file] = folder
            if valid_files:
                msg = "Got a zip with valid files\nAdding data for the following samples:\n" + "\n".join(
                    sorted(set(valid_files.values()))
                )
                if not invalid_files:
                    return msg, "green", False, {"file": "zip", "data": valid_files}
                if invalid_files:
                    msg += "\n\nSKIPPING the following files:\n" + "\n".join(
                        file + "\n" + reason for file, reason in invalid_files.items()
                    )
                    return msg, "orange", False, {"file": "zip", "data": valid_files}
            if invalid_files:
                msg = "No valid files found:\n" + "\n".join(
                    file + "\n" + reason for file, reason in invalid_files.items()
                )
                return msg, "red", True, {"file": None, "data": None}
            return "No files found in zip", "red", False, {"file": None, "data": None}

        return "File not understood", "red", True, {"file": None, "data": None}

    def is_samples_json(obj: list | str | dict) -> bool:
        """Check if an uploaded json object is a samples file."""
        return isinstance(obj, list) and all(isinstance(s, dict) for s in obj) and all(s.get("Sample ID") for s in obj)

    def is_battinfo_jsonld(obj: list | str | dict) -> bool:
        """Check if an uploaded jsonld object is a battinfo file."""
        if isinstance(obj, dict) and obj.get("@context"):
            comments = obj.get("rdfs:comment")
            return isinstance(comments, list) and len(comments) >= 1 and comments[0].startswith("BattINFO")
        return False

    def is_aux_jsonld(obj: list | str | dict) -> bool:
        return isinstance(obj, dict) and bool(obj.get("@context")) and (not is_battinfo_jsonld(obj))

    def is_unicycler_protocol(obj: list | str | dict) -> bool:
        return isinstance(obj, dict) and bool(obj.get("unicycler"))

    # If you leave the upload modal, wipe the contents
    @app.callback(
        Output("upload-data-button", "contents"),
        Output("upload-data-button", "filename"),
        Input("upload-modal", "opened"),
        prevent_initial_call=True,
    )
    def wipe_upload(is_open: bool) -> tuple[str, str]:
        if not is_open:
            return ",", ""
        raise PreventUpdate

    # When hitting confirm, process the file
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("upload-data-confirm-button", "n_clicks"),
        State("upload-store", "data"),
        State("upload-data-button", "contents"),
        State("selected-rows-store", "data"),
        running=[
            (Output("loading-message-store", "data"), "Processing data...", ""),
            (Output("notify-interval", "interval"), active_time, idle_time),
        ],
        prevent_initial_call=True,
    )
    def process_file(n_clicks: int, data: dict, contents: str, selected_rows: list) -> int:
        if not n_clicks:
            raise PreventUpdate
        _content_type, content_string = contents.split(",")
        if not content_string:
            msg = "No contents"
            raise ValueError(msg)
        match data.get("file"):
            case "samples-json":
                logger.info("Adding samples from file")
                samples = data["data"]
                try:
                    logger.info("Adding samples %s", ", ".join(s.get("Sample ID") for s in samples))
                    add_samples_from_object(samples, overwrite=True)
                except Exception as e:
                    logger.exception("Error adding samples")
                    error_notification(
                        "Error adding samples",
                        f"{e!s}",
                        queue=True,
                    )
                    return 0
                success_notification(
                    "Samples added",
                    f"{len(samples)} added to database",
                    queue=True,
                )
                return 1

            case "battinfo-jsonld" | "battinfo-xlsx":
                try:
                    # If xlsx, first convert to dict
                    battinfo_jsonld = (
                        convert_excel_to_jsonld(io.BytesIO(base64.b64decode(content_string)))
                        if data["file"] == "battinfo-xlsx"
                        else data["data"]
                    )

                    # Merge json with database info and save
                    samples = [s.get("Sample ID") for s in selected_rows if s.get("Sample ID")]
                    for s in samples:
                        sample_data = get_sample_data(s)
                        merged_jsonld = merge_battinfo_with_db_data(battinfo_jsonld, sample_data)
                        run_id = run_from_sample(s)
                        save_path = CONFIG["Processed snapshots folder path"] / run_id / s / f"battinfo.{s}.jsonld"
                        logger.info("Saving battinfo json-ld file to %s", save_path)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with save_path.open("w", encoding="utf-8") as f:
                            json.dump(merged_jsonld, f, indent=4)
                    success_notification(
                        "BattINFO json-ld uploaded",
                        f"JSON-LD merged with data from {len(samples)} samples",
                        queue=True,
                    )
                except Exception as e:
                    logger.exception("Failed to convert, merge, save BattINFO json-ld")
                    error_notification(
                        "Error saving BattINFO json-ld",
                        f"{e!s}",
                        queue=True,
                    )
                return 1

            case "aux-jsonld":
                # No need to convert or add anything, just save the file
                try:
                    aux_jsonld = data["data"]
                    samples = [s.get("Sample ID") for s in selected_rows if s.get("Sample ID")]
                    for s in samples:
                        run_id = run_from_sample(s)
                        save_path = CONFIG["Processed snapshots folder path"] / run_id / s / f"aux.{s}.jsonld"
                        logger.info("Saving auxiliary json-ld to %s", save_path)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with save_path.open("w", encoding="utf-8") as f:
                            json.dump(aux_jsonld, f, indent=4)
                    success_notification(
                        "Aux json-ld uploaded",
                        f"Aux JSON-LD added to {len(samples)} samples",
                        queue=True,
                    )
                except Exception as e:
                    logger.exception("Failed to upload auxiliary json-ld")
                    error_notification(
                        "Error saving aux json-ld",
                        f"{e!s}",
                        queue=True,
                    )
                return 1

            case "unicycler-json":
                try:
                    protocol = data["data"]
                    Protocol.from_dict(protocol)
                    jobs = [s.get("Job ID") for s in selected_rows if s.get("Job ID")]
                    for job in jobs:
                        add_protocol_to_job(job, protocol)
                    success_notification(
                        "Protocols added",
                        f"Protocols added to {len(jobs)} jobs",
                        queue=True,
                    )
                except (ValueError, AttributeError, TypeError) as e:
                    logger.exception("Error processing and uploading unicycler protocol")
                    error_notification(
                        "Error adding protocol",
                        f"{e}",
                        queue=True,
                    )
                return 1

            case "zip":
                zip_buffer = io.BytesIO(base64.b64decode(content_string))
                valid_files = data["data"]
                successful_samples = set()
                with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                    for filepath, sample_id in valid_files.items():
                        filename = filepath.split("/")[-1]
                        try:
                            with zip_file.open(filepath) as file:
                                match filename.split(".")[-1]:
                                    case "mpr":
                                        # Check if there is an associated mpl file
                                        mpl_filename = filename.replace(".mpr", ".mpl")
                                        if mpl_filename in zip_file.namelist():
                                            with zip_file.open(mpl_filename) as f:
                                                mpl_file = f.read()
                                        else:
                                            mpl_file = None
                                        # Convert and save hdf from mpr file
                                        logger.info("Processing file: %s", filename)
                                        convert_mpr(
                                            file.read(),
                                            mpl_file=mpl_file,
                                            update_database=True,
                                            sample_id=sample_id,
                                            file_name=filename,
                                        )
                                        successful_samples.add(sample_id)
                                        success_notification(
                                            "File processed",
                                            f"{filename}",
                                            queue=True,
                                        )
                        except Exception as e:
                            logger.exception("Error processing file: %s", filename)
                            error_notification(
                                "Error processing file",
                                f"{filename}: {e!s}",
                                queue=True,
                            )

                for sample_id in successful_samples:
                    logger.info("Analysing sample: %s", sample_id)
                    analyse_sample(sample_id)
                    success_notification("Sample analysed", f"{sample_id}", queue=True)
                success_notification(
                    "All data processed and analysed",
                    f"{len(valid_files)} files added to database",
                    queue=True,
                )
                return 1

            case _:
                error_notification("Oh no", f"Could not understand filetype {data}")
                return 0
