"""Copyright © 2025, Empa.

Database view tab layout and callbacks for the visualiser app.
"""

import base64
import json
import logging
from datetime import datetime

import dash_ag_grid as dag
import dash_mantine_components as dmc
import paramiko
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate

from aurora_cycler_manager.analysis import update_sample_metadata
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import (
    add_samples_from_object,
    delete_samples,
    get_batch_details,
    update_sample_label,
)
from aurora_cycler_manager.server_manager import ServerManager
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
accessible_servers = []
database_access = False
sm: ServerManager | None = None
try:
    sm = ServerManager()
    accessible_servers = list(sm.servers.keys())
    database_access = bool(accessible_servers)
except (paramiko.SSHException, FileNotFoundError, ValueError) as e:
    logger.warning(e)
    logger.warning("You cannot access any servers. Running in view-only mode.")

# ----------------------------- Layout - tables ----------------------------- #

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
    "load-button",
    "eject-button",
    "ready-button",
    "unready-button",
    "submit-button",
    "cancel-button",
    "view-button",
    "snapshot-button",
    "create-batch-button",
    "delete-button",
    "add-samples-button",
    "label-button",
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
        "load-button",
        "eject-button",
        "ready-button",
        "unready-button",
        "submit-button",
        "cancel-button",
        "view-button",
        "snapshot-button",
        "label-button",
        "create-batch-button",
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
        "create-batch-button",
    },
    "samples": {
        "table-container",
        "view-button",
        "batch-button",
        "delete-button",
        "add-samples-button",
        "label-button",
        "create-batch-button",
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
                    "Ready",
                    leftSection=html.I(className="bi bi-play"),
                    id="ready-button",
                ),
                dmc.Button(
                    "Unready",
                    leftSection=html.I(className="bi bi-slash-circle"),
                    id="unready-button",
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
                dcc.Upload(
                    dmc.Button(
                        "Add samples",
                        leftSection=html.I(className="bi bi-database-add"),
                        id="add-samples-button-element",
                    ),
                    id="add-samples-button",
                    accept=".json",
                    max_size=2 * 1024 * 1024,
                    multiple=False,
                    style_disabled={"opacity": "1"},
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
                        disabled=not accessible_servers,
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
                    id="load-incrememt",
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

ready_modal = dmc.Modal(
    title="Ready",
    children=dmc.Stack(
        [
            dmc.Text("Are you sure you want to ready the selected pipelines?"),
            dmc.Text("You may need to update the database afterwards to check if jobs have started."),
            dmc.Button(
                "Ready",
                id="ready-yes-close",
            ),
        ]
    ),
    id="ready-modal",
    centered=True,
)

unready_modal = dmc.Modal(
    title="Unready",
    children=dmc.Stack(
        [
            dmc.Text("Are you sure you want to un-ready the selected pipelines?"),
            dmc.Button(
                "Unready",
                id="unready-yes-close",
            ),
        ]
    ),
    id="unready-modal",
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
        ]
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
        ]
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
        ]
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
        ]
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
        ]
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
        ]
    ),
    id="label-modal",
    centered=True,
)

# ------------------------------- Main layout -------------------------------- #


db_view_layout = html.Div(
    style={"height": "100%", "padding": "10px"},
    children=[
        # invisible div just to make the loading spinner work when no outputs are changed
        html.Div(
            id="loading-database",
            style={"display": "none"},
        ),
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
        ready_modal,
        unready_modal,
        submit_modal,
        cancel_modal,
        snapshot_modal,
        create_batch_modal,
        delete_modal,
        label_modal,
    ],
)


# -------------------------------- Callbacks --------------------------------- #


def register_db_view_callbacks(app: Dash) -> None:  # noqa: C901, PLR0915
    """Register callbacks for the database view layout."""
    register_batch_edit_callbacks(app, database_access)
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

    # Update the selected rows in the table
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
    def update_selected_rows(samples, pipelines, jobs, results, table, lens):
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
    def refresh_database(n_clicks, n_intervals):
        db_data = get_database()
        dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_checked = get_db_last_update()
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
        Input("update-database", "n_clicks"),
        running=[(Output("loading-message-store", "data"), "Updating databse - querying servers...", "")],
        prevent_initial_call=True,
    )
    def update_database(n_clicks):
        if n_clicks is None:
            raise PreventUpdate
        sm.update_db()
        return 1

    # Enable or disable buttons (load, eject, etc.) depending on what is selected in the table
    @app.callback(
        [Output(b, "disabled") for b in BUTTONS],
        Input("selected-rows-store", "data"),
        State("table-select", "value"),
        prevent_initial_call=True,
    )
    def enable_buttons(selected_rows, table):
        enabled = set()
        # Add buttons to enabled set with union operator |=
        if database_access and table == "samples":
            enabled |= {"add-samples-button"}
        if selected_rows:
            enabled |= {"copy-button"}
            if accessible_servers:  # Need cycler permissions to do anything except copy, view or upload
                if table == "samples":
                    if all(s.get("Sample ID") is not None for s in selected_rows):
                        enabled |= {"delete-button", "label-button", "create-batch-button"}
                elif table == "pipelines":
                    all_samples = all(s.get("Sample ID") is not None for s in selected_rows)
                    all_servers = all(s.get("Server label") in accessible_servers for s in selected_rows)
                    all_tomato = all(s.get("Server type") == "tomato" for s in selected_rows)
                    no_samples = all(s.get("Sample ID") is None for s in selected_rows)
                    if all_samples:
                        enabled |= {"label-button", "create-batch-button"}
                        if all_servers:
                            enabled |= {"submit-button", "snapshot-button"}
                            if all(s["Job ID"] is None for s in selected_rows):
                                enabled |= {"eject-button"}
                                if all_tomato:
                                    enabled |= {"ready-button", "unready-button"}
                            elif all(s.get("Job ID") is not None for s in selected_rows):
                                enabled |= {"cancel-button"}
                    elif all_servers and no_samples:
                        enabled |= {"load-button"}
                elif table == "jobs":
                    if all(s.get("Server label") in accessible_servers for s in selected_rows):
                        enabled |= {"snapshot-button"}
                        if all(s.get("Status") in ["r", "q", "qw"] for s in selected_rows):
                            enabled |= {"cancel-button"}
                elif table == "results":
                    if all(s.get("Sample ID") is not None for s in selected_rows):
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
    def copy_button(n, selected_rows, nclip):
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
    def eject_sample_button(eject_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "eject-button":
            return not is_open
        if button_id == "eject-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When eject button confirmed, eject samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("eject-yes-close", "n_clicks"),
        State("table-data-store", "data"),
        State("selected-rows-store", "data"),
        running=[(Output("loading-message-store", "data"), "Ejecting samples...", "")],
        prevent_initial_call=True,
    )
    def eject_sample(yes_clicks, data, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            logger.info("Ejecting %s", row["Pipeline"])
            sm.eject(row["Pipeline"])
        return no_update, 1

    # Load button pop up, includes dynamic dropdowns for selecting samples to load
    @app.callback(
        Output("load-modal", "opened"),
        Output("load-modal-text", "children"),
        Output("load-incrememt", "style"),
        Input("load-button", "n_clicks"),
        Input("load-yes-close", "n_clicks"),
        State("load-modal", "opened"),
        State("selected-rows-store", "data"),
        State("samples-store", "data"),
        prevent_initial_call=True,
    )
    def load_sample_button(load_clicks, yes_clicks, is_open, selected_rows, possible_samples):
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
        Input("load-incrememt", "n_clicks"),
        Input("load-clear", "n_clicks"),
        State({"type": "load-dropdown", "index": ALL}, "value"),
        State("samples-store", "data"),
        prevent_initial_call=True,
    )
    def update_load_selection(inc_clicks, clear_clicks, selected_samples, possible_samples):
        if not ctx.triggered:
            return selected_samples
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # If clear, clear all selected samples
        if button_id == "load-clear":
            return [None for _ in selected_samples]

        # If auto-increment, go through the list, if the sample is empty increment the previous sample
        if button_id == "load-incrememt":
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
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("load-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        State({"type": "load-dropdown", "index": ALL}, "value"),
        running=[(Output("loading-message-store", "data"), "Loading samples...", "")],
        prevent_initial_call=True,
    )
    def load_sample(yes_clicks, selected_rows, selected_samples):
        if not yes_clicks:
            return no_update, 0
        pipelines = [s["Pipeline"] for s in selected_rows]
        pipelines = [s for _, s in sorted(zip(make_pipelines_comparable(pipelines), pipelines, strict=True))]
        for sample, pipeline in zip(selected_samples, pipelines, strict=True):
            if not sample:
                continue
            logger.info("Loading %s to %s", sample, pipeline)
            sm.load(sample, pipeline)
        return no_update, 1

    # Ready button pop up
    @app.callback(
        Output("ready-modal", "opened"),
        Input("ready-button", "n_clicks"),
        Input("ready-yes-close", "n_clicks"),
        State("ready-modal", "opened"),
        prevent_initial_call=True,
    )
    def ready_pipeline_button(ready_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "ready-button":
            return True
        if button_id == "ready-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When ready button confirmed, ready pipelines and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("ready-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        running=[
            (Output("loading-message-store", "data"), "Readying pipelines...", ""),
            (Output("notify-interval", "interval"), active_time, idle_time),
        ],
        prevent_initial_call=True,
    )
    def ready_pipeline(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            sm.ready(row["Pipeline"])
            success_notification("", f"Pipeline {row['Pipeline']} ready", queue=True)
        return no_update, 1

    # Unready button pop up
    @app.callback(
        Output("unready-modal", "opened"),
        Input("unready-button", "n_clicks"),
        Input("unready-yes-close", "n_clicks"),
        State("unready-modal", "opened"),
        prevent_initial_call=True,
    )
    def unready_pipeline_button(unready_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "unready-button":
            return True
        if button_id == "unready-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When unready button confirmed, unready pipelines and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("unready-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        running=[(Output("loading-message-store", "data"), "Unreadying pipelines...", "")],
        prevent_initial_call=True,
    )
    def unready_pipeline(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            logger.info("Unreadying %s", row["Pipeline"])
            _output = sm.unready(row["Pipeline"])
        return no_update, 1

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
    def submit_pipeline_button(submit_clicks, yes_clicks, is_open, selected_rows):
        if not ctx.triggered:
            return no_update, no_update, no_update
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "submit-button":
            samples = [s.get("Sample ID") for s in selected_rows]
            capacities = {
                mode: {s: sm.safe_get_sample_capacity(s, mode) for s in samples}
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
    def check_payload(opened, filename, selected_rows):
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

        if any(s["Server type"] == "tomato" for s in selected_rows):
            # Should be a tomato JSON or unicycler JSON
            if not isinstance(payload, dict):
                return "❌ cannot submit .xml to tomato", {}
            if "tomato" not in payload and "unicycler" not in payload:
                msg = f"❌ {filename} is not a tomato json or unicycler json"
                return msg, {}

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
    def submit_custom_crate(crate, capacities):
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
    def enable_submit(payload, crate, capacity, capacity_vals):
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
        Output("loading-database", "children", allow_duplicate=True),
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
    def submit_pipeline(yes_clicks, selected_rows, payload, crate_calc, capacity):
        if not yes_clicks:
            return no_update, 0
        # capacity_Ah: float | 'areal','mass','nominal'
        capacity_Ah = capacity / 1000 if crate_calc == "custom" else crate_calc
        if not isinstance(capacity_Ah, float) and capacity_Ah not in ["areal", "mass", "nominal"]:
            logger.error("Invalid capacity calculation method: %s", capacity_Ah)
            return no_update, 0
        for row in selected_rows:
            try:
                sm.submit(row["Sample ID"], payload, capacity_Ah)
                success_notification("", f"Sample {row['Sample ID']} submitted", queue=True)
            except Exception as e:  # noqa: BLE001, PERF203
                error_notification("", f"Error submitting sample {row['Sample ID']}: {e}", queue=True)
        return no_update, 1

    # When selecting create batch, switch to batch sub-tab with samples selected
    @app.callback(
        Output("table-select", "value"),
        Output("create-batch-store", "data", allow_duplicate=True),
        Input("create-batch-button", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def create_batch(n_clicks, selected_rows):
        return "batches", [s.get("Sample ID") for s in selected_rows]

    # Cancel button pop up
    @app.callback(
        Output("cancel-modal", "opened"),
        Input("cancel-button", "n_clicks"),
        Input("cancel-yes-close", "n_clicks"),
        State("cancel-modal", "opened"),
        prevent_initial_call=True,
    )
    def cancel_job_button(cancel_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "cancel-button":
            return not is_open
        if button_id == "cancel-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When cancel confirmed, cancel the jobs and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("cancel-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def cancel_job(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        for row in selected_rows:
            logger.info("Cancelling job %s", row["Job ID"])
            sm.cancel(row["Job ID"])
        return no_update, 1

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
    def view_data(n_clicks, selected_rows):
        if not n_clicks or not selected_rows:
            raise PreventUpdate
        sample_id = [s["Sample ID"] for s in selected_rows]
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
    def snapshot_sample_button(snapshot_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "snapshot-button":
            return not is_open
        if button_id == "snapshot-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When snapshot confirmed, snapshot the samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Input("snapshot-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        running=[(Output("loading-message-store", "data"), "Snapshotting data...", "")],
        prevent_initial_call=True,
    )
    def snapshot_sample(yes_clicks, selected_rows):
        if not yes_clicks:
            raise PreventUpdate
        for row in selected_rows:
            if row.get("Job ID"):
                logger.info("Snapshotting %s", row["Job ID"])
                sm.snapshot(row["Job ID"])
            else:
                logger.info("Snapshotting %s", row["Sample ID"])
                sm.snapshot(row["Sample ID"])
        raise PreventUpdate

    # Delete button pop up
    @app.callback(
        Output("delete-modal", "opened"),
        Input("delete-button", "n_clicks"),
        Input("delete-yes-close", "n_clicks"),
        State("delete-modal", "opened"),
        prevent_initial_call=True,
    )
    def delete_sample_button(delete_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "delete-button":
            return not is_open
        if button_id == "delete-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When delete confirmed, delete the samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("delete-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def delete_sample(yes_clicks, selected_rows):
        if not yes_clicks:
            return no_update, 0
        sample_ids = [s["Sample ID"] for s in selected_rows]
        logger.info("Deleting [%s]", ", ".join(sample_ids))
        delete_samples(sample_ids)
        return no_update, 1

    # Add samples button pop up
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Output("add-samples-confirm", "displayed"),
        Input("add-samples-button", "contents"),
        State("add-samples-button", "filename"),
        prevent_initial_call=True,
    )
    def upload_samples(contents, filename):
        if contents:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            samples = json.loads(decoded)
            logger.info("Adding samples %s", filename)
            try:
                add_samples_from_object(samples)
            except ValueError as e:
                if "already exist" in str(e):
                    logger.warning("Sample upload would overwrite existing samples")
                    return no_update, True  # Open confirm dialog
            return 1, no_update  # Refresh the database
        raise PreventUpdate

    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("add-samples-confirm", "submit_n_clicks"),
        State("add-samples-button", "contents"),
        State("add-samples-button", "filename"),
        prevent_initial_call=True,
    )
    def upload_overwrite_samples(submit_n_clicks, contents, filename):
        if submit_n_clicks and contents:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            samples = json.loads(decoded)
            logger.info("Adding samples %s", filename)
            add_samples_from_object(samples, overwrite=True)
            return 1
        raise PreventUpdate

    # Label button pop up
    @app.callback(
        Output("label-modal", "opened"),
        Input("label-button", "n_clicks"),
        Input("label-yes-close", "n_clicks"),
        State("label-modal", "opened"),
        prevent_initial_call=True,
    )
    def label_sample_button(label_clicks, yes_clicks, is_open):
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "label-button":
            return not is_open
        if button_id == "label-yes-close" and yes_clicks:
            return False
        return is_open, no_update, no_update, no_update

    # When label confirmed, label the samples and refresh the database
    @app.callback(
        Output("loading-database", "children", allow_duplicate=True),
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("label-yes-close", "n_clicks"),
        State("selected-rows-store", "data"),
        State("label-input", "value"),
        prevent_initial_call=True,
    )
    def label_sample(yes_clicks, selected_rows, label):
        if not yes_clicks:
            return no_update, 0
        sample_ids = [s["Sample ID"] for s in selected_rows]
        logger.info("Labelling [%s] with '%s'", ", ".join(sample_ids), label)
        update_sample_label(sample_ids, label)
        logger.info("Updating metadata in cycles.*.json and full.*.h5 files")
        update_sample_metadata(sample_ids)
        return no_update, 1

    # Synchronise add-samples button disabled state
    @app.callback(
        Output("add-samples-button-element", "disabled"),
        Input("add-samples-button", "disabled"),
        prevent_initial_call=True,
    )
    def disabled_sync(disabled):
        return disabled
