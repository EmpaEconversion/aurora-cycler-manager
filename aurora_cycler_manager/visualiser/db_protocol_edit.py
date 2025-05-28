"""Batch edit sub-layout for the database tab."""

import base64
import json
import logging
import uuid
from decimal import Decimal

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash_ag_grid import AgGrid
from pydantic import ValidationError

from aurora_cycler_manager.unicycler import (
    BaseTechnique,
    ConstantCurrent,
    ConstantVoltage,
    Loop,
    OpenCircuitVoltage,
    Tag,
    from_dict,
)
from aurora_cycler_manager.visualiser.notifications import error_notification

logger = logging.getLogger(__name__)

TECHNIQUE_NAMES = {
    "constant_current": "Constant current",
    "constant_voltage": "Constant voltage",
    "open_circuit_voltage": "Open circuit voltage",
    "loop": "Loop",
    "tag": "Tag",
}
ALL_TECHNIQUES = {
    "Constant current": ConstantCurrent,
    "Constant voltage": ConstantVoltage,
    "Open circuit voltage": OpenCircuitVoltage,
    "Loop": Loop,
    "Tag": Tag,
}
ALL_TECHNIQUES_REV = {v: k for k, v in ALL_TECHNIQUES.items()}
ALL_TECHNIQUE_INPUTS = {k for v in ALL_TECHNIQUES.values() for k in v.model_fields}
ALL_TECHNIQUE_INPUTS.remove("name")
ALL_TECHNIQUE_INPUTS.remove("id")

column_defs = [
    {
        "headerName": "Technique",
        "field": "technique",
        "rowDrag": True,
        "width": 200,
        "resizable": True,
        "cellRenderer": "agAnimateShowChangeCellRenderer",
    },
    {
        "headerName": "Description",
        "field": "description",
        "flex": 1,  # Expands to fill space
        "cellRenderer": "agAnimateShowChangeCellRenderer",
    },
    {
        "headerName": "Index",
        "field": "index",
        "hide": True,
    },
    {
        "headerName": "id",
        "field": "id",
        "hide": True,
    },
    {
        "headerName": "loop",
        "field": "loop",
        "hide": True,
    },
]


def seconds_to_time(seconds: float | Decimal | None) -> str:
    """Convert seconds to a time string."""
    if seconds is None:
        return "forever"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds_string = []
    if hours > 0:
        seconds_string.append(f"{hours:.0f} h")
    if minutes > 0:
        seconds_string.append(f"{minutes:.0f} min")
    if seconds > 0:
        seconds_string.append(f"{seconds:.0f} s")
    return " ".join(seconds_string) if seconds_string else "0 s"


def describe_row(technique: dict) -> str:
    """Generate a description for a row based on the technique."""
    name = technique.get("name")
    if not name:
        description = "Select technique"
    elif name == "constant_current":
        conditions = []
        if voltage := technique.get("until_voltage_V") is not None:
            conditions.append(f"until {voltage} V")
        if (time := technique.get("until_time_s")) is not None and time > 0:
            conditions.append(f"until {seconds_to_time(time)}")
        description = f"{technique.get('rate_C')} C " + " or ".join(conditions)
    elif name == "constant_voltage":
        conditions = []
        if c_rate := technique["until_rate_C"] is not None:
            conditions.append(f"until {c_rate} C")
        elif current := technique.get("until_current_mA") is not None:
            conditions.append(f"until {current} mA")
        if (time := technique.get("until_time_s")) is not None and time > 0:
            conditions.append(f"until {seconds_to_time(time)}")
        description = f"{technique.get('voltage_V')} V " + " or ".join(conditions)
    elif name == "open_circuit_voltage":
        if (time := technique.get("until_time_s")) is not None and time > 0:
            description = f"until {seconds_to_time(time)}"
    elif name == "loop":
        start_step = technique.get("start_step")
        start_step_str = f"'{start_step}'" if isinstance(start_step, str) else f"technqiue {start_step} (1-indexed)"
        description = f"to {start_step_str} for {technique.get('cycle_count')} cycles"
    elif name == "tag":
        description = f"{technique.get('tag')}"
    else:
        description = "Select technique"
    return description


def protocol_dict_to_row_data(protocol: dict) -> list[dict]:
    """Convert protocol dict to row data for the ag grid."""
    return [
        {
            "index": i,
            "id": technique.get("id"),
            "technique": TECHNIQUE_NAMES.get(technique.get("name", "...")),
            "description": describe_row(technique),
        }
        for i, technique in enumerate(protocol.get("method", []))
    ]


protocol_edit_grid = AgGrid(
    id="protocol-edit-grid",
    columnDefs=column_defs,
    rowData=[],
    getRowId="params.data.id",
    defaultColDef={
        "editable": False,
        "sortable": False,
        "filter": False,
        "resizable": True,
        "cellRenderer": "agAnimateShowChangeCellRenderer",
    },
    dashGridOptions={
        "rowDragManaged": True,
        "rowSelection": "multiple",
        "animateRows": True,
        "rowDragMultiRow": True,
    },
    style={"height": "calc(100vh - 300px)", "width": "100%", "minHeight": "300px"},
)

protocol_edit_buttons = html.Div(
    style={"display": "flex", "alignItems": "left", "width": "100%"},
    children=[
        dbc.Button(
            html.I(className="bi bi-plus-circle", style={"fontSize": "1.5em"}),
            id="add-row-button",
            color="success",
            size="sm",
            className="me-1",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "40px",
                "height": "40px",
            },
        ),
        dbc.Button(
            html.I(className="bi bi-dash-circle", style={"fontSize": "1.5em"}),
            id="remove-row-button",
            color="danger",
            size="sm",
            className="me-1",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "40px",
                "height": "40px",
            },
        ),
        # copy button
        dbc.Button(
            html.I(className="bi bi-copy", style={"fontSize": "1.5em"}),
            id="copy-rows-button",
            color="primary",
            size="sm",
            className="me-1",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "40px",
                "height": "40px",
            },
        ),
        # paste button
        dbc.Button(
            html.I(className="bi bi-clipboard-plus", style={"fontSize": "1.5em"}),
            id="paste-rows-button",
            color="primary",
            size="sm",
            className="me-1",
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "40px",
                "height": "40px",
            },
        ),
    ],
)

protocol_header = html.Div(
    [
        dbc.Input(
            id="protocol-name",
            type="text",
            placeholder="Protocol name",
        ),
        html.Div(
            [
                dbc.Button(
                    html.I(className="bi bi-exclamation-triangle-fill", style={"fontSize": "1.5em"}),
                    id="protocol-warning",
                    color="danger",
                    size="sm",
                    className="me-1",
                ),
                dbc.Popover(
                    "This explains why your thing is wrong!",
                    id="protocol-warning-message",
                    target="protocol-warning",
                    body=True,
                    trigger="focus",
                    placement="bottom",
                ),
            ],
            id="protocol-warning-group",
            style={"visibility": "hidden"},
        ),
    ],
    style={"display": "flex", "alignItems": "left", "width": "100%"},
)

# Load/save protocol buttons
load_protocol_button = dcc.Upload(
    dbc.Button(
        [html.I(className="bi bi-folder2-open me-2"), "Load protocol"],
        id="load-protocol-button-element",
        color="primary",
        className="me-1",
    ),
    id="load-protocol-button",
    accept=".json",
    max_size=2 * 1024 * 1024,
    multiple=False,
    style_disabled={"opacity": "1"},
)

# Input fields must have the same id as the inputs in ALL_TECHNIQUE_INPUTS
# and be inside a div with the id "{input_name}-group", used to hide inputs
step_edit_menu = html.Div(
    id="step-edit-menu",
    children=[
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Technqiue"),
                    dbc.Select(
                        id="technique-select",
                        options=[{"label": k, "value": k} for k in ALL_TECHNIQUES],
                        placeholder="Select technique",
                    ),
                ],
                className="mb-3",
            ),
            id="technique-select-group",
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Rate"),
                    dbc.Input(
                        id="rate_C",
                        type="number",
                    ),
                    dbc.InputGroupText("C"),
                ],
                className="mb-3",
            ),
            id="rate_C-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Current"),
                    dbc.Input(
                        id="current_mA",
                        type="number",
                        style={"width": "100px"},
                    ),
                    dbc.InputGroupText("mA"),
                ],
                className="mb-3",
            ),
            id="current_mA-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Voltage"),
                    dbc.Input(
                        id="voltage_V",
                        type="number",
                    ),
                    dbc.InputGroupText("V"),
                ],
                className="mb-3",
            ),
            id="voltage_V-group",
            style={"display": "none"},
        ),
        html.Div(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Until time"),
                        dbc.Input(
                            id="input_time_h",
                            type="number",
                            debounce=True,
                        ),
                        dbc.InputGroupText("h"),
                        dbc.Input(
                            id="input_time_m",
                            type="number",
                            debounce=True,
                        ),
                        dbc.InputGroupText("min"),
                        dbc.Input(
                            id="input_time_s",
                            type="number",
                            debounce=True,
                        ),
                        dbc.InputGroupText("s"),
                    ],
                    className="mb-3",
                ),
                dbc.Input(  # for storing the actual time
                    id="until_time_s",
                    type="number",
                    style={"display": "none"},
                ),
            ],
            id="until_time_s-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Until voltage"),
                    dbc.Input(
                        id="until_voltage_V",
                        type="number",
                    ),
                    dbc.InputGroupText("V"),
                ],
                className="mb-3",
            ),
            id="until_voltage_V-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Until rate"),
                    dbc.Input(
                        id="until_rate_C",
                        type="number",
                    ),
                    dbc.InputGroupText("C"),
                ],
                className="mb-3",
            ),
            id="until_rate_C-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Until current"),
                    dbc.Input(
                        id="until_current_mA",
                        type="number",
                    ),
                    dbc.InputGroupText("mA"),
                ],
                className="mb-3",
            ),
            id="until_current_mA-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Go to step"),
                    dbc.Input(
                        id="start_step",
                        type="text",
                        placeholder="Start step",
                    ),
                ],
                className="mb-3",
            ),
            id="start_step-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("For"),
                    dbc.Input(
                        id="cycle_count",
                        type="number",
                        placeholder="Cycle count",
                    ),
                    dbc.InputGroupText("cycles"),
                ],
                className="mb-3",
            ),
            id="cycle_count-group",
            style={"display": "none"},
        ),
        html.Div(
            dbc.InputGroup(
                [
                    dbc.InputGroupText("Tag"),
                    dbc.Input(
                        id="tag",
                        type="text",
                    ),
                ],
                className="mb-3",
            ),
            id="tag-group",
            style={"display": "none"},
        ),
        html.Div(
            [
                dbc.Button("Update", id="submit", className="me-1"),
                html.Div(
                    [
                        dbc.Button(
                            html.I(className="bi bi-exclamation-triangle-fill", style={"fontSize": "1.5em"}),
                            id="step-warning",
                            color="danger",
                            size="sm",
                            className="me-1",
                        ),
                        dbc.Popover(
                            "This explains why your thing is wrong!",
                            id="step-warning-message",
                            target="step-warning",
                            body=True,
                            trigger="focus",
                            placement="bottom",
                        ),
                    ],
                    id="step-warning-group",
                    style={"display": "none"},
                ),
            ],
            style={"display": "flex", "alignItems": "left"},
        ),
    ],
)

protocol_edit_layout = html.Div(
    id="protocol-container",
    children=[
        dcc.Store(id="protocol-store", data={"protocol": {"method": []}, "selected": []}),
        dcc.Store(id="protocol-edit-clipboard", data=[]),  # For copy/paste functionality
        html.Div(
            style={"display": "flex", "height": "100%"},
            children=[
                html.Div(
                    style={"width": "800px", "padding": "10px"},
                    children=[
                        load_protocol_button,
                        html.Div(style={"height": "20px"}),
                        step_edit_menu,
                    ],
                ),
                html.Div(
                    style={"width": "100%", "padding": "10px"},
                    children=[
                        protocol_header,
                        html.Div(style={"height": "20px"}),
                        protocol_edit_grid,
                        protocol_edit_buttons,
                    ],
                ),
            ],
        ),
    ],
)


### Callbacks ###
def register_protocol_edit_callbacks(app: Dash) -> None:  # noqa: C901, PLR0915
    """Register callbacks for the protocol edit tab."""

    # If the data changes, update the grid
    @app.callback(
        Output("protocol-edit-grid", "selectedRows"),
        Output("protocol-edit-grid", "rowData"),
        Input("protocol-store", "data"),
        prevent_initial_call=True,
    )
    def update_grid(protocol_store: dict) -> tuple[list[dict], list[dict]]:
        """Update the grid with the new data."""
        protocol_dict = protocol_store.get("protocol")
        selected_indices = protocol_store.get("selected", [])
        if protocol_dict is None or not protocol_dict:  # return empty grid data
            return [], []
        row_data = protocol_dict_to_row_data(protocol_dict)
        new_selected_rows = [row_data[i] for i in selected_indices] if selected_indices else []
        return new_selected_rows, row_data

    # Load a protocol button
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Output("protocol-name", "value"),
        Output("notifications-container", "children", allow_duplicate=True),
        Input("load-protocol-button", "contents"),
        State("load-protocol-button", "filename"),
        prevent_initial_call=True,
    )
    def load_protocol(contents: str, filename: str) -> tuple:
        if contents:
            _content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            samples = json.loads(decoded)
            try:
                protocol = from_dict(samples).model_dump()
                # add an id to each technique
                for technique in protocol["method"]:
                    technique["id"] = uuid.uuid4()
                return {"protocol": protocol, "selected": []}, filename[:-5], no_update
            except ValidationError:
                return no_update, no_update, error_notification("Oh no", "This is not a valid protocol file!")
        raise PreventUpdate

    # Remove rows button
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Input("remove-row-button", "n_clicks"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        State("protocol-edit-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def remove_rows(
        n_clicks: int,
        protocol_store: dict,
        grid_data: list[dict],
        selected_rows: list[dict],
    ) -> dict:
        """Remove the selected rows from the data store."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        if selected_rows is None or not selected_rows:
            raise PreventUpdate
        # Get the new indices, remove if the index is in the selected indices
        selected_indices = [row["index"] for row in selected_rows]
        indices = [row["index"] for row in grid_data if row["index"] not in selected_indices]
        protocol_dict = protocol_store.get("protocol", {})
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # update the store with the new data
        return {"protocol": protocol_dict, "selected": []}

    # Add rows button
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Input("add-row-button", "n_clicks"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        State("protocol-edit-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def add_row(n_clicks: int, protocol_store: dict, grid_data: list[dict], selected_rows: list[dict]) -> dict:
        """Add a new row to the data store."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        # reorder the techniques if the user has dragged rows around
        indices = [row["index"] for row in grid_data]
        protocol_dict = protocol_store.get("protocol", {})
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # get the largest selected index
        index = None
        if selected_rows is not None and selected_rows:
            selected_indices = [s["index"] for s in selected_rows]
            reordered_indices = [i for i, row in enumerate(indices) if row in selected_indices]
            index = min(reordered_indices) if reordered_indices else None
        # add a new row to the data store
        new_row = BaseTechnique(name="Select technique").model_dump()
        new_row["id"] = uuid.uuid4()
        if index is not None:
            protocol_dict["method"].insert(index, new_row)
        else:
            protocol_dict["method"].append(new_row)
            index = len(protocol_dict["method"]) - 1
        # update the store with the new data
        return {"protocol": protocol_dict, "selected": [index]}

    # On 'copy', store the corresponding data rows in the clipboard
    @app.callback(
        Output("protocol-edit-clipboard", "data"),
        Input("copy-rows-button", "n_clicks"),
        State("protocol-edit-grid", "selectedRows"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def copy_rows(n_clicks: int, selected_rows: list[dict], protocol_store: dict, grid_data: list) -> list[dict]:
        """Copy the selected rows to the clipboard."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        if selected_rows is None or not selected_rows:
            raise PreventUpdate
        # Get the indices of the selected rows
        selected_indices = [row["index"] for row in selected_rows]
        virtual_indices = [row["index"] for row in grid_data]
        # Get the new indicies of those rows, in case user has dragged rows around
        reordered_indices = [i for i, row in enumerate(virtual_indices) if row in selected_indices]
        # Get the order of these new indices
        ordering = [index for index, _ in sorted(enumerate(reordered_indices), key=lambda x: x[1])]
        # Reorder the original selected indices so they copy paste in the correct order
        indices = [selected_indices[i] for i in ordering]
        # Get the actual list of techniques from the protocol store, store this
        protocol_dict = protocol_store.get("protocol", {})
        return [protocol_dict["method"][i] for i in indices]

    # On 'paste', add the rows from the clipboard to the protocol
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Input("paste-rows-button", "n_clicks"),
        State("protocol-edit-clipboard", "data"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        State("protocol-edit-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def paste_rows(
        n_clicks: int,
        clipboard_data: list[dict],
        protocol_store: dict,
        grid_data: list[dict],
        selected_rows: list[dict],
    ) -> dict:
        """Paste the rows from the clipboard to the protocol."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        if not clipboard_data:
            raise PreventUpdate
        # reorder the techniques if the user has dragged rows around
        indices = [row["index"] for row in grid_data]
        protocol_dict = protocol_store.get("protocol", {})
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # get the largest selected index
        index = None
        if selected_rows is not None and selected_rows:
            selected_indices = [s["index"] for s in selected_rows]
            reordered_indices = [i for i, row in enumerate(indices) if row in selected_indices]
            index = max(reordered_indices) if reordered_indices else None
        # insert the clipboard data into the protocol
        selected_indices = []
        for row in clipboard_data:
            # insert the technique into the protocol
            if index is not None:
                index += 1
                row["id"] = uuid.uuid4()
                protocol_dict["method"].insert(index, row)
                selected_indices.append(index)
            else:
                row["id"] = uuid.uuid4()
                protocol_dict["method"].append(row)
                selected_indices.append(len(protocol_dict["method"]) - 1)
        return {"protocol": protocol_dict, "selected": selected_indices}

    # If user selects a row, show it in the step edit menu
    @app.callback(
        Output("technique-select", "value"),
        [Output(x, "value", allow_duplicate=True) for x in ALL_TECHNIQUE_INPUTS],
        Input("protocol-edit-grid", "selectedRows"),
        State("protocol-store", "data"),
        prevent_initial_call=True,
    )
    def update_step_edit_menu(selected_rows: list[dict], protocol_store: dict) -> tuple[str, ...]:
        """Update the step edit menu with the selected row data."""
        if selected_rows is None or not selected_rows:
            raise PreventUpdate
        selected_row = selected_rows[0]
        index = selected_row["index"]
        protocol_dict = protocol_store.get("protocol", {})
        technique = protocol_dict["method"][index]
        input_values = [technique.get(x, None) for x in ALL_TECHNIQUE_INPUTS]
        return selected_row["technique"], *input_values

    # If user selects a technique, show the inputs for that technique
    @app.callback(
        [Output(x + "-group", "style") for x in ALL_TECHNIQUE_INPUTS],
        Input("technique-select", "value"),
    )
    def update_technique_inputs(technique: str) -> list[dict]:
        """Update the input fields based on the selected technique."""
        show = {"display": "block"}
        hide = {"display": "none"}
        if technique not in ALL_TECHNIQUES:
            return [hide for _ in ALL_TECHNIQUE_INPUTS]
        return [show if x in ALL_TECHNIQUES[technique].model_fields else hide for x in ALL_TECHNIQUE_INPUTS]

    # If user changes time, update the 'real' total time in seconds
    @app.callback(
        Output("until_time_s", "value", allow_duplicate=True),
        Input("input_time_h", "value"),
        Input("input_time_m", "value"),
        Input("input_time_s", "value"),
        prevent_initial_call=True,
    )
    def update_until_time_s(hours: float, mins: float, secs: float) -> float:
        """Update the total time in seconds based on the input fields. Ensure no nones or negatives."""
        hours = max(hours or 0, 0)
        mins = max(mins or 0, 0)
        secs = max(secs or 0, 0)
        return hours * 3600 + mins * 60 + secs

    # If the real time input changes, update the hour/min/sec inputs
    @app.callback(
        Output("input_time_h", "value"),
        Output("input_time_m", "value"),
        Output("input_time_s", "value"),
        Input("until_time_s", "value"),
        prevent_initial_call=True,
    )
    def update_time_inputs(until_time_s: float) -> tuple[float, float, float]:
        """Update the time inputs based on the until_time_s value."""
        if until_time_s is None:
            return 0, 0, 0
        hours = int(until_time_s // 3600)
        minutes = int((until_time_s % 3600) // 60)
        seconds = int(until_time_s % 60)
        return hours, minutes, seconds

    # if you change a value in the step edit menu, check if the technique is valid
    @app.callback(
        Output("step-warning-group", "style"),
        Output("step-warning-message", "children"),
        Output("submit", "disabled"),
        Input("technique-select", "value"),
        [Input(x, "value") for x in ALL_TECHNIQUE_INPUTS],
    )
    def validate_step(technique: str, *input_values: list[str | float | None]) -> tuple[dict, str, bool]:
        # If no valid technique, don't validate
        if not technique:
            raise PreventUpdate
        technique_cls = ALL_TECHNIQUES.get(technique)
        if technique_cls is None:
            raise PreventUpdate
        try:
            technique_cls(
                **{
                    name: value
                    for name, value in zip(ALL_TECHNIQUE_INPUTS, input_values)
                    if name in technique_cls.model_fields
                },
            )
        except ValidationError as e:
            logger.exception("Pydantic validation error of individual technique")
            friendly_error = str(e).split("\n", 1)[1] if "\n" in str(e) else str(e)
            friendly_error = friendly_error.split("[", 1)[0].strip()
            return {"display": "block"}, friendly_error, True
        return {"display": "none"}, "", False

    # If user changes value in the step edit menu and presses submit, update the protocol store
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Input("submit", "n_clicks"),
        State("protocol-edit-grid", "selectedRows"),
        State("protocol-edit-grid", "virtualRowData"),
        State("protocol-store", "data"),
        State("technique-select", "value"),
        [State(x, "value") for x in ALL_TECHNIQUE_INPUTS],
        prevent_initial_call=True,
    )
    def sync_protocol_dict(
        n_clicks: int,
        selected_rows: list[dict],
        grid_data: list[dict],
        protocol_store: dict,
        technique: str,
        *input_values: list[str | float | None],
    ) -> dict:
        """Update the protocol store with the new data."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        index = selected_rows[0]["index"] if selected_rows else None
        technique_id = selected_rows[0]["id"] if selected_rows else None
        technique_cls = ALL_TECHNIQUES.get(technique)
        if technique_cls is None:
            raise PreventUpdate  # no valid technique selected
        new_technique = technique_cls(
            **{
                name: value
                for name, value in zip(ALL_TECHNIQUE_INPUTS, input_values)
                if name in technique_cls.model_fields
            },
        ).model_dump()
        new_technique["id"] = technique_id if technique_id else uuid.uuid4()

        protocol_dict = protocol_store.get("protocol", {})
        # If a technique is selected (index not None), update that technique
        if index is not None:
            protocol_dict["method"][index] = new_technique
        # Reorder the techniques in case the user has dragged rows around
        indices = [row["index"] for row in grid_data] if grid_data else []
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        selected_indices = [row["index"] for row in selected_rows] if selected_rows else []
        new_selected_indicies = [i for i, index in enumerate(indices) if index in selected_indices]

        # If no technique was selected, append the new technique to the end after reordering
        if index is None:
            if isinstance(protocol_dict["method"], list):
                protocol_dict["method"].append(new_technique)
            elif not protocol_dict["method"]:
                protocol_dict["method"] = [new_technique]

        return {"protocol": protocol_dict, "selected": new_selected_indicies}

    # If the virtual data changes (dragging, updating data), check if protocol is valid
    @app.callback(
        Output("protocol-warning-message", "children"),
        Output("protocol-warning-group", "style"),
        Input("protocol-edit-grid", "virtualRowData"),
        State("protocol-store", "data"),
        prevent_initial_call=True,
    )
    def validate_protocol(grid_data: list[dict], protocol_store: dict) -> tuple[str, dict]:
        """Validate the protocol and update the grid data."""
        protocol_dict = protocol_store.get("protocol", {})
        # Reorder the techniques in case the user has dragged rows around
        indices = [row["index"] for row in grid_data] if grid_data else []
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # Validate the protocol
        try:
            from_dict(protocol_dict)
        except ValidationError as e:
            logger.error("Pydantic validation error for whole protocol: %s", e)  # noqa: TRY400
            friendly_error = str(e).split("\n", 1)[1] if "\n" in str(e) else str(e)
            friendly_error = friendly_error.split("[", 1)[0].strip()
            return friendly_error, {"visibility": "visible"}
        return "", {"visibility": "hidden"}
