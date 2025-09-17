"""Copyright © 2025, Empa.

Protocol editing sub-layout for the database tab.
"""

import base64
import json
import logging
import uuid
from decimal import Decimal
from pathlib import Path

import dash_mantine_components as dmc
from aurora_unicycler import (
    ConstantCurrent,
    ConstantVoltage,
    ImpedanceSpectroscopy,
    Loop,
    OpenCircuitVoltage,
    Protocol,
    Step,
    Tag,
)
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash_ag_grid import AgGrid
from pydantic import ValidationError

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.visualiser.notifications import error_notification, success_notification

logger = logging.getLogger(__name__)
CONFIG = get_config()

TECHNIQUE_NAMES = {
    "constant_current": "Constant current",
    "constant_voltage": "Constant voltage",
    "open_circuit_voltage": "Open circuit voltage",
    "impedance_spectroscopy": "Impedance spectroscopy",
    "loop": "Loop",
    "tag": "Tag",
}
ALL_TECHNIQUES = {
    "Constant current": ConstantCurrent,
    "Constant voltage": ConstantVoltage,
    "Open circuit voltage": OpenCircuitVoltage,
    "Impedance spectroscopy": ImpedanceSpectroscopy,
    "Loop": Loop,
    "Tag": Tag,
}
ALL_TECHNIQUES_REV = {v: k for k, v in ALL_TECHNIQUES.items()}
ALL_TECHNIQUE_INPUTS = {k for v in ALL_TECHNIQUES.values() for k in v.model_fields}
ALL_TECHNIQUE_INPUTS.remove("step")
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
    name = technique.get("step")
    if not name:
        description = "Select technique"
    elif name == "constant_current":
        conditions = []
        if (voltage := technique.get("until_voltage_V")) is not None:
            conditions.append(f"until {voltage} V")
        if (time := technique.get("until_time_s")) is not None and time > 0:
            conditions.append(f"until {seconds_to_time(time)}")
        description = f"{technique.get('rate_C')} C " + " or ".join(conditions)
    elif name == "constant_voltage":
        conditions = []
        if (c_rate := technique["until_rate_C"]) is not None:
            conditions.append(f"until {c_rate} C")
        elif (current := technique.get("until_current_mA")) is not None:
            conditions.append(f"until {current} mA")
        if (time := technique.get("until_time_s")) is not None and time > 0:
            conditions.append(f"until {seconds_to_time(time)}")
        description = f"{technique.get('voltage_V')} V " + " or ".join(conditions)
    elif name == "open_circuit_voltage":
        if (time := technique.get("until_time_s")) is not None and time > 0:
            description = f"until {seconds_to_time(time)}"
    elif name == "loop":
        loop_to = technique.get("loop_to")
        loop_to_str = f"'{loop_to}'" if isinstance(loop_to, str) else f"technqiue {loop_to} (1-indexed)"
        description = f"to {loop_to_str} for {technique.get('cycle_count')} cycles"
    elif name == "tag":
        description = f"{technique.get('tag')}"
    elif name == "impedance_spectroscopy":
        if technique.get("amplitude_V"):
            description = f"±{technique.get('amplitude_V')} V "
        else:
            description = f"±{technique.get('amplitude_mA')} mA "
        description += (
            f"from {technique.get('start_frequency_Hz'):.6g} Hz "
            f"to {technique.get('end_frequency_Hz'):.6g} Hz, "
            f"{technique.get('points_per_decade')} pts/dec, "
            f"{technique.get('measures_per_point')} meas/pt"
        )
        if technique.get("drift_correction"):
            description += ", drift correct"
    else:
        description = "Select technique"
    return description


def protocol_dict_to_row_data(protocol: dict) -> list[dict]:
    """Convert protocol dict to row data for the ag grid."""
    return [
        {
            "index": i,
            "id": technique.get("id"),
            "technique": TECHNIQUE_NAMES.get(technique.get("step", "...")),
            "description": describe_row(technique),
        }
        for i, technique in enumerate(protocol.get("method", []))
    ]


protocol_edit_grid = AgGrid(
    id="protocol-edit-grid",
    columnDefs=column_defs,
    rowData=[],
    virtualRowData=[],
    selectedRows=[],
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
    style={"height": "calc(100vh - 255px)", "width": "100%", "minHeight": "300px"},
    className="ag-theme-quartz",
)

protocol_edit_buttons = dmc.Group(
    justify="flex-start",
    align="center",
    gap="xs",
    pt="xs",
    children=[
        dmc.ActionIcon(
            html.I(className="bi bi-plus-circle"),
            id="add-row-button",
            color="green",
            size="lg",
        ),
        dmc.ActionIcon(
            html.I(className="bi bi-dash-circle"),
            id="remove-row-button",
            color="red",
            size="lg",
        ),
        dmc.ActionIcon(
            html.I(className="bi bi-copy"),
            id="copy-rows-button",
            size="lg",
        ),
        dmc.ActionIcon(
            html.I(className="bi bi-clipboard-plus"),
            id="paste-rows-button",
            size="lg",
        ),
    ],
)

protocol_header = dmc.Flex(
    [
        dmc.TextInput(
            label="Protocol name",
            id="protocol-name",
            style={"width": "100%"},
        ),
        dmc.Popover(
            [
                dmc.PopoverTarget(
                    dmc.ActionIcon(
                        html.I(className="bi bi-exclamation-triangle-fill"),
                        color="red",
                        id="protocol-warning",
                        style={"visibility": "hidden"},
                        size="input-sm",
                    ),
                ),
                dmc.PopoverDropdown(
                    dmc.Text(
                        "This explains why your thing is wrong!",
                        id="protocol-warning-message",
                    ),
                ),
            ],
        ),
    ],
    align="flex-end",
)

# Load/save protocol buttons
load_save_protocol_buttons = dmc.Group(
    pt="md",
    children=[
        dcc.Upload(
            dmc.Button(
                "Load protocol",
                leftSection=html.I(className="bi bi-folder2-open"),
                id="load-protocol-button-element",
            ),
            id="load-protocol-button",
            accept=".json",
            max_size=2 * 1024 * 1024,
            multiple=False,
            style_disabled={"opacity": "1"},
        ),
        dmc.Button(
            "Save protocol",
            leftSection=html.I(className="bi bi-save"),
            id="protocol-save-button",
        ),
    ],
)
# Input fields must have the same id as the inputs in ALL_TECHNIQUE_INPUTS
# and be inside a div with the id "{input_name}-group", used to hide inputs
step_edit_menu = dmc.Stack(
    id="step-edit-menu",
    children=[
        load_save_protocol_buttons,
        dmc.Select(
            id="technique-select",
            label="Select technique",
            data=[{"label": k, "value": k} for k in ALL_TECHNIQUES],
            placeholder="Select technique",
            checkIconPosition="right",
            comboboxProps={"offset": 0},
        ),
        dmc.Fieldset(
            legend="Technique settings",
            id="technique-settings",
            children=[
                html.Div(
                    dmc.NumberInput(
                        id="rate_C",
                        label="Rate (C)",
                        suffix=" C",
                        hideControls=True,
                    ),
                    id="rate_C-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="current_mA",
                        label="Current (mA)",
                        suffix=" mA",
                        hideControls=True,
                    ),
                    id="current_mA-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="voltage_V",
                        label="Voltage (V)",
                        suffix=" V",
                        hideControls=True,
                    ),
                    id="voltage_V-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="amplitude_V",
                        label="Amplitude (V)",
                        suffix=" V",
                        hideControls=True,
                        allowNegative=False,
                    ),
                    id="amplitude_V-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="amplitude_mA",
                        label="Amplitude (mA)",
                        suffix=" mA",
                        hideControls=True,
                        allowNegative=False,
                    ),
                    id="amplitude_mA-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="start_frequency_Hz",
                        label="Start frequency (Hz)",
                        suffix=" Hz",
                        thousandSeparator=",",
                        hideControls=True,
                        allowNegative=False,
                    ),
                    id="start_frequency_Hz-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="end_frequency_Hz",
                        label="End frequency (Hz)",
                        suffix=" Hz",
                        thousandSeparator=",",
                        hideControls=True,
                        allowNegative=False,
                    ),
                    id="end_frequency_Hz-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="points_per_decade",
                        label="Points per decade",
                        hideControls=True,
                        allowDecimal=False,
                        allowNegative=False,
                    ),
                    id="points_per_decade-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="measures_per_point",
                        label="Measures per frequency",
                        hideControls=True,
                        allowDecimal=False,
                        allowNegative=False,
                    ),
                    id="measures_per_point-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.Checkbox(
                        id="drift_correction",
                        label="Drift correction",
                    ),
                    id="drift_correction-group",
                    style={"display": "none"},
                ),
                dmc.NumberInput(
                    id="until_time_s",
                    label="Until time (s)",
                    suffix=" s",
                    style={"display": "none"},
                    hideControls=True,
                ),
                # 3 number inputs side by side
                html.Div(
                    dmc.Flex(
                        gap="xs",
                        justify="flex-start",
                        wrap="nowrap",
                        align="center",
                        children=[
                            dmc.NumberInput(
                                id="input_time_h",
                                label="Until time",
                                suffix=" h",
                                placeholder="0 h",
                                hideControls=True,
                                debounce=True,
                            ),
                            dmc.NumberInput(
                                id="input_time_m",
                                label=" ",
                                suffix=" min",
                                placeholder="0 min",
                                hideControls=True,
                                debounce=True,
                            ),
                            dmc.NumberInput(
                                id="input_time_s",
                                label=" ",
                                suffix=" s",
                                placeholder="0 s",
                                hideControls=True,
                                debounce=True,
                            ),
                        ],
                    ),
                    id="until_time_s-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="until_voltage_V",
                        label="Until voltage (V)",
                        suffix=" V",
                        hideControls=True,
                    ),
                    id="until_voltage_V-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="until_rate_C",
                        label="Until rate (C)",
                        suffix=" C",
                        hideControls=True,
                    ),
                    id="until_rate_C-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="until_current_mA",
                        label="Until current (mA)",
                        suffix=" mA",
                        hideControls=True,
                    ),
                    id="until_current_mA-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.TextInput(
                        id="loop_to",
                        label="Start step",
                        placeholder="Tag string or start step (1-indexed)",
                    ),
                    id="loop_to-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.NumberInput(
                        id="cycle_count",
                        label="Cycle count",
                        placeholder="Cycle count",
                    ),
                    id="cycle_count-group",
                    style={"display": "none"},
                ),
                html.Div(
                    dmc.TextInput(
                        id="tag",
                        label="Tag",
                        placeholder="Tag",
                    ),
                    id="tag-group",
                    style={"display": "none"},
                ),
                dmc.Group(
                    pt="sm",
                    children=[
                        dmc.Button(
                            "Update",
                            leftSection=html.I(className="bi bi-check2", style={"fontSize": "1.5em"}),
                            id="submit",
                        ),
                        dmc.Popover(
                            [
                                dmc.PopoverTarget(
                                    dmc.ActionIcon(
                                        html.I(className="bi bi-exclamation-triangle-fill"),
                                        color="red",
                                        id="step-warning",
                                        style={"display": "none"},
                                        size="input-sm",
                                    ),
                                ),
                                dmc.PopoverDropdown(
                                    dmc.Text(
                                        "This explains why your thing is wrong!",
                                        id="step-warning-message",
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Menu for editing record and safety parameters
global_edit_menu = dmc.Stack(
    id="global-edit-menu",
    children=[
        dmc.Fieldset(
            legend="Measurement parameters",
            children=[
                dmc.NumberInput(
                    id="record_interval_s",
                    placeholder="Time interval (s)",
                    style={"width": "100%"},
                    suffix=" s",
                    hideControls=True,
                    debounce=True,
                ),
                dmc.NumberInput(
                    id="record_interval_v",
                    placeholder="Voltage interval (V)",
                    style={"width": "100%"},
                    suffix=" V",
                    hideControls=True,
                    debounce=True,
                ),
                dmc.NumberInput(
                    id="record_interval_mA",
                    placeholder="Current interval (mA)",
                    style={"width": "100%"},
                    suffix=" mA",
                    hideControls=True,
                    debounce=True,
                ),
            ],
        ),
        dmc.Fieldset(
            legend="Safety limits",
            children=[
                dmc.Flex(
                    gap="xs",
                    justify="flex-start",
                    wrap="nowrap",
                    align="center",
                    children=[
                        dmc.NumberInput(
                            id="min_voltage_V",
                            placeholder="Minimum voltage (V)",
                            suffix=" V",
                            hideControls=True,
                            debounce=True,
                        ),
                        dmc.Text("to"),
                        dmc.NumberInput(
                            id="max_voltage_V",
                            placeholder="Maximum voltage (V)",
                            suffix=" V",
                            hideControls=True,
                            debounce=True,
                        ),
                    ],
                ),
                dmc.Flex(
                    gap="xs",
                    justify="flex-start",
                    wrap="nowrap",
                    align="center",
                    children=[
                        dmc.NumberInput(
                            id="min_current_mA",
                            placeholder="Minimum current (mA)",
                            suffix=" mA",
                            hideControls=True,
                            debounce=True,
                        ),
                        dmc.Text("to"),
                        dmc.NumberInput(
                            id="max_current_mA",
                            placeholder="Maximum current (mA)",
                            suffix=" mA",
                            hideControls=True,
                            debounce=True,
                        ),
                    ],
                ),
                dmc.Flex(
                    gap="xs",
                    justify="flex-start",
                    wrap="nowrap",
                    align="center",
                    children=[
                        dmc.NumberInput(
                            id="delay_s",
                            placeholder="Delay before safety stop (s)",
                            suffix=" s",
                            hideControls=True,
                            debounce=True,
                        ),
                    ],
                ),
            ],
        ),
    ],
)


protocol_edit_layout = html.Div(
    id="protocol-container",
    children=[
        dcc.Store(id="protocol-store", data={"method": [], "record": {}, "safety": {}}),
        dcc.Store(id="protocol-store-selected", data=[]),  # For selected rows
        dcc.Store(id="protocol-edit-clipboard", data=[]),  # For copy/paste functionality
        html.Div(
            style={"display": "flex", "height": "100%"},
            children=[
                html.Div(
                    style={"width": "800px", "padding": "10px"},
                    children=dmc.Stack(
                        [
                            step_edit_menu,
                            global_edit_menu,
                        ],
                        justify="space-between",
                        style={"height": "100%", "overflowY": "scroll"},
                    ),
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
        dcc.ConfirmDialog(
            id="save-protocol-confirm",
            message="Save as new protocol?",
        ),
        dcc.ConfirmDialog(
            id="overwrite-protocol-confirm",
            message="Warning: This will overwrite an existing protocol. Are you sure?",
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
        Output("record_interval_s", "value"),
        Output("record_interval_v", "value"),
        Output("record_interval_mA", "value"),
        Output("min_voltage_V", "value"),
        Output("max_voltage_V", "value"),
        Output("min_current_mA", "value"),
        Output("max_current_mA", "value"),
        Output("delay_s", "value"),
        Input("protocol-store", "data"),
        State("protocol-store-selected", "data"),
        prevent_initial_call=True,
    )
    def update_grid(protocol_dict: dict, selected_indices: list) -> tuple:
        """Update the grid with the new data."""
        if protocol_dict is None or not protocol_dict:  # return empty grid data
            return [], []
        row_data = protocol_dict_to_row_data(protocol_dict)
        new_selected_rows = [row_data[i] for i in selected_indices] if selected_indices else []
        # Update the record and safety parameters
        record_interval_s = protocol_dict.get("record", {}).get("time_s")
        record_interval_v = protocol_dict.get("record", {}).get("voltage_V")
        record_interval_mA = protocol_dict.get("record", {}).get("current_mA")
        safety_max_V = protocol_dict.get("safety", {}).get("max_voltage_V")
        safety_min_V = protocol_dict.get("safety", {}).get("min_voltage_V")
        safety_max_mA = protocol_dict.get("safety", {}).get("max_current_mA")
        safety_min_mA = protocol_dict.get("safety", {}).get("min_current_mA")
        safety_delay_s = protocol_dict.get("safety", {}).get("delay_s")
        return (
            new_selected_rows,
            row_data,
            record_interval_s,
            record_interval_v,
            record_interval_mA,
            safety_min_V,
            safety_max_V,
            safety_min_mA,
            safety_max_mA,
            safety_delay_s,
        )

    # Load a protocol button
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Output("protocol-store-selected", "data", allow_duplicate=True),
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
                protocol = Protocol.from_dict(samples).model_dump()
                # add an id to each technique
                for technique in protocol["method"]:
                    technique["id"] = uuid.uuid4()
                return protocol, [], filename[:-5], no_update
            except ValidationError:
                return (
                    no_update,
                    no_update,
                    no_update,
                    error_notification("Oh no", "This is not a valid protocol file!"),
                )
        raise PreventUpdate

    # Remove rows button
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Output("protocol-store-selected", "data", allow_duplicate=True),
        Input("remove-row-button", "n_clicks"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        State("protocol-edit-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def remove_rows(
        n_clicks: int,
        protocol_dict: dict,
        grid_data: list[dict],
        selected_rows: list[dict],
    ) -> tuple[dict, list[int]]:
        """Remove the selected rows from the data store."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        if selected_rows is None or not selected_rows:
            raise PreventUpdate
        # Get the new indices, remove if the index is in the selected indices
        selected_indices = [row["index"] for row in selected_rows]
        indices = [row["index"] for row in grid_data if row["index"] not in selected_indices]
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # update the store with the new data
        return protocol_dict, []

    # Add rows button
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Output("protocol-store-selected", "data", allow_duplicate=True),
        Input("add-row-button", "n_clicks"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        State("protocol-edit-grid", "selectedRows"),
        prevent_initial_call=True,
    )
    def add_row(
        n_clicks: int, protocol_dict: dict, grid_data: list[dict], selected_rows: list[dict]
    ) -> tuple[dict, list[int]]:
        """Add a new row to the data store."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        # reorder the techniques if the user has dragged rows around
        indices = [row["index"] for row in grid_data]
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # get the largest selected index
        index = None
        if selected_rows is not None and selected_rows:
            selected_indices = [s["index"] for s in selected_rows]
            reordered_indices = [i for i, row in enumerate(indices) if row in selected_indices]
            index = max(reordered_indices) + 1 if reordered_indices else None
        # add a new row to the data store
        new_row = Step().model_dump()
        new_row["id"] = uuid.uuid4()
        if index is not None:
            protocol_dict["method"].insert(index, new_row)
        else:
            protocol_dict["method"].append(new_row)
            index = len(protocol_dict["method"]) - 1
        # update the store with the new data
        return protocol_dict, [index]

    # On 'copy', store the corresponding data rows in the clipboard
    @app.callback(
        Output("protocol-edit-clipboard", "data"),
        Input("copy-rows-button", "n_clicks"),
        State("protocol-edit-grid", "selectedRows"),
        State("protocol-store", "data"),
        State("protocol-edit-grid", "virtualRowData"),
        prevent_initial_call=True,
    )
    def copy_rows(n_clicks: int, selected_rows: list[dict], protocol_dict: dict, grid_data: list) -> list[dict]:
        """Copy the selected rows to the clipboard."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        if selected_rows is None or not selected_rows:
            raise PreventUpdate
        # Get the indices of the selected rows
        selected_indices = [row["index"] for row in selected_rows]
        # All rows after dragging
        virtual_indices = [row["index"] for row in grid_data]
        # Reorder the selected indices so they are in the same order as the virtual indices
        reordered_indices = [row for row in virtual_indices if row in selected_indices]
        # Get the actual list of techniques from the protocol store, store this
        return [protocol_dict["method"][i] for i in reordered_indices]

    # On 'paste', add the rows from the clipboard to the protocol
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Output("protocol-store-selected", "data", allow_duplicate=True),
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
        protocol_dict: dict,
        grid_data: list[dict],
        selected_rows: list[dict],
    ) -> tuple[dict, list[int]]:
        """Paste the rows from the clipboard to the protocol."""
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate
        if not clipboard_data:
            raise PreventUpdate
        # reorder the techniques if the user has dragged rows around
        indices = [row["index"] for row in grid_data]
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
        return protocol_dict, selected_indices

    # If user selects a row, show it in the step edit menu
    @app.callback(
        Output("technique-select", "value"),
        [Output(x, "value", allow_duplicate=True) for x in ALL_TECHNIQUE_INPUTS],
        Input("protocol-edit-grid", "selectedRows"),
        State("protocol-store", "data"),
        prevent_initial_call=True,
    )
    def update_step_edit_menu(selected_rows: list[dict], protocol_dict: dict) -> tuple[str | None, ...]:
        """Update the step edit menu with the selected row data."""
        if selected_rows is None or not selected_rows:
            return "", *([""] * len(ALL_TECHNIQUE_INPUTS))
        selected_row = selected_rows[0]
        index = selected_row["index"]
        technique = protocol_dict["method"][index]
        input_values = [technique.get(x, "") for x in ALL_TECHNIQUE_INPUTS]
        return selected_row["technique"], *input_values

    # If user selects a technique, show the inputs for that technique
    @app.callback(
        [Output(x + "-group", "style") for x in ALL_TECHNIQUE_INPUTS],
        Output("technique-settings", "style"),
        Input("technique-select", "value"),
        prevent_initial_call=True,
    )
    def update_technique_inputs(technique: str) -> list[dict]:
        """Update the input fields based on the selected technique."""
        show = {"display": "block"}
        hide = {"display": "none"}
        if technique not in ALL_TECHNIQUES:
            return [hide for _ in ALL_TECHNIQUE_INPUTS] + [hide]
        return [show if x in ALL_TECHNIQUES[technique].model_fields else hide for x in ALL_TECHNIQUE_INPUTS] + [show]

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
        hours = hours or 0
        mins = mins or 0
        secs = secs or 0
        hours = max(int(hours), 0)
        mins = max(int(mins), 0)
        secs = max(int(secs), 0)
        return hours * 3600 + mins * 60 + secs

    # If the real time input changes, update the hour/min/sec inputs
    @app.callback(
        Output("input_time_h", "value"),
        Output("input_time_m", "value"),
        Output("input_time_s", "value"),
        Input("until_time_s", "value"),
        prevent_initial_call=True,
    )
    def update_time_inputs(until_time_s: float) -> tuple[int | str, int | str, int | str]:
        """Update the time inputs based on the until_time_s value."""
        if until_time_s is None or until_time_s == "":
            return 0, 0, 0
        hours = int(until_time_s) // 3600
        minutes = (int(until_time_s) % 3600) // 60
        seconds = int(until_time_s) % 60
        return hours, minutes, seconds

    @app.callback(
        Output("drift_correction", "value"),
        Input("drift_correction", "checked"),
    )
    def update_drift_correction(checked: bool) -> bool:
        """Dmc uses 'checked' for checkbox and 'value' for everything else."""
        return checked

    # if you change a value in the step edit menu, check if the technique is valid
    @app.callback(
        Output("step-warning", "style"),
        Output("step-warning-message", "children"),
        Output("submit", "disabled"),
        Input("technique-select", "value"),
        [Input(x, "value") for x in ALL_TECHNIQUE_INPUTS],
        prevent_initial_call=True,
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
                    for name, value in zip(ALL_TECHNIQUE_INPUTS, input_values, strict=True)
                    if name in technique_cls.model_fields
                },
            )
        except (ValidationError, TypeError) as e:
            logger.exception("Pydantic validation error of individual technique")
            friendly_error = str(e).split("\n", 1)[1] if "\n" in str(e) else str(e)
            friendly_error = friendly_error.split("[", 1)[0].strip()
            return {"display": "block"}, friendly_error, True
        return {"display": "none"}, "", False

    # If user changes value in the step edit menu and presses submit, update the protocol store
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Output("protocol-store-selected", "data", allow_duplicate=True),
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
        protocol_dict: dict,
        technique: str,
        *input_values: list[str | float | None],
    ) -> tuple[dict, list[int]]:
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
                for name, value in zip(ALL_TECHNIQUE_INPUTS, input_values, strict=True)
                if name in technique_cls.model_fields
            },
        ).model_dump()
        new_technique["id"] = technique_id if technique_id else uuid.uuid4()

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

        return protocol_dict, new_selected_indicies

    # If the virtual data changes (dragging, updating data) or global settings change, check if protocol is valid
    @app.callback(
        Output("protocol-warning-message", "children"),
        Output("protocol-warning", "style"),
        Input("protocol-edit-grid", "virtualRowData"),
        Input("protocol-store", "data"),
        prevent_initial_call=True,
    )
    def validate_protocol(grid_data: list[dict], protocol_dict: dict) -> tuple[str, dict]:
        """Validate the protocol and update the grid data."""
        # Reorder the techniques in case the user has dragged rows around
        indices = [row["index"] for row in grid_data] if grid_data else []
        protocol_dict["method"][:] = [protocol_dict["method"][i] for i in indices]
        # Validate the protocol
        try:
            Protocol.from_dict(protocol_dict)
        except ValidationError as e:
            logger.exception("Pydantic validation error for whole protocol")
            friendly_error = str(e).split("\n", 1)[1] if "\n" in str(e) else str(e)
            friendly_error = friendly_error.split("[", 1)[0].strip()
            return friendly_error, {"visibility": "visible"}
        return "", {"visibility": "hidden"}

    # If name or validity of protocol changes, change save button
    @app.callback(
        Output("protocol-save-button", "disabled"),
        Input("protocol-warning", "style"),
        Input("protocol-name", "value"),
        prevent_initial_call=True,
    )
    def update_save_button(warning: dict, name: str) -> bool:
        """Update the save button based on the protocol validity and name."""
        return not name or name.strip() == "" or warning.get("visibility") == "visible"

    # If any safety or record parameters change, update the protocol store
    @app.callback(
        Output("protocol-store", "data", allow_duplicate=True),
        Input("record_interval_s", "value"),
        Input("record_interval_v", "value"),
        Input("record_interval_mA", "value"),
        Input("min_voltage_V", "value"),
        Input("max_voltage_V", "value"),
        Input("min_current_mA", "value"),
        Input("max_current_mA", "value"),
        Input("delay_s", "value"),
        State("protocol-store", "data"),
        prevent_initial_call=True,
    )
    def update_global_parameters(
        record_interval_s: float,
        record_interval_v: float,
        record_interval_mA: float,
        min_voltage_V: float,
        max_voltage_V: float,
        min_current_mA: float,
        max_current_mA: float,
        delay_s: float,
        protocol_dict: dict,
    ) -> dict:
        """Update the global parameters in the protocol store."""
        protocol_dict["record"]["time_s"] = record_interval_s
        protocol_dict["record"]["voltage_V"] = record_interval_v
        protocol_dict["record"]["current_mA"] = record_interval_mA
        protocol_dict["safety"]["min_voltage_V"] = min_voltage_V
        protocol_dict["safety"]["max_voltage_V"] = max_voltage_V
        protocol_dict["safety"]["min_current_mA"] = min_current_mA
        protocol_dict["safety"]["max_current_mA"] = max_current_mA
        protocol_dict["safety"]["delay_s"] = delay_s
        return protocol_dict

    # Pressing save opens a confirm dialog
    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Output("save-protocol-confirm", "displayed"),
        Output("overwrite-protocol-confirm", "displayed"),
        Input("protocol-save-button", "n_clicks"),
        State("protocol-name", "value"),
        prevent_initial_call=True,
    )
    def save_protocol_button(n_clicks: int, name: str) -> tuple:
        """Open the confirm dialog for saving the protocol."""
        # Check if the protocol name exists in protocols folder
        folder = CONFIG.get("Protocols folder path")
        if not folder:
            return (
                error_notification("No protocols folder", "Please check config"),
                no_update,
                no_update,
            )
        if n_clicks:
            file_path = Path(folder) / f"{name}.json"
            if file_path.exists():
                return no_update, no_update, True
            return no_update, True, no_update
        raise PreventUpdate

    # If the user confirms saving, save the protocol to the protocols folder
    @app.callback(
        Output("notifications-container", "children", allow_duplicate=True),
        Input("save-protocol-confirm", "submit_n_clicks"),
        Input("overwrite-protocol-confirm", "submit_n_clicks"),
        State("protocol-store", "data"),
        State("protocol-name", "value"),
        prevent_initial_call=True,
    )
    def save_protocol(
        save_clicks: int,
        overwrite_clicks: int,
        protocol_dict: dict,
        name: str,
    ) -> list:
        """Save the protocol to the protocols folder."""
        folder = CONFIG.get("Protocols folder path")
        if not folder:
            return error_notification("Error", "No protocols folder path set in config!")
        file_path = Path(folder) / f"{name}.json"
        if file_path.exists() and overwrite_clicks is None:
            return error_notification(
                "Cannot save",
                "Protocol with this name already exists! Please choose another name.",
            )
        try:
            protocol_copy = protocol_dict.copy()
            for technique in protocol_copy.get("method", []):
                if "id" in technique:
                    del technique["id"]
            protocol = Protocol.from_dict(protocol_copy)
            folder.mkdir(parents=True, exist_ok=True)
            with file_path.open("w") as f:
                f.write(protocol.model_dump_json(exclude_none=True, indent=4))
            return success_notification("Success", f"'{name}' saved to protocols folder")
        except Exception as e:
            logger.exception("Error saving protocol")
            return error_notification("Error", f"Could not save protocol: {e}")
