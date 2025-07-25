"""Copyright © 2025, Empa.

Samples tab layout and callbacks for the visualiser app.
"""

import json
import logging
import os
from pathlib import Path

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from aurora_cycler_manager.analysis import calc_dqdv, combine_jobs
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.utils import run_from_sample

CONFIG = get_config()
logger = logging.getLogger(__name__)
graph_template = "seaborn"
graph_margin = {"l": 75, "r": 20, "t": 50, "b": 75}

# Side menu for the samples tab
samples_menu = html.Div(
    style={"overflow": "auto", "height": "100%"},
    children=dmc.Stack(
        p="xs",
        children=[
            dmc.MultiSelect(
                id="samples-dropdown",
                label="Select samples",
                searchable=True,
                clearable=True,
                checkIconPosition="right",
                comboboxProps={"offset": 0},
                value=[],
                data=[],
            ),
            dmc.Tooltip(
                dmc.Checkbox(
                    id="compressed-files",
                    label="Use compressed files",
                    checked=True,
                ),
                label="Use compressed time-series data where available - better performance, less accurate.",
                multiline=True,
                openDelay=1000,
            ),
            dmc.Fieldset(
                legend="Time graph",
                children=[
                    dmc.Select(
                        id="samples-time-x",
                        label="X-axis:",
                        data=[
                            "Datetime",
                            "Unix time",
                            "From start",
                            "From formation",
                            "From cycling",
                        ],
                        value="From start",
                        checkIconPosition="right",
                        comboboxProps={"offset": 0},
                    ),
                    dmc.Select(
                        id="samples-time-units",
                        label="X-axis units:",
                        data=["Seconds", "Minutes", "Hours", "Days"],
                        value="Hours",
                        checkIconPosition="right",
                        comboboxProps={"offset": 0},
                    ),
                    dmc.Select(
                        id="samples-time-y",
                        label="Y-axis:",
                        data=["V (V)"],
                        value="V (V)",
                        searchable=True,
                        checkIconPosition="right",
                        comboboxProps={"offset": 0},
                    ),
                ],
            ),
            dmc.Fieldset(
                legend="Cycles graph",
                children=[
                    dmc.Text("X-axis: Cycle"),
                    dmc.Select(
                        id="samples-cycles-y",
                        label="Y-axis:",
                        data=[
                            "Specific discharge capacity (mAh/g)",
                        ],
                        value="Specific discharge capacity (mAh/g)",
                        searchable=True,
                        checkIconPosition="right",
                        comboboxProps={"offset": 0},
                    ),
                ],
            ),
            dmc.Fieldset(
                legend="One cycle graph",
                children=[
                    dmc.Select(
                        id="samples-cycle-x",
                        label="X-axis:",
                        data=["Q (mAh)", "V (V)", "dQ/dV (mAh/V)", "Q (mAh/g)", "dQ/dV (mAh/gV)"],
                        value="Q (mAh)",
                        searchable=True,
                        checkIconPosition="right",
                        comboboxProps={"offset": 0},
                    ),
                    dmc.Select(
                        id="samples-cycle-y",
                        label="Y-axis:",
                        data=["Q (mAh)", "V (V)", "dQ/dV (mAh/V)", "Q (mAh/g)", "dQ/dV (mAh/gV)"],
                        value="V (V)",
                        searchable=True,
                        checkIconPosition="right",
                        comboboxProps={"offset": 0},
                    ),
                    dmc.NumberInput(
                        id="cycle-number",
                        label="Cycle number:",
                        placeholder="Cycle number",
                        min=1,
                        value=1,
                    ),
                ],
            ),
        ],
    ),
)

time_graph = dcc.Graph(
    id="time-graph",
    style={"height": "100%", "width": "100%"},
    config={"scrollZoom": True, "displaylogo": False},
    figure={
        "data": [],
        "layout": go.Layout(
            template=graph_template,
            margin=graph_margin,
            title="",
            xaxis={"title": "Time"},
            yaxis={"title": ""},
            showlegend=False,
        ),
    },
)

cycles_graph = dcc.Graph(
    id="cycles-graph",
    style={"height": "100%", "width": "100%"},
    config={"scrollZoom": True, "displaylogo": False},
    figure={
        "data": [],
        "layout": go.Layout(
            template=graph_template,
            margin=graph_margin,
            title="",
            xaxis={"title": "Cycle"},
            yaxis={"title": ""},
            showlegend=False,
        ),
    },
)

one_cycle_graph = dcc.Graph(
    id="cycle-graph",
    config={"scrollZoom": True, "displaylogo": False},
    style={"height": "100%", "width": "100%"},
    figure={
        "data": [],
        "layout": go.Layout(
            template=graph_template,
            margin=graph_margin,
            title="",
            xaxis={"title": ""},
            yaxis={"title": ""},
            showlegend=False,
        ),
    },
)

samples_layout = html.Div(
    style={"height": "100%"},
    children=[
        dcc.Store(
            id="samples-data-store",
            data={"data_sample_time": {}, "data_sample_cycle": {}},
        ),
        PanelGroup(
            id="samples-panel-group",
            direction="horizontal",
            style={"height": "100%"},
            children=[
                Panel(
                    id="samples-menu",
                    children=samples_menu,
                    defaultSizePercentage=20,
                    collapsible=True,
                ),
                PanelResizeHandle(
                    html.Div(className="resize-handle-horizontal"),
                ),
                Panel(
                    id="samples-graphs",
                    minSizePercentage=50,
                    children=[
                        PanelGroup(
                            id="samples-graph-group",
                            direction="vertical",
                            children=[
                                Panel(
                                    time_graph,
                                    id="samples-top-graph",
                                ),
                                PanelResizeHandle(
                                    html.Div(className="resize-handle-vertical"),
                                ),
                                Panel(
                                    id="samples-bottom-graphs",
                                    children=[
                                        PanelGroup(
                                            id="samples-bottom-graph-group",
                                            direction="horizontal",
                                            children=[
                                                Panel(
                                                    cycles_graph,
                                                    id="samples-bottom-left-graph",
                                                ),
                                                PanelResizeHandle(
                                                    html.Div(className="resize-handle-horizontal"),
                                                ),
                                                Panel(
                                                    one_cycle_graph,
                                                    id="samples-bottom-right-graph",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# --------------------------------- CALLBACKS ----------------------------------#
def register_samples_callbacks(app: Dash) -> None:
    """Register all callbacks for the samples tab."""

    # Sample list has updated, update dropdowns
    @app.callback(
        Output("samples-dropdown", "data"),
        Output("batch-samples-dropdown", "data"),
        Output("batch-edit-samples", "data"),
        Input("samples-store", "data"),
        prevent_initial_call=True,
    )
    def update_samples_dropdown(samples: list):
        """Update available samples in the dropdown."""
        return samples, samples, samples

    # Update the samples data store
    @app.callback(
        Output("samples-data-store", "data"),
        Output("samples-time-y", "data"),
        Output("samples-cycles-y", "data"),
        Input("samples-dropdown", "value"),
        Input("compressed-files", "checked"),
        State("samples-data-store", "data"),
        running=[(Output("loading-message-store", "data"), "Loading data...", "")],
        prevent_initial_call=True,
    )
    def update_sample_data(samples: list, compressed: bool, data: dict) -> tuple[dict, list, list]:
        """Load data for selected samples and put in data store."""
        # Get rid of samples that are no longer selected
        for sample in list(data["data_sample_time"].keys()):
            if sample not in samples:
                data["data_sample_time"].pop(sample)
                if sample in data["data_sample_cycle"]:
                    data["data_sample_cycle"].pop(sample)

        for sample in samples or []:
            # Check if already in data store
            if sample in data["data_sample_time"]:
                # Check if it's already the correct format
                if not compressed and not data["data_sample_time"][sample].get("Shrunk"):
                    continue
                if compressed and data["data_sample_time"][sample].get("Shrunk"):
                    continue

            # Otherwise import the data
            run_id = run_from_sample(sample)
            data_folder = CONFIG["Processed snapshots folder path"]
            file_location = str(data_folder / run_id / sample)

            # Get raw data
            try:
                files = os.listdir(file_location)
            except FileNotFoundError:
                continue
            if compressed and any(f.startswith("shrunk") and f.endswith(".h5") for f in files):
                filepath = next(f for f in files if f.startswith("shrunk") and f.endswith(".h5"))
                df = pd.read_hdf(f"{file_location}/{filepath}")
                data_dict = df.to_dict(orient="list")
                data_dict["Shrunk"] = True
                data["data_sample_time"][sample] = data_dict
            elif any(f.startswith("full") and f.endswith(".h5") for f in files):
                filepath = next(f for f in files if f.startswith("full") and f.endswith(".h5"))
                df = pd.read_hdf(f"{file_location}/{filepath}")
                data["data_sample_time"][sample] = df.to_dict(orient="list")
            else:
                cycling_files = [
                    os.path.join(file_location, f) for f in files if (f.startswith("snapshot") and f.endswith(".h5"))
                ]
                if not cycling_files:
                    logger.info("No cycling files found in %s", file_location)
                    continue
                df, metadata = combine_jobs([Path(f) for f in cycling_files])
                data["data_sample_time"][sample] = df.to_dict(orient="list")

            # Get the analysed file
            try:
                analysed_file = next(f for f in files if (f.startswith("cycles") and f.endswith(".json")))
            except StopIteration:
                continue
            with open(f"{file_location}/{analysed_file}", encoding="utf-8") as f:
                cycle_dict = json.load(f)["data"]
            if not cycle_dict or "Cycle" not in cycle_dict:
                continue
            data["data_sample_cycle"][sample] = cycle_dict

        # Update the y-axis options
        time_y_vars = {"V (V)"}
        for data_dict in data["data_sample_time"].values():
            time_y_vars.update(data_dict.keys())
        time_y_vars.discard("Shrunk")

        cycles_y_vars = {"Specific discharge capacity (mAh/g)"}
        for data_dict in data["data_sample_cycle"].values():
            cycles_y_vars.update([k for k, v in data_dict.items() if isinstance(v, list)])

        return data, list(time_y_vars), list(cycles_y_vars)

    # Update the time graph
    @app.callback(
        Output("time-graph", "figure"),
        State("time-graph", "figure"),
        Input("samples-data-store", "data"),
        Input("samples-time-x", "value"),
        Input("samples-time-units", "value"),
        Input("samples-time-y", "value"),
        running=[(Output("loading-message-store", "data"), "Plotting time-series...", "")],
        prevent_initial_call=True,
    )
    def update_time_graph(fig: dict, data: dict, xvar: str, xunits: str, yvar: str) -> dict:
        """When data or x/y variables change, update the time graph."""
        fig["data"] = []
        fig["layout"]["xaxis"]["title"] = f"Time ({xunits.lower()})" if xvar != "Datetime" else "Datetime (UTC)"
        fig["layout"]["yaxis"]["title"] = yvar
        if not data["data_sample_time"] or not xvar or not yvar or not xunits:
            if not data["data_sample_time"]:
                fig["layout"]["title"] = "No data..."
            elif not xvar or not yvar or not xunits:
                fig["layout"]["title"] = "Select x and y variables"
            return fig
        fig["layout"]["title"] = f"{yvar} vs time"
        go_fig = go.Figure(layout=fig["layout"])
        multiplier = (
            {"Seconds": 1, "Minutes": 60, "Hours": 3600, "Days": 86400}[xunits]
            if xvar != "Datetime"
            else 0.001  # To get UTC datetime from unix time stamp in milliseconds
        )
        for sample, data_dict in data["data_sample_time"].items():
            uts = np.array(data_dict["uts"])
            if xvar == "From start":
                offset = uts[0]
            elif xvar == "From formation":
                offset = uts[next(i for i, x in enumerate(data_dict["Cycle"]) if x >= 1)]
            elif xvar == "From cycling":
                # grab n formation
                formation_cycle_count = data["data_sample_cycle"].get(sample, {}).get("Formation cycles", 3)
                try:
                    offset = uts[next(i for i, x in enumerate(data_dict["Cycle"]) if x > formation_cycle_count)]
                except StopIteration:
                    offset = uts[-1]
            else:
                offset = 0

            trace = go.Scatter(
                x=(np.array(data_dict["uts"]) - offset) / multiplier,
                y=data_dict[yvar],
                mode="lines",
                name=sample,
                hovertemplate=f"{sample}<br>Time: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            go_fig.add_trace(trace)
        if xvar == "Datetime":
            go_fig.update_layout(xaxis={"type": "date", "tickformat": "%Y-%m-%d %H:%M:%S"})
        else:
            go_fig.update_layout(xaxis={"type": "linear"})
        return go_fig

    # Update the cycles graph
    @app.callback(
        Output("cycles-graph", "figure"),
        State("cycles-graph", "figure"),
        Input("samples-data-store", "data"),
        Input("samples-cycles-y", "value"),
        running=[(Output("loading-message-store", "data"), "Plotting cycles...", "")],
        prevent_initial_call=True,
    )
    def update_cycles_graph(fig: dict, data: dict, yvar: str) -> dict:
        """When data or y variable changes, update the cycles graph."""
        fig["data"] = []
        if yvar:
            fig["layout"]["title"] = f"{yvar} vs cycle"
            fig["layout"]["yaxis"]["title"] = yvar
        else:
            fig["layout"]["title"] = "Select y variable"
            return fig
        if not data["data_sample_cycle"]:
            fig["layout"]["title"] = "No data..."
            return fig
        for sample, cycle_dict in data["data_sample_cycle"].items():
            trace = go.Scattergl(
                x=cycle_dict["Cycle"],
                y=cycle_dict[yvar],
                mode="lines+markers",
                name=sample,
                hovertemplate=f"{sample}<br>Cycle: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            fig["data"].append(trace)
        return go.Figure(data=fig["data"], layout=fig["layout"])

    # When the user clicks on a point, update the cycle number
    @app.callback(
        Output("cycle-number", "value"),
        Input("cycles-graph", "clickData"),
        prevent_initial_call=True,
    )
    def update_cycle_number(click_data: dict) -> int:
        """When the user clicks on a point, update the cycle number input."""
        if not click_data:
            return 1
        point = click_data["points"][0]
        return point["x"]

    # Update the one cycle graph
    @app.callback(
        Output("cycle-graph", "figure"),
        State("cycle-graph", "figure"),
        Input("cycle-number", "value"),
        Input("samples-data-store", "data"),
        Input("samples-cycle-x", "value"),
        Input("samples-cycle-y", "value"),
        running=[(Output("loading-message-store", "data"), "Plotting one-cycle...", "")],
        prevent_initial_call=True,
    )
    def update_cycle_graph(fig: dict, cycle: int, data: dict, xvar: str, yvar: str) -> dict:
        """When data or x/y variables change, update the one cycle graph."""
        fig["data"] = []
        fig["layout"]["xaxis"]["title"] = xvar if xvar else "Select x variable"
        fig["layout"]["yaxis"]["title"] = yvar if yvar else "Select y variable"
        if not data["data_sample_cycle"]:
            fig["layout"]["title"] = "No data..."
            return fig
        if not xvar or not yvar:
            return fig
        for sample, data_dict in data["data_sample_time"].items():
            # find where the cycle = cycle
            mask = np.array(data_dict["Cycle"]) == cycle
            if not any(mask):
                # increment colour anyway by adding an empty trace
                fig["data"].append(go.Scattergl())
                continue
            mask_dict = {}
            mask_dict["V (V)"] = np.array(data_dict["V (V)"])[mask]
            mask_dict["Q (mAh)"] = np.array(data_dict["dQ (mAh)"])[mask].cumsum()
            mask_dict["dQ (mAh)"] = np.array(data_dict["dQ (mAh)"])[mask]
            if "dQ/dV (mAh/V)" in [xvar, yvar] or "dQ/dV (mAh/gV)" in [xvar, yvar]:
                if "dQ/dV (mAh/V)" in data_dict:
                    mask_dict["dQ/dV (mAh/V)"] = np.array(data_dict["dQ/dV (mAh/V)"], dtype=float)[mask]
                else:
                    mask_dict["dQ/dV (mAh/V)"] = calc_dqdv(
                        mask_dict["V (V)"],
                        mask_dict["Q (mAh)"],
                        mask_dict["dQ (mAh)"],
                    )
            m_mg = None
            if "Q (mAh/g)" in [xvar, yvar] or "dQ/dV (mAh/gV)" in [xvar, yvar]:
                m_mg = data["data_sample_cycle"][sample].get("Cathode active material mass (mg)")
                if "Q (mAh/g)" in [xvar, yvar]:
                    mask_dict["Q (mAh/g)"] = mask_dict["Q (mAh)"] / m_mg * 1000 if m_mg else None
                if "dQ/dV (mAh/gV)" in [xvar, yvar]:
                    mask_dict["dQ/dV (mAh/gV)"] = mask_dict["dQ/dV (mAh/V)"] / m_mg * 1000 if m_mg else None

            trace = go.Scattergl(
                x=mask_dict.get(xvar),
                y=mask_dict.get(yvar),
                mode="lines",
                name=sample,
                hovertemplate=f"{sample}<br>{xvar}: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            fig["data"].append(trace)
        fig["layout"]["title"] = f"{yvar} vs {xvar} for cycle {cycle}"
        return go.Figure(data=fig["data"], layout=fig["layout"])
