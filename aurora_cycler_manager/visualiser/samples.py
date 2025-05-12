"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Samples tab layout and callbacks for the visualiser app.
"""

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html
from dash_bootstrap_components import Checklist, Tooltip
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from aurora_cycler_manager.analysis import combine_jobs
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.utils import run_from_sample
from aurora_cycler_manager.visualiser.funcs import smoothed_derivative

CONFIG = get_config()

graph_template = "seaborn"
graph_margin = {"l": 50, "r": 10, "t": 50, "b": 75}

# Side menu for the samples tab
samples_menu = html.Div(
    style={"overflow": "scroll", "height": "100%"},
    children=[
        html.H5("Select samples to plot:"),
        dcc.Dropdown(
            id="samples-dropdown",
            options=[],  # updated by callback
            value=[],
            multi=True,
        ),
        Checklist(
            options=[
                {
                    "label": "Use compressed files",
                    "value": 1,
                },
            ],
            value=[1],
            id="compressed-files",
        ),
        Tooltip(
            "Use compressed time-series data where available - better performance, less accurate.",
            target="compressed-files",
            delay={"show": 1000},
        ),
        html.Div(style={"margin-top": "50px"}),
        html.H5("Time graph"),
        html.Label("X-axis:", htmlFor="samples-time-x"),
        dcc.Dropdown(
            id="samples-time-x",
            options=["Unix time", "From start", "From formation", "From cycling"],
            value="From start",
            multi=False,
        ),
        dcc.Dropdown(
            id="samples-time-units",
            options=["Seconds", "Minutes", "Hours", "Days"],
            value="Hours",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Y-axis:", htmlFor="samples-time-y"),
        dcc.Dropdown(
            id="samples-time-y",
            options=["V (V)"],
            value="V (V)",
            multi=False,
        ),
        html.Div(style={"margin-top": "50px"}),
        html.H5("Cycles graph"),
        html.P("X-axis: Cycle"),
        html.Label("Y-axis:", htmlFor="samples-cycles-y"),
        dcc.Dropdown(
            id="samples-cycles-y",
            options=[
                "Specific discharge capacity (mAh/g)",
                "Efficiency (%)",
            ],
            value="Specific discharge capacity (mAh/g)",
            multi=False,
        ),
        html.Div(style={"margin-top": "50px"}),
        html.H5("One cycle graph"),
        html.Label("X-axis:", htmlFor="samples-cycle-x"),
        dcc.Dropdown(
            id="samples-cycle-x",
            options=["Q (mAh)", "V (V)", "dQdV (mAh/V)"],
            value="Q (mAh)",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Y-axis:", htmlFor="samples-cycle-y"),
        dcc.Dropdown(
            id="samples-cycle-y",
            options=["Q (mAh)", "V (V)", "dQdV (mAh/V)"],
            value="V (V)",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Cycle number:", htmlFor="cycle-number"),
        dcc.Input(
            id="cycle-number",
            type="number",
            placeholder="Cycle number",
            min=1,
            value=1,
            style={"width": "100%"},
        ),
        html.Div(style={"margin-top": "100px"}),
    ],
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
            title="vs time",
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
            title="vs cycle",
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
            title="One cycle",
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
                    className="menu-panel",
                    children=samples_menu,
                    defaultSizePercentage=16,
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
        Output("samples-dropdown", "options"),
        Output("batch-samples-dropdown", "data"),
        Output("batch-edit-samples", "data"),
        Input("samples-store", "data"),
    )
    def update_samples_dropdown(samples: list):
        """Update available samples in the dropdown."""
        options = [{"label": s, "value": s} for s in samples]
        return options, options, options

    # Update the samples data store
    @app.callback(
        Output("samples-data-store", "data"),
        Output("samples-time-y", "options"),
        Output("samples-cycles-y", "options"),
        Input("samples-dropdown", "value"),
        Input("compressed-files", "value"),
        State("samples-data-store", "data"),
        running=[(Output("loading-message-store", "data"), "Loading data...", "")],
    )
    def update_sample_data(samples: list, compressed: list, data: dict) -> tuple[dict, list, list]:
        """Load data for selected samples and put in data store."""
        # Get rid of samples that are no longer selected
        for sample in list(data["data_sample_time"].keys()):
            if sample not in samples:
                data["data_sample_time"].pop(sample)
                if sample in data["data_sample_cycle"]:
                    data["data_sample_cycle"].pop(sample)

        for sample in samples:
            # Check if already in data store
            if sample in data["data_sample_time"]:
                # Check if it's already the correct format
                if 1 not in compressed and not data["data_sample_time"][sample].get("Shrunk", False):
                    continue
                if 1 in compressed and data["data_sample_time"][sample].get("Shrunk", False):
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
            if 1 in compressed and any(f.startswith("shrunk") and f.endswith(".h5") for f in files):
                filepath = next(f for f in files if f.startswith("shrunk") and f.endswith(".h5"))
                df = pd.read_hdf(f"{file_location}/{filepath}")
                data_dict = df.to_dict(orient="list")
                data_dict["Shrunk"] = True
                data["data_sample_time"][sample] = data_dict
            elif any(f.startswith("full") and f.endswith(".h5") for f in files):
                filepath = next(f for f in files if f.startswith("full") and f.endswith(".h5"))
                df = pd.read_hdf(f"{file_location}/{filepath}")
                data["data_sample_time"][sample] = df.to_dict(orient="list")
            elif any(f.startswith("full") and f.endswith(".json.gz") for f in files):
                filepath = next(f for f in files if f.startswith("full") and f.endswith(".json.gz"))
                with gzip.open(f"{file_location}/{filepath}", "rb") as f:
                    data_dict = json.load(f)["data"]
                data["data_sample_time"][sample] = data_dict
            else:
                cycling_files = [
                    os.path.join(file_location, f) for f in files if (f.startswith("snapshot") and f.endswith(".h5"))
                ]
                if not cycling_files:
                    cycling_files = [
                        os.path.join(file_location, f)
                        for f in files
                        if (f.startswith("snapshot") and f.endswith(".json.gz"))
                    ]
                    if not cycling_files:
                        print(f"No cycling files found in {file_location}")
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

        cycles_y_vars = {"Specific discharge capacity (mAh/g)", "Normalised discharge capacity (%)", "Efficiency (%)"}
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
    )
    def update_time_graph(fig: dict, data: dict, xvar: str, xunits: str, yvar: str) -> dict:
        """When data or x/y variables change, update the time graph."""
        fig["data"] = []
        fig["layout"]["xaxis"]["title"] = f"Time ({xunits.lower()})" if xunits else None
        fig["layout"]["yaxis"]["title"] = yvar
        if not data["data_sample_time"] or not xvar or not yvar or not xunits:
            if not data["data_sample_time"]:
                fig["layout"]["title"] = "No data..."
            elif not xvar or not yvar or not xunits:
                fig["layout"]["title"] = "Select x and y variables"
            return fig
        fig["layout"]["title"] = f"{yvar} vs time"
        go_fig = go.Figure(data=fig["data"], layout=fig["layout"])
        multiplier = {"Seconds": 1, "Minutes": 60, "Hours": 3600, "Days": 86400}[xunits]
        for sample, data_dict in data["data_sample_time"].items():
            uts = np.array(data_dict["uts"])
            if xvar == "From start":
                offset = uts[0]
            elif xvar == "From formation":
                offset = uts[next(i for i, x in enumerate(data_dict["Cycle"]) if x >= 1)]
            elif xvar == "From cycling":
                offset = uts[next(i for i, x in enumerate(data_dict["Cycle"]) if x >= 4)]
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
        return go_fig

    # Update the cycles graph
    @app.callback(
        Output("cycles-graph", "figure"),
        State("cycles-graph", "figure"),
        Input("samples-data-store", "data"),
        Input("samples-cycles-y", "value"),
        running=[(Output("loading-message-store", "data"), "Plotting cycles...", "")],
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
            mask_dict["dQdV (mAh/V)"] = smoothed_derivative(mask_dict["V (V)"], mask_dict["Q (mAh)"])
            trace = go.Scattergl(
                x=mask_dict[xvar],
                y=mask_dict[yvar],
                mode="lines",
                name=sample,
                hovertemplate=f"{sample}<br>{xvar}: %{{x}}<br>{yvar}: %{{y}}<extra></extra>",
            )
            fig["data"].append(trace)
        fig["layout"]["title"] = f"{yvar} vs {xvar} for cycle {cycle}"
        return go.Figure(data=fig["data"], layout=fig["layout"])
