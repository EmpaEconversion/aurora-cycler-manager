# Copyright © 2025-2026, Empa.
"""Samples tab layout and callbacks for the visualiser app."""

import logging

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objs as go
import polars as pl
from dash import Dash, Input, Output, State, dcc, html
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from aurora_cycler_manager.analysis import calc_dqdv
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.data_parse import get_metadata
from aurora_cycler_manager.visualiser.data_cache import cache_stats, get_cycling_frame, get_frame

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
            dmc.InputWrapper(
                dcc.Dropdown(
                    id="samples-dropdown",
                    options=[],
                    value=[],
                    multi=True,
                    labels={"select_all": None, "deselect_all": None},
                    className="dmc",
                    debounce=True,
                    maxHeight=500,
                ),
                label="Select samples",
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
                            "Discharge capacity (mAh)",
                        ],
                        value="Discharge capacity (mAh)",
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
    config={"scrollZoom": True, "displaylogo": False, "toImageButtonOptions": {"format": "svg"}},
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
    config={"scrollZoom": True, "displaylogo": False, "toImageButtonOptions": {"format": "svg"}},
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
    config={"scrollZoom": True, "displaylogo": False, "toImageButtonOptions": {"format": "svg"}},
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
            data={"samples": [], "compressed": True, "metadata": {}},
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
        Output("samples-dropdown", "options"),
        Output("batch-samples-dropdown", "options"),
        Output("batch-edit-samples", "options"),
        Input("samples-store", "data"),
        prevent_initial_call=True,
    )
    def update_samples_dropdown(samples: list) -> tuple[list, list, list]:
        """Update available samples in the dropdown."""
        return samples, samples, samples

    # Update the samples data store
    @app.callback(
        Output("samples-data-store", "data"),
        Output("samples-time-y", "data"),
        Output("samples-cycles-y", "data"),
        Input("samples-dropdown", "value"),
        Input("compressed-files", "checked"),
        running=[(Output("loading-message-store", "data"), "Loading data...", "")],
        prevent_initial_call=True,
    )
    def update_sample_data(samples: list, compressed: bool) -> tuple[dict, list, list]:
        """Load selected samples into the frame cache and put metadata in store."""
        samples = samples or []
        working_set = set(samples)
        metadata = {}
        time_y_vars = {"V (V)"}
        cycles_y_vars = {"Discharge capacity (mAh)"}
        found = []

        for sample in samples:
            df = get_cycling_frame(sample, compressed=compressed, working_set=working_set)
            if df is None:
                logger.info("No cycling found for %s", sample)
                continue
            found.append(sample)
            time_y_vars.update(df.columns)

            sample_metadata = get_metadata(sample)
            metadata[sample] = sample_metadata["sample_data"] if sample_metadata else {}

            cycles = get_frame(sample, "cycles", working_set)
            if cycles is not None:
                cycles_y_vars.update(cycles.columns)

        logger.info("Frame cache: %s", cache_stats())
        data = {"samples": found, "compressed": compressed, "metadata": metadata}
        return data, sorted(time_y_vars), sorted(cycles_y_vars)

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
        if not data["samples"] or not xvar or not yvar or not xunits:
            if not data["samples"]:
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
        working_set = set(data["samples"])
        for sample in data["samples"]:
            df = get_cycling_frame(sample, compressed=data["compressed"], working_set=working_set)
            if df is None:
                continue
            uts = df["uts"].to_numpy()
            if xvar == "From start":
                offset = uts[0]
            elif xvar in ("From formation", "From cycling"):
                thresh = 0 if xvar == "From formation" else data["metadata"].get(sample, {}).get("Formation cycles", 3)
                past = np.flatnonzero(df["Cycle"].to_numpy() > thresh)
                offset = uts[past[0]] if past.size else uts[-1]
            else:
                offset = 0

            trace = go.Scattergl(
                x=(uts - offset) / multiplier,
                y=df[yvar].to_numpy() if yvar in df.columns else [np.nan] * len(uts),
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
        if not data["samples"]:
            fig["layout"]["title"] = "No data..."
            return fig
        working_set = set(data["samples"])
        for sample in data["samples"]:
            cycles = get_frame(sample, "cycles", working_set)
            if cycles is None:
                continue
            cycle_numbers = cycles["Cycle"].to_numpy()
            trace = go.Scattergl(
                x=cycle_numbers,
                y=cycles[yvar].to_numpy() if yvar in cycles.columns else [np.nan] * len(cycle_numbers),
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
        fig["layout"]["xaxis"]["title"] = xvar or "Select x variable"
        fig["layout"]["yaxis"]["title"] = yvar or "Select y variable"
        if not data["samples"]:
            fig["layout"]["title"] = "No data..."
            return fig
        if not xvar or not yvar:
            return fig
        working_set = set(data["samples"])
        for sample in data["samples"]:
            df = get_cycling_frame(sample, compressed=data["compressed"], working_set=working_set)
            one_cycle = df.filter(pl.col("Cycle") == cycle) if df is not None else None
            if one_cycle is None or one_cycle.is_empty():
                # increment colour anyway by adding an empty trace
                fig["data"].append(go.Scattergl())
                continue
            mask_dict = {}
            mask_dict["V (V)"] = one_cycle["V (V)"].to_numpy()
            mask_dict["Q (mAh)"] = one_cycle["dQ (mAh)"].to_numpy().cumsum()
            mask_dict["dQ (mAh)"] = one_cycle["dQ (mAh)"].to_numpy()
            if "dQ/dV (mAh/V)" in [xvar, yvar] or "dQ/dV (mAh/gV)" in [xvar, yvar]:
                if "dQ/dV (mAh/V)" in one_cycle.columns:
                    mask_dict["dQ/dV (mAh/V)"] = one_cycle["dQ/dV (mAh/V)"].to_numpy().astype(float)
                else:
                    mask_dict["dQ/dV (mAh/V)"] = calc_dqdv(
                        mask_dict["V (V)"],
                        mask_dict["Q (mAh)"],
                        mask_dict["dQ (mAh)"],
                    )
            m_mg = None
            if "Q (mAh/g)" in [xvar, yvar] or "dQ/dV (mAh/gV)" in [xvar, yvar]:
                m_mg = data["metadata"].get(sample, {}).get("Cathode active material mass (mg)")
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
