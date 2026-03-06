"""Copyright © 2026, Empa.

Time-series graph and callback.
"""

import logging

import dash_mantine_components as dmc
import plotly.graph_objs as go
import polars as pl
from dash import Dash, Input, Output, Patch, State, dcc, html
from dash.dependencies import MATCH

from aurora_cycler_manager.data_parse import LazySampleDataBundle
from aurora_cycler_manager.visualiser.plots.defaults import graph_margin, graph_template

cycles_options = [
    "Cycle",
    "Charge capacity (mAh)",
    "Discharge capacity (mAh)",
    "Charge energy (mWh)",
    "Discharge energy (mWh)",
    "Charge average current (A)",
    "Discharge average current (A)",
    "Charge average voltage (V)",
    "Discharge average voltage (V)",
    "Coulombic efficiency (%)",
    "Energy efficiency (%)",
    "Voltage efficiency (%)",
    "Delta V (V)",
    "Specific charge capacity (mAh/g)",
    "Specific discharge capacity (mAh/g)",
    "Specific charge energy (mWh/g)",
    "Specific discharge energy (mWh/g)",
    "Normalised discharge capacity (%)",
    "Normalised discharge energy (%)",
]

logger = logging.getLogger(__name__)


def make_cycles_graph(instance_id: str) -> html.Div:
    """Make per-cycle plot and controls."""
    return html.Div(
        id={"type": "cycle-graph-container", "index": instance_id},
        style={
            "height": "100%",
            "width": "100%",
            "display": "flex",
            "flex-direction": "column",
            "overflow": "hidden",
        },
        children=[
            dmc.Group(
                style={"flex-shrink": "0"},
                children=[
                    dmc.Select(
                        id={"type": "cycles-graph-x", "index": instance_id},
                        data=cycles_options,
                        value="Cycle",
                    ),
                    dmc.Text("vs"),
                    dmc.Select(
                        id={"type": "cycles-graph-y", "index": instance_id},
                        data=cycles_options,
                        value="Discharge capacity (mAh)",
                    ),
                ],
            ),
            dcc.Graph(
                id={"type": "cycles-graph", "index": instance_id},
                style={"flex": "1", "min-height": "0"},
                figure={
                    "data": [],
                    "layout": go.Layout(
                        template=graph_template,
                        margin=graph_margin,
                    ),
                },
                config={
                    "scrollZoom": True,
                    "displaylogo": False,
                    "toImageButtonOptions": {"format": "svg", "width": None, "height": None},
                    "responsive": True,
                },
            ),
        ],
    )


def register_cycles_callbacks(app: Dash) -> None:
    """Register callbacks for cycles plot."""

    @app.callback(
        Output({"type": "cycles-graph", "index": MATCH}, "figure", allow_duplicate=True),
        State({"type": "cycles-graph", "index": MATCH}, "figure"),
        State("sample:color", "data"),
        Input("selected-samples", "data"),
        Input({"type": "cycles-graph-x", "index": MATCH}, "value"),
        Input({"type": "cycles-graph-y", "index": MATCH}, "value"),
        Input({"type": "refresh-graph", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def plot_cycles(
        fig: dict, sample_color: dict[str, dict], samples: dict[str, str], xvar: str, yvar: str, _trigger: str
    ) -> go.Figure:
        fig["data"] = []
        if not xvar or not yvar or xvar == yvar:
            return go.Figure(fig)
        fig["layout"].setdefault("xaxis", {})["title"] = xvar
        fig["layout"].setdefault("yaxis", {})["title"] = yvar

        sample_data = {s: LazySampleDataBundle(s).cycles_summary for s in samples}

        frames = []
        for sample, data in sample_data.items():
            if data is None:
                continue
            if xvar not in data.collect_schema().names() or yvar not in data.collect_schema().names():
                continue
            cols = list({xvar, yvar})
            frames.append(data.select(cols).with_columns(pl.lit(sample).alias("_sample")))

        if not frames:
            return go.Figure(fig)

        # Collect once for polars speedup
        collected = pl.concat(frames).collect()

        for (sample,), group in collected.group_by("_sample"):
            fig["data"].append(
                go.Scattergl(
                    x=group[xvar].to_arrow(),
                    y=group[yvar].to_arrow(),
                    name=sample,
                    marker={
                        "color": sample_color.get(sample, "#888888"),
                    },
                )
            )
        return go.Figure(fig)

    @app.callback(
        Output({"type": "cycles-graph", "index": MATCH}, "figure", allow_duplicate=True),
        State("sample:color", "data"),
        Input("redraw-trigger", "data"),
        prevent_initial_call=True,
    )
    def recolor_traces(sample_color: dict[str, str], _trigger: str) -> Patch:
        """Just recolor traces, do not replot."""
        patched = Patch()
        for i, color in enumerate(sample_color.values()):
            patched["data"][i]["line"]["color"] = color
            patched["data"][i]["marker"]["color"] = color
        return patched
