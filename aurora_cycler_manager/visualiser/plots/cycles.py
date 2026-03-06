"""Copyright © 2026, Empa.

Time-series graph and callback.
"""

import logging

import dash_mantine_components as dmc
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html
from dash.dependencies import MATCH

from aurora_cycler_manager.data_parse import LazySampleDataBundle
from aurora_cycler_manager.visualiser.plots.defaults import graph_margin, graph_template, plot_lazyframe

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
        Output({"type": "cycles-graph", "index": MATCH}, "figure"),
        State({"type": "cycles-graph", "index": MATCH}, "figure"),
        Input("selected-samples", "data"),
        Input({"type": "cycles-graph-x", "index": MATCH}, "value"),
        Input({"type": "cycles-graph-y", "index": MATCH}, "value"),
    )
    def plot_time_series(fig: dict, samples: list[str], xvar: str, yvar: str) -> go.Figure:
        sample_data = {s: LazySampleDataBundle(s).cycles_summary for s in samples}
        return plot_lazyframe(fig, sample_data, xvar, yvar)
