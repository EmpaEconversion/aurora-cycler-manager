"""Copyright © 2026, Empa.

Time-series graph and callback.
"""

import logging

import dash_mantine_components as dmc
import plotly.graph_objs as go
import polars as pl
from dash import Dash, Input, Output, State, dcc, html
from dash.dependencies import MATCH

from aurora_cycler_manager.data_parse import LazySampleDataBundle
from aurora_cycler_manager.visualiser.plots.defaults import graph_margin, graph_template, plot_lazyframe

ts_options = [
    "Total time (s)",
    "Total time (m)",
    "Total time (h)",
    "Datetime",
    "V (V)",
    "I (A)",
    "Q (mAh)",
    "E (mWh)",
    "Cycle",
]
logger = logging.getLogger(__name__)


def enrich_df(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add derived columns."""
    return df.with_columns(
        [
            (pl.col("uts") - pl.col("uts").first()).alias("Total time (s)"),
            ((pl.col("uts") - pl.col("uts").first()) / 60).alias("Total time (m)"),
            ((pl.col("uts") - pl.col("uts").first()) / 3600).alias("Total time (h)"),
            ((pl.col("uts") - pl.col("uts").first()) / 86400).alias("Total time (d)"),
            pl.from_epoch(pl.col("uts")).alias("Datetime"),
            (pl.col("dQ (mAh)").cum_sum()).alias("Q (mAh)"),
            ((pl.col("dQ (mAh)") * pl.col("V (V)")).cum_sum()).alias("E (mWh)"),
        ]
    )


def make_ts_graph(instance_id: str) -> html.Div:
    """Generate a time-series plot controls and empty graph."""
    return html.Div(
        id={"type": "ts-graph-container", "index": instance_id},
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
                        id={"type": "ts-graph-x", "index": instance_id},
                        data=ts_options,
                        value="Total time (h)",
                    ),
                    dmc.Text("vs"),
                    dmc.Select(
                        id={"type": "ts-graph-y", "index": instance_id},
                        data=ts_options,
                        value="V (V)",
                    ),
                ],
            ),
            dcc.Graph(
                id={"type": "ts-graph", "index": instance_id},
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
                },
                responsive=True,
            ),
        ],
    )


def register_ts_callbacks(app: Dash) -> None:
    """Register time series plot callbacks."""

    @app.callback(
        Output({"type": "ts-graph", "index": MATCH}, "figure"),
        State({"type": "ts-graph", "index": MATCH}, "figure"),
        Input("selected-samples", "data"),
        Input({"type": "ts-graph-x", "index": MATCH}, "value"),
        Input({"type": "ts-graph-y", "index": MATCH}, "value"),
    )
    def plot_time_series(fig: dict, samples: list[str], xvar: str, yvar: str) -> go.Figure:
        sample_data = {s: LazySampleDataBundle(s).cycling_shrunk for s in samples}
        sample_data = {s: d.pipe(enrich_df) if d is not None else None for s, d in sample_data.items()}
        return plot_lazyframe(fig, sample_data, xvar, yvar)
