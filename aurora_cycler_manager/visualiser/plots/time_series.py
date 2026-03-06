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
        Output({"type": "ts-graph", "index": MATCH}, "figure", allow_duplicate=True),
        State({"type": "ts-graph", "index": MATCH}, "figure"),
        State("sample:color", "data"),
        Input("selected-samples", "data"),
        Input({"type": "ts-graph-x", "index": MATCH}, "value"),
        Input({"type": "ts-graph-y", "index": MATCH}, "value"),
        Input({"type": "refresh-graph", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def plot_ts(
        fig: dict, sample_color: dict[str, dict], samples: dict[str, str], xvar: str, yvar: str, _trigger: str
    ) -> go.Figure:
        fig["data"] = []
        if not xvar or not yvar or xvar == yvar:
            return go.Figure(fig)
        fig["layout"].setdefault("xaxis", {})["title"] = xvar
        fig["layout"].setdefault("yaxis", {})["title"] = yvar

        sample_data = {s: LazySampleDataBundle(s).cycling_shrunk for s in samples}
        sample_data = {s: d.pipe(enrich_df) if d is not None else None for s, d in sample_data.items()}

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
        Output({"type": "ts-graph", "index": MATCH}, "figure", allow_duplicate=True),
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
