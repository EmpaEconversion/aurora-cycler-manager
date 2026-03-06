"""Copyright © 2026, Empa.

EIS graph and callback.
"""

import logging

import dash_mantine_components as dmc
import plotly.graph_objs as go
import polars as pl
from dash import Dash, Input, Output, Patch, State, dcc, html
from dash.dependencies import MATCH

from aurora_cycler_manager.data_parse import LazySampleDataBundle
from aurora_cycler_manager.visualiser.plots.defaults import graph_margin, graph_template

eis_options = ["f (Hz)", "Re(Z) (ohm)", "Im(Z) (ohm)", "-Im(Z) (ohm)", "|Z| (ohm)", "uts", "V (V)"]
logger = logging.getLogger(__name__)


def enrich_df(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add derived columns."""
    return df.with_columns(
        [
            (-pl.col("Im(Z) (ohm)")).alias("-Im(Z) (ohm)"),
            ((pl.col("Re(Z) (ohm)") ** 2 + pl.col("Im(Z) (ohm)") ** 2) ** 0.5).alias("|Z| (ohm)"),
        ]
    )


def make_eis_graph(instance_id: str) -> html.Div:
    """Generate EIS plot and controls."""
    return html.Div(
        id={"type": "eis-graph-container", "index": instance_id},
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
                    dmc.Select(id={"type": "eis-graph-x", "index": instance_id}, data=eis_options, value="Re(Z) (ohm)"),
                    dmc.Text("vs"),
                    dmc.Select(
                        id={"type": "eis-graph-y", "index": instance_id}, data=eis_options, value="-Im(Z) (ohm)"
                    ),
                ],
            ),
            dcc.Graph(
                id={"type": "eis-graph", "index": instance_id},
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


def register_eis_callbacks(app: Dash) -> None:
    """Register callbacks for eis plot."""

    @app.callback(
        Output({"type": "eis-graph", "index": MATCH}, "figure", allow_duplicate=True),
        State({"type": "eis-graph", "index": MATCH}, "figure"),
        State("sample:color", "data"),
        Input("selected-samples", "data"),
        Input({"type": "eis-graph-x", "index": MATCH}, "value"),
        Input({"type": "eis-graph-y", "index": MATCH}, "value"),
        Input({"type": "refresh-graph", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def plot_eis(
        fig: dict, sample_color: dict[str, dict], samples: dict[str, str], xvar: str, yvar: str, _trigger: str
    ) -> go.Figure:
        fig["data"] = []
        if not xvar or not yvar or xvar == yvar:
            return go.Figure(fig)
        fig["layout"].setdefault("xaxis", {})["title"] = xvar
        fig["layout"].setdefault("yaxis", {})["title"] = yvar

        sample_data = {s: LazySampleDataBundle(s).eis for s in samples}
        sample_data = {s: d.pipe(enrich_df) if d is not None else None for s, d in sample_data.items()}

        frames = []
        for sample, data in sample_data.items():
            if data is None:
                continue
            if xvar not in data.collect_schema().names() or yvar not in data.collect_schema().names():
                continue
            cols = list({xvar, yvar, "Cycle", "uts"})
            frames.append(data.select(cols).with_columns(pl.lit(sample).alias("_sample")))

        if not frames:
            return go.Figure(fig)

        # Collect once for polars speedup
        collected = pl.concat(frames).sort("uts").collect()

        for (sample, cycle), group in collected.group_by("_sample", "Cycle", maintain_order=True):
            fig["data"].append(
                go.Scattergl(
                    x=group[xvar].to_arrow(),
                    y=group[yvar].to_arrow(),
                    name=f"{sample}: Cycle {cycle}",
                    marker={
                        "color": sample_color.get(sample, "#888888"),
                    },
                )
            )

        return go.Figure(fig)

    @app.callback(
        Output({"type": "eis-graph", "index": MATCH}, "figure", allow_duplicate=True),
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
