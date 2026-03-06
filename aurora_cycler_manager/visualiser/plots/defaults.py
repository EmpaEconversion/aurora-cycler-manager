"""Defaults and resuable plotting functions."""

import plotly.graph_objs as go
import polars as pl

graph_template = "seaborn"
graph_margin = {"l": 75, "r": 20, "t": 50, "b": 75}


def plot_lazyframe(fig: dict, sample_data: dict[str, pl.LazyFrame | None], xvar: str, yvar: str) -> go.Figure:
    """Plot a lazy frame."""
    fig["data"] = []
    if not xvar or not yvar or xvar == yvar:
        return go.Figure(fig)
    fig["layout"]["xaxis"]["title"] = xvar
    fig["layout"]["yaxis"]["title"] = yvar

    frames = []
    for sample, data in sample_data.items():
        if data is None:
            continue
        if xvar not in data.collect_schema().names() or yvar not in data.collect_schema().names():
            continue
        frames.append(data.select(xvar, yvar, pl.lit(sample).alias("_sample")))

    if not frames:
        return go.Figure(fig)

    # Collect once for polars speedup
    collected = pl.concat(frames).collect()

    for sample, group in collected.group_by("_sample"):
        fig["data"].append(
            go.Scattergl(
                x=group[xvar].to_arrow(),
                y=group[yvar].to_arrow(),
                name=sample[0],
            )
        )

    return go.Figure(fig)
