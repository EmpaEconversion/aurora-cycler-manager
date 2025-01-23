"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Batches tab layout and callbacks for the visualiser app.
"""
import json
import os
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle
from plotly.colors import sample_colorscale

from aurora_cycler_manager.visualiser.funcs import correlation_matrix

graph_template = "seaborn"
graph_margin = {"l": 50, "r": 10, "t": 50, "b": 50}

# Define available color scales
all_color_scales = {}
all_color_scales.update(px.colors.sequential.__dict__)
all_color_scales.update(px.colors.diverging.__dict__)
all_color_scales.update(px.colors.cyclical.__dict__)
all_color_scales = {k: v for k, v in all_color_scales.items() if isinstance(v, list)}
colorscales = [{"label": k, "value": k} for k in all_color_scales]

batches_menu = html.Div(
    style = {"overflow": "scroll", "height": "100%"},
    children = [
        html.H5("Select batches to plot:"),
        dcc.Dropdown(
            id="batches-dropdown",
            options=[], # Updated by callback
            value=[],
            multi=True,
        ),
        html.Div(style={"margin-top": "50px"}),
        html.H5("Cycles graph"),
        html.P("X-axis: Cycle"),
        html.Label("Y-axis:", htmlFor="batch-cycle-y"),
        dcc.Dropdown(
            id="batch-cycle-y",
            options=["Specific discharge capacity (mAh/g)"],
            value="Specific discharge capacity (mAh/g)",
            multi=False,
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Colormap", htmlFor="batch-cycle-colormap"),
        dcc.Dropdown(
            id="batch-cycle-color",
            options=[
                "Run ID",
            ],
            value="Run ID",
            multi=False,
        ),
        dcc.Dropdown(
            id="batch-cycle-colormap",
            options=colorscales,
            value="Viridis",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Style", htmlFor="batch-cycle-style"),
        dcc.Dropdown(
            id="batch-cycle-style",
            options=[
            ],
        ),
        html.Div(style={"margin-top": "50px"}),
        html.H5("Correlation graph"),
        html.Label("X-axis:", htmlFor="batch-correlation-x"),
        dcc.Dropdown(
            id="batch-correlation-x",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Y-axis:", htmlFor="batch-correlation-y"),
        dcc.Dropdown(
            id="batch-correlation-y",
        ),
        html.Div(style={"margin-top": "10px"}),
        dcc.Checklist(
            id="batch-correlation-legend",
            options=[{"label": " Show legend", "value": True}],
            value=[True],
        ),
    ],
)

batch_cycle_graph = dcc.Graph(
    id="batch-cycle-graph",
    figure={
        "data": [],
        "layout": go.Layout(
            template = graph_template,
            margin = graph_margin,
            title = "vs cycle",
            xaxis = {"title": "Cycle"},
            yaxis = {"title": ""},
        ),
    },
    config={
        "scrollZoom": True,
        "displaylogo":False,
        "toImageButtonOptions": {"format": "svg"},
    },
    style={"height": "100%"},
)

batch_correlation_map = dcc.Graph(
    id="batch-correlation-map",
    figure=px.imshow([[0]],color_continuous_scale="balance",aspect="auto",zmin=-1,zmax=1).update_layout(
        template = graph_template,
        margin = graph_margin,
        title="Correlation matrix",
        coloraxis_colorbar={"title": "Correlation","tickvals": [-1, 0, 1], "ticktext": ["-1", "0", "1"]},
        xaxis={"tickfont": {"size": 8}, "title": ""},
        yaxis={"tickfont": {"size": 8}, "title": ""},
    ),
    config={
        "scrollZoom": False,
        "displaylogo":False,
        "modeBarButtonsToRemove" : ["zoom2d","pan2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"],
        "toImageButtonOptions": {"format": "png", "width": 1000, "height": 800},
    },
    style={"height": "100%"},
)

batch_correlation_graph = dcc.Graph(
    id="batch-correlation-graph",
    figure={
        "data": [],
        "layout": go.Layout(
            template=graph_template,
            margin = graph_margin,
            title="",
            xaxis={"title": ""},
            yaxis={"title": ""},
        ),
    },
    config={
        "scrollZoom": True,
        "displaylogo":False,
        "toImageButtonOptions": {"format": "svg"},
    },
    style={"height": "100%"},
)

batches_layout =  html.Div(
    style = {"height": "100%"},
    children = [
        dcc.Store(id="batches-data-store", data={"data_batch_cycle": {}}),
        dcc.Store(id="trace-style-store", data={}),
        PanelGroup(
            id="batches-panel-group",
            direction="horizontal",
            children=[
                Panel(
                    id="batches-menu",
                    className="menu-panel",
                    children=batches_menu,
                    defaultSizePercentage=16,
                    collapsible=True,
                ),
                PanelResizeHandle(html.Div(className="resize-handle-horizontal")),
                Panel(
                    id="graphs",
                    minSizePercentage=50,
                    children=[
                        PanelGroup(
                            id="graph group",
                            direction="vertical",
                            children=[
                                Panel(
                                    batch_cycle_graph,
                                    id="top graph",
                                ),
                                PanelResizeHandle(
                                    html.Div(className="resize-handle-vertical"),
                                ),
                                Panel(
                                    id="bottom graphs",
                                    children=[
                                        PanelGroup(
                                            id="bottom graph group",
                                            direction="horizontal",
                                            children=[
                                                Panel(
                                                    batch_correlation_map,
                                                    id="bottom left graph",
                                                ),
                                                PanelResizeHandle(
                                                    html.Div(className="resize-handle-horizontal"),
                                                ),
                                                Panel(
                                                    batch_correlation_graph,
                                                    id="bottom right graph",
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

#------------------------------ USEFUL FUNCTIONS ------------------------------#
def add_legend_colorbar(fig: dict, sdata: dict) -> go.Figure:
    """Add legend and/or colorbar to figure based on color/style data dict."""
    # Convert figure dict to graph object
    fig = go.Figure(fig)

    # If there is a numerical color scale, add a colorbar
    if sdata["color_mode"] == "numerical":
        colorbar_trace = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "color": [sdata["cmin"], sdata["cmax"]],
                "colorscale": sdata["colormap"],
                "showscale": True,
                "colorbar": {"thickness": 20, "title": {"text":sdata["color_by"], "side": "right"}},
            },
            showlegend=False,
        )
        fig.add_trace(colorbar_trace)

    # If there is a categorical color scale, add a legend by adding fake traces
    elif sdata["color_mode"]:
        title = "<br>".join(textwrap.wrap(sdata["color_by"], width=24))
        for uval,uind in zip(sdata["unique_color_labels"],sdata["unique_color_indices"]):
            if isinstance(uval, float):
                label = f"{uval:.6g}"
            elif isinstance(uval, int):
                label = f"{uval:d}"
            elif isinstance(uval, str):
                # wrap to prevent long strings from breaking the layout
                label = "<br>".join(textwrap.wrap(uval, width=24))
            else:
                label = str(uval)
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines+markers",
                line={"width": 1},
                marker={"size": 8, "color": sdata["colors"][uind]},
                legendgroup="color",
                legendgrouptitle={"text": title},
                name=label,
                showlegend=True,
            ))

    # If there markers are styled, add a legend by adding fake traces
    if sdata["symbols"]:
        title = "<br>".join(textwrap.wrap(sdata["style_by"], width=24))
        for ustyle,uind in zip(sdata["unique_style_labels"],sdata["unique_style_indices"]):
            if isinstance(ustyle, float):
                label = f"{ustyle:.6g}"
            elif isinstance(ustyle, int):
                label = f"{ustyle:d}"
            elif isinstance(ustyle, str):
                # wrap to prevent long strings from breaking the layout
                label = "<br>".join(textwrap.wrap(ustyle, width=24))
            else:
                label = str(ustyle)
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines+markers",
                marker={"size": 8, "color": "rgb(0, 0, 0)", "symbol": sdata["symbols"][uind]},
                legendgroup="style",
                legendgrouptitle={"text": title},
                name=label,
                showlegend=True,
            ))

    # Adjust the layout to prevent overlap
    fig.update_layout(
        showlegend=True,
        legend={
            "x": 1,
            "y": 1,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255, 255, 255, 0.5)",
        },
        coloraxis_colorbar={
            "x": 1,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return fig

#----------------------------- BATCHES CALLBACKS ------------------------------#

def register_batches_callbacks(app: Dash, config: dict) -> None:
    """Register all callbacks for the batches tab."""

    # Batch list has updated, update dropdowns
    @app.callback(
        Output("batches-dropdown", "options"),
        Input("batches-store", "data"),
    )
    def update_batches_dropdown(batches: list) -> list[dict]:
        return [{"label": name, "value": name} for name in batches]

    # Update the batches data store
    @app.callback(
        Output("batches-data-store", "data"),
        Output("batch-cycle-y", "options"),
        Output("batch-cycle-y", "value"),
        Output("batch-cycle-color", "options"),
        Output("batch-cycle-style", "options"),
        Input("batches-dropdown", "value"),
        State("batch-cycle-y", "value"),
        State("batches-data-store", "data"),
    )
    def update_batch_data(batches: list, y_val: str, data: dict) -> tuple[dict, list[str], str, list[str], list[str]]:
        """Update the data store when the user selects new batches."""
        # TODO: add faster way to add/subtract data instead of re-reading all files
        data_folder = config["Batches folder path"]
        data["data_batch_cycle"] = []
        for batch in batches:
            file_location = os.path.join(data_folder, batch)
            files = os.listdir(file_location)
            try:
                analysed_file = next(f for f in files if (f.startswith("batch") and f.endswith(".json")))
            except StopIteration:
                continue
            with open(f"{file_location}/{analysed_file}", encoding="utf-8") as f:
                json_data = json.load(f)
            # add batch to every sample in json_data["data"]
            [sample.update({"Batch": batch}) for sample in json_data["data"]]
            data["data_batch_cycle"] += json_data["data"]

        # Remove duplicate samples
        sample_id = [d["Sample ID"] for d in data["data_batch_cycle"]]
        data["data_batch_cycle"] = [d for d in data["data_batch_cycle"] if sample_id.count(d["Sample ID"]) == 1]

        # y-axis options are lists in data
        # color options are non-lists
        y_vars = set()
        color_vars = set()
        for data_dict in data["data_batch_cycle"]:
            y_vars.update([k for k,v in data_dict.items() if isinstance(v,list)])
            color_vars.update([k for k,v in data_dict.items() if not isinstance(v,list)])
        y_vars = list(y_vars)
        color_vars = list(color_vars)

        # Set a default y choice if none is picked
        if y_vars and (not y_val or y_val not in y_vars):
            if "Specific discharge capacity (mAh/g)" in y_vars:
                y_val = "Specific discharge capacity (mAh/g)"
            elif "Discharge capacity (mAh)" in y_vars:
                y_val = "Discharge capacity (mAh)"
            else:
                y_val = y_vars[0]

        return data, y_vars, y_val, color_vars, color_vars

    # Create a list of styles and colors corresponding to the traces
    @app.callback(
        Output("trace-style-store", "data"),
        Input("batches-data-store", "data"),
        Input("batch-cycle-color", "value"),
        Input("batch-cycle-colormap", "value"),
        Input("batch-cycle-style", "value"),
    )
    def update_color_style_store(data: dict, color: str, colormap: str, style: str) -> dict:
        """Update the color and style data store based on the selected color and style."""
        # get the color for each trace
        if color:
            colormap = all_color_scales.get(colormap, all_color_scales.get("Viridis"))
            color_values = [sample.get(color, None) for sample in data["data_batch_cycle"]]
            cmin, cmax = 0, 1

            # Try to figure out the coloring mode
            if all(v is None for v in color_values):
                color_mode = "none"
            elif len(set(color_values)) == 1:
                color_mode = "single_value"
            elif not all(isinstance(v, (int, float)) or v is None for v in color_values):
                color_mode = "categorical"
            elif len(set(color_values)) < 5:
                color_mode = "numerical_categorical"
            else:
                color_mode = "numerical"

            if color_mode == "none":
                    color_values_norm = [None] * len(color_values)
                    unique_color_labels = [None]
                    unique_color_indices = [0]
            elif color_mode == "categorical":
                    color_values = [v if v is not None else "None" for v in color_values]
                    len_unique_colors = len(set(color_values))
                    color_values_norm = [list(set(color_values)).index(v)/(len_unique_colors-1) for v in color_values]
                    unique_color_labels, unique_color_indices = np.unique(color_values, return_index=True)
            elif color_mode == "numerical_categorical":
                    cmin = min([v for v in color_values if v is not None])
                    cmax = max([v for v in color_values if v is not None])
                    color_values_norm = [(v - cmin) / (cmax - cmin) if v else None for v in color_values]
                    unique_color_labels, unique_color_indices = np.unique(color_values, return_index=True)
            elif color_mode == "numerical":
                    cmin = min([v for v in color_values if v is not None])
                    cmax = max([v for v in color_values if v is not None])
                    color_values_norm = [(v - cmin) / (cmax - cmin) if v else None for v in color_values]
                    unique_color_labels = [None]
                    unique_color_indices = [0]
            elif color_mode == "single_value":
                    color_values_norm = [0.5] * len(color_values)
                    unique_color_labels = [color_values[0]]
                    unique_color_indices = [0]

            colors = [
                sample_colorscale(colormap, [v])[0] if v is not None else "rgb(150, 150, 150)"
                for v in color_values_norm
            ]

        # If style, add a different style for each in the category
        if style:
            styles = [sample[style] for sample in data["data_batch_cycle"]]
            styles = [s if s is not None else "None" for s in styles]
            unique_style_labels, unique_style_indices = np.unique(styles, return_index=True)
            symbols = [list(set(styles)).index(v) for v in styles]
            # to keep symbol values in the ranges 0-32,100-132,200-224
            symbols = [(s%88)%32 + 100*((s%88)//32) for s in symbols]

        return {
            "colormap": colormap if color else None,
            "color_by": color if color else None,
            "color_mode": color_mode if color else None,
            "colors": colors if color else None,
            "color_values": color_values if color else None,
            "unique_color_labels": unique_color_labels if color else None,
            "unique_color_indices": unique_color_indices if color else None,
            "cmin": cmin if color else None,
            "cmax": cmax if color else None,
            "style_by": style if style else None,
            "symbols": symbols if style else None,
            "unique_style_labels": unique_style_labels if style else None,
            "unique_style_indices": unique_style_indices if style else None,
        }

    # Update the batch cycle graph
    @app.callback(
        Output("batch-cycle-graph", "figure"),
        State("batch-cycle-graph", "figure"),
        State("batches-data-store", "data"),
        Input("trace-style-store", "data"),
        Input("batch-cycle-y", "value"),
    )
    def update_batch_cycle_graph(fig: dict, data: dict, sdata: dict, yvar: str) -> go.Figure:
        # remove old data
        fig["data"] = []
        if not data or not data["data_batch_cycle"]:
            fig["layout"]["title"] = "No data..."
            return fig

        # data["data_batch_cycle"] is a list of dicts of samples
        if not data["data_batch_cycle"]:
            fig["layout"]["title"] = "No data..."
            return fig
        # Add 1/Formation C if Formation C is in the data
        [
            d.update({"1/Formation C": 1 / d["Formation C"] if d["Formation C"] != 0 else 0})
            for d in data["data_batch_cycle"] if "Formation C" in d
        ]

        fig["layout"]["yaxis"]["title"] = yvar
        fig["layout"]["title"] = f"{yvar} vs cycle"

        # add trace for each sample
        always_show_legend = False
        show_legend = not sdata["color_mode"] or always_show_legend
        for i,sample in enumerate(data["data_batch_cycle"]):
            color_label = sample.get(sdata["color_by"],"") if sdata["color_by"] else ""
            if isinstance(color_label, float):
                color_label = f"{color_label:.6g}"
            style_label = sample.get(sdata["style_by"],"") if sdata["style_by"] else ""
            if isinstance(style_label, float):
                style_label = f"{style_label:.6g}"
            hovertemplate="<br>".join(
                [
                    f"<b>{sample['Sample ID']}</b>",
                    "Cycle: %{x}",
                    f"{yvar}: %{{y}}",
                ] +
                ([f"{sdata['color_by']}: {color_label}"] if sdata["color_by"] else []) +
                ([f"{sdata['style_by']}: {style_label}"] if sdata["style_by"] else []) +
                ["<extra></extra>"],
            )
            trace = go.Scattergl(
                x=sample["Cycle"],
                y=sample[yvar],
                mode="lines+markers",
                name=sample["Sample ID"],
                line={"width": 1},
                marker={
                    "size": 8,
                    "color": sdata["colors"][i] if sdata["colors"] else None,
                    "symbol": sdata["symbols"][i] if sdata["symbols"] else None,
                    "line": {"width": 0.5, "color": "white"},
                },
                showlegend=show_legend,
                hovertemplate=hovertemplate,
            )
            fig["data"].append(trace)

        return add_legend_colorbar(fig, sdata)

    # Update the correlation map
    @app.callback(
        Output("batch-correlation-map", "figure"),
        Output("batch-correlation-x", "options"),
        Output("batch-correlation-y", "options"),
        State("batch-correlation-map", "figure"),
        Input("batches-data-store", "data"),
    )
    def update_correlation_map(fig: dict, data: dict) -> tuple[dict, list[dict], list[dict]]:
        """Update correlation map when new data is loaded."""
        # data is a list of dicts
        fig["data"] = []
        if not data["data_batch_cycle"]:
            return fig, [], []
        data_correlations = [
            {k:v for k,v in s.items() if v and not isinstance(v, list)}
            for s in data["data_batch_cycle"]
        ]
        dfs = [pd.DataFrame(d, index=[0]) for d in data_correlations]
        if not dfs:
            return fig, [], []
        df = pd.concat(dfs, ignore_index=True)

        if "Formation C" in df.columns:
            df["1/Formation C"] = 1 / df["Formation C"]

        # remove columns where all values are the same
        df = df.loc[:, df.nunique() > 1]

        # remove other unnecessary columns
        columns_not_needed = [
            "Sample ID",
            "Last efficiency (%)",
            "Last specific discharge capacity (mAh/g)",
            "Capacity loss (%)",
        ]
        df = df.drop(columns=columns_not_needed)

        # sort columns reverse alphabetically
        df = df.reindex(sorted(df.columns), axis=1)
        options = df.columns
        df.columns = ["<br>".join(textwrap.wrap(col,width=24)) for col in df.columns]

        # Calculate the correlation matrix
        corr = correlation_matrix(df)

        # Use Plotly Express to create the heatmap
        fig["data"] = [
            go.Heatmap(
                z=corr,
                x=df.columns,
                y=df.columns,
                colorscale="balance",
                zmin=-1,
                zmax=1,
                hoverongaps=False,
                hoverinfo="x+y+z",
            ),
        ]

        return fig, options, options

    # On clicking the correlation map, update the X-axis and Y-axis dropdowns
    @app.callback(
        Output("batch-correlation-x", "value"),
        Output("batch-correlation-y", "value"),
        Input("batch-correlation-map", "clickData"),
    )
    def update_correlation_vars(click_data: dict) -> tuple[str, str]:
        """Update the x and y variables based on the clicked data."""
        if not click_data:
            return no_update
        point = click_data["points"][0]
        xvar = point["x"].replace("<br>", " ")
        yvar = point["y"].replace("<br>", " ")
        return xvar, yvar

    # On changing x and y axes, update the correlation graph
    @app.callback(
        Output("batch-correlation-graph", "figure"),
        State("batch-correlation-graph", "figure"),
        State("batches-data-store", "data"),
        Input("trace-style-store", "data"),
        Input("batch-correlation-x", "value"),
        Input("batch-correlation-y", "value"),
        Input("batch-correlation-legend", "value"),
    )
    def update_correlation_graph(
        fig: dict,
        data: dict,
        sdata: dict,
        xvar: str,
        yvar: str,
        show_legend: bool,
    ) -> go.Figure:
        """Update the correlation graph based on the selected x and y variables."""
        fig["data"] = []
        if not xvar or not yvar:
            return fig
        fig["layout"]["title"] = f"{xvar} vs {yvar}"
        fig["layout"]["xaxis"]["title"] = xvar
        fig["layout"]["yaxis"]["title"] = yvar
        hover_info = [
            "Sample ID",
            "Actual N:P ratio",
            "Formation C",
            "Rack position",
            "Run ID",
        ]
        customdata = [[s.get(col,"") for col in hover_info] for s in data["data_batch_cycle"]]
        hovertemplate="<br>".join(
            [
                "Sample ID: %{customdata[0]}",
                f"{xvar}: %{{x}}",
                f"{yvar}: %{{y}}",
            ] +
            [f"{col}: %{{customdata[{i+1}]}}" for i, col in enumerate(hover_info[1:])] +
            ["<extra></extra>"],
        )
        trace = go.Scatter(
            x=[s[xvar] for s in data["data_batch_cycle"]],
            y=[s[yvar] for s in data["data_batch_cycle"]],
            mode="markers",
            marker={
                "size": 10,
                "color": sdata["colors"] if sdata["colors"] else None,
                "symbol": sdata["symbols"] if sdata["symbols"] else None,
                "line": {"width": 1, "color": "black"},
            },
            showlegend=False,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )

        fig = go.Figure(fig)
        fig.add_trace(trace)
        if show_legend:
            fig = add_legend_colorbar(fig, sdata)
        return fig
