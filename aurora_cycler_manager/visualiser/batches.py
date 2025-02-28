"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Batches tab layout and callbacks for the visualiser app.
"""
import json
import os
import textwrap
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash import callback_context as ctx
import dash_mantine_components as dmc
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle
from plotly.colors import sample_colorscale

from aurora_cycler_manager.utils import run_from_sample
from aurora_cycler_manager.visualiser.funcs import correlation_matrix

graph_template = "seaborn"
graph_margin = {"l": 50, "r": 10, "t": 50, "b": 75}

# Define available color scales
cont_color_dict = {}
cont_color_dict.update(px.colors.sequential.__dict__)
cont_color_dict.update(px.colors.diverging.__dict__)
cont_color_dict.update(px.colors.cyclical.__dict__)
cont_color_dict = {k: v for k, v in cont_color_dict.items() if isinstance(v, list) and not k.startswith("__")}
cont_color_options = [{"label": k, "value": k} for k in cont_color_dict]

discrete_color_dict = {}
discrete_color_dict.update(px.colors.qualitative.__dict__)
discrete_color_dict = {k: v for k, v in discrete_color_dict.items() if isinstance(v, list) and not k.startswith("__")}
discrete_color_options = [{"label": k, "value": k} for k in discrete_color_dict]

batches_menu = html.Div(
    style = {"overflow": "scroll", "height": "100%"},
    children = [
        html.Div(style={"margin-top": "10px"}),
        dbc.Button(
            [html.I(className="bi bi-plus-circle-fill me-2"),"Select samples to plot"],
            id="batch-samples-button",
            color="primary",
            style={"width": "100%", "margin-top": "50 px", "margin-bottom": "50px"},
        ),
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
        html.Label("Color by:", htmlFor="batch-cycle-colormap"),
        dcc.Dropdown(
            id="batch-cycle-color",
            options=[
                "Run ID",
            ],
            value="Run ID",
            multi=False,
        ),
        html.Label("Continuous colorscale:", htmlFor="batch-cycle-colormap"),
        dcc.Dropdown(
            id="batch-cycle-colormap",
            options=cont_color_options,
            value="Viridis",
        ),
        html.Label("Discrete colors:", htmlFor="batch-cycle-discrete-colormap"),
        dcc.Dropdown(
            id="batch-cycle-discrete-colormap",
            options=discrete_color_options,
            value="Plotly",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Style by:", htmlFor="batch-cycle-style"),
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

samples_modal = dmc.Modal(
    children=[
        dmc.MultiSelect(
            id="batch-batch-dropdown",
            label="Select batches",
            data=[], # Updated by callback
            value=[],
            clearable=True,
            searchable=True,
            checkIconPosition="right",
        ),
        html.Div(style={"margin-top": "10px"}),
        dmc.MultiSelect(
            id="batch-samples-dropdown",
            label="Select individual samples",
            data=[], # Updated by callback
            value=[],
            clearable=True,
            searchable=True,
            checkIconPosition="right",
        ),
        html.Div(style={"margin-top": "10px"}),
        dbc.Alert([html.I(className="bi bi-info-circle me-2"),"You can also load samples from Database > Samples > View data"], color="light", dismissable=True),
        html.Div(
            children=[
                dbc.Button(
                    "Load", id="batch-yes-close", className="ms-auto", n_clicks=0, color="primary",
                ),
                dbc.Button(
                    "Go back", id="batch-no-close", className="ms-auto", n_clicks=0, color="secondary",
                ),
            ],
            style={"flex": "1 1 auto", "display": "flex", "justify-content": "space-between"},
        ),
    ],
    id="batch-modal",
    title="Select samples to plot",
    centered=True,
    opened=False,
    size="xl",
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
        dcc.Store(id="batches-data-store", data={}),
        dcc.Store(id="trace-style-store", data={}),
        samples_modal,
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
def add_legend_colorbar(fig_dict: dict, sdata: dict) -> go.Figure:
    """Add legend and/or colorbar to figure based on color/style data dict."""
    # Convert figure dict to graph object
    fig = go.Figure(fig_dict)

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
        Output("batch-batch-dropdown", "data"),
        Input("batches-store", "data"),
    )
    def update_batches_dropdown(batches: dict[str, list]) -> list[dict]:
        return [{"label": b, "value": b} for b in batches]

    # When the user clicks the "Select samples to plot" button, open the modal
    @app.callback(
        Output("batch-modal", "opened", allow_duplicate=True),
        Input("batch-samples-button", "n_clicks"),
        Input("batch-yes-close", "n_clicks"),
        Input("batch-no-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_samples_modal(select_clicks: int, yes: int, no: int) -> bool:
        if not ctx.triggered:
            return False
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return button_id == "batch-samples-button"

    # When the user hits yes close, load the selected samples
    @app.callback(
        Output("batches-data-store", "data"),
        Output("batch-cycle-y", "options"),
        Output("batch-cycle-y", "value"),
        Output("batch-cycle-color", "options"),
        Output("batch-cycle-style", "options"),
        Output("batch-modal", "opened", allow_duplicate=True),
        Input("batch-yes-close", "n_clicks"),
        State("batch-samples-dropdown", "value"),
        State("batch-batch-dropdown", "value"),
        State("batches-data-store", "data"),
        State("batches-store", "data"),
        State("batch-cycle-y", "value"),
        prevent_initial_call=True,
    )
    def load_selected_samples(n_clicks: int, samples: list, batches: list, data: dict, batch_defs: dict[str, list], y_val: str):
        """Load the selected samples into the data store."""
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update

        # Add the samples from batches to samples
        sample_set = set(samples)
        for batch in batches:
            sample_set.update(batch_defs[batch])

        # Go through the keys in the data store, if they're not in the samples, remove them
        del_keys = [key for key in data if key not in sample_set]
        for key in del_keys:
            del data[key]

        # Go through samples and add to data
        for s in sample_set:
            if s in data:
                continue
            run_id = _run_from_sample(s)
            file_location = Path(config["Processed snapshots folder path"])/run_id/s/f"cycles.{s}.json"
            if not file_location.exists():
                continue
            with file_location.open(encoding="utf-8") as f:
                data[s] = json.load(f)["data"]

        # y-axis options are lists in data
        # color options are non-lists
        y_vars_set = set()
        color_vars_set = set()
        for sample in data.values():
            y_vars_set.update([k for k,v in sample.items() if isinstance(v,list)])
            color_vars_set.update([k for k,v in sample.items() if v is not None and not isinstance(v,list)])
        y_vars = list(y_vars_set)
        color_vars = list(color_vars_set)

        # Set a default y choice if none is picked
        if y_vars and (not y_val or y_val not in y_vars):
            if "Specific discharge capacity (mAh/g)" in y_vars:
                y_val = "Specific discharge capacity (mAh/g)"
            elif "Discharge capacity (mAh)" in y_vars:
                y_val = "Discharge capacity (mAh)"
            else:
                y_val = y_vars[0]
        # return the new data
        return data, y_vars, y_val, color_vars, color_vars, False

    # Create a list of styles and colors corresponding to the traces
    @app.callback(
        Output("trace-style-store", "data"),
        Input("batches-data-store", "data"),
        Input("batch-cycle-color", "value"),
        Input("batch-cycle-colormap", "value"),
        Input("batch-cycle-discrete-colormap", "value"),
        Input("batch-cycle-style", "value"),
    )
    def update_color_style_store(data: dict, color: str, colormap: list[str], discrete_colormap: list[str], style: str) -> dict:
        """Update the color and style data store based on the selected color and style."""
        # get the color for each trace
        if color:
            color_values = [sample.get(color, None) for sample in data.values()]
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
                    color_values_norm: list[float|None] = [None] * len(color_values)
                    unique_color_labels = [None]
                    unique_color_indices = [0]
            elif color_mode == "categorical":
                    unique_valid_values = sorted({v for v in color_values if v is not None})
                    assigned_values = {}
                    for i, v in enumerate(unique_valid_values):
                        assigned_values[v] = i / (len(unique_valid_values) - 1) if len(unique_valid_values) > 1 else 0
                    color_values_norm = [assigned_values.get(v) if v is not None else None for v in color_values]
                    color_values = [v if v is not None else "None" for v in color_values]
                    unique_color_labels, unique_color_indices = np.unique(color_values, return_index=True)
                    colormap_list = discrete_color_dict.get(discrete_colormap, discrete_color_dict.get("Plotly"))
                    color_map: dict[str,str] = {label: colormap_list[i % len(colormap_list)] for i, label in enumerate(unique_color_labels)}
                    colors = [color_map[str(v)] if v is not None else "rgb(150, 150, 150)" for v in color_values]
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

            if color_mode != "categorical":
                colormap_list = cont_color_dict.get(colormap, cont_color_dict.get("Viridis"))
                colors = [
                    sample_colorscale(colormap_list, [v])[0] if v is not None else "rgb(150, 150, 150)"
                    for v in color_values_norm
                ]

        # If style, add a different style for each in the category
        if style:
            styles = [sample.get(style) for sample in data.values()]
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
        if not data:
            fig["layout"]["title"] = "No data..."
            return fig

        # Add 1/Formation C if Formation C is in the data
        [
            d.update({"1/Formation C": 1 / d["Formation C"] if d["Formation C"] != 0 else 0})
            for d in data.values() if "Formation C" in d
        ]

        fig["layout"]["yaxis"]["title"] = yvar
        fig["layout"]["title"] = f"{yvar} vs cycle"

        # add trace for each sample
        always_show_legend = False
        show_legend = not sdata["color_mode"] or always_show_legend
        for i,sample in enumerate(data.values()):
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
                    "line": {"width": 0.5, "color": "black"},
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
        if not data or len(data) < 3:
            return fig, [], []
        data_correlations = [
            {k:v for k,v in s.items() if v and not isinstance(v, list)}
            for s in data.values()
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
        x = [s.get(xvar) for s in data.values()]
        y = [s.get(yvar) for s in data.values()]

        # Check if axes are categorical or numerical
        x_categorical = any(isinstance(val, str) for val in x)
        y_categorical = any(isinstance(val, str) for val in y)

        if x_categorical:
            fig["layout"]["xaxis"]["type"] = "category"
            x = [s.get(xvar, "None") for s in data.values()]
            x_categories = sorted(set(x))
            fig["layout"]["xaxis"]["categoryorder"] = "array"
            fig["layout"]["xaxis"]["categoryarray"] = x_categories
            # put None at the end
            if "None" in x_categories:
                x_categories.remove("None")
                x_categories.append("None")
        else:
            fig["layout"]["xaxis"]["type"] = "linear"

        if y_categorical:
            fig["layout"]["yaxis"]["type"] = "category"
            y = [s.get(yvar, "None") for s in data.values()]
            y_categories = sorted(set(y))
            fig["layout"]["yaxis"]["categoryorder"] = "array"
            fig["layout"]["yaxis"]["categoryarray"] = y_categories
            # put None at the end
            if "None" in y_categories:
                y_categories.remove("None")
                y_categories.append("None")
        else:
            fig["layout"]["yaxis"]["type"] = "linear"

        hover_info = [
            "Sample ID",
            "N:P ratio",
            "Formation C",
            "Rack position",
            "Run ID",
        ]
        customdata = [[s.get(col,"") for col in hover_info] for s in data.values()]
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
            x=x,
            y=y,
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
        gofig = go.Figure(fig)
        gofig.add_trace(trace)

        if show_legend:
            gofig = add_legend_colorbar(gofig, sdata)
        return gofig
