"""Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Batches tab layout and callbacks for the visualiser app.
"""
import json
import os
import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from aurora_cycler_manager.visualiser.funcs import correlation_matrix

graph_template = "seaborn"
graph_margin = {"l": 50, "r": 10, "t": 50, "b": 50}
colorscales = px.colors.named_colorscales()

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
            value="turbo",
        ),
        html.Div(style={"margin-top": "10px"}),
        html.Label("Style", htmlFor="batch-cycle-style"),
        dcc.Dropdown(
            id="batch-cycle-style",
            options=[
            ],
            multi=False,
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
        html.Label("Colormap", htmlFor="batch-correlation-color"),
        dcc.Dropdown(
            id="batch-correlation-color",
            options=[
                "Run ID",
            ],
            value="Run ID",
            multi=False,
        ),
        dcc.Dropdown(
            id="batch-correlation-colorscale",
            options=colorscales,
            value="turbo",
            multi=False,
        ),
        html.Div(style={"margin-top": "100px"}),
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
            xaxis = {"title": "X-axis Title"},
            yaxis = {"title": "Y-axis Title"},
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
        coloraxis_colorbar={"title": "Correlation","tickvals": [-1, 0, 1], "ticktext": ["-1", "0", "1"]},
        xaxis={"tickfont": {"size": 8}},
        yaxis={"tickfont": {"size": 8}},
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
            title="params",
            xaxis={"title": "X-axis Title"},
            yaxis={"title": "Y-axis Title"},
        ),
    },
    config={
        "scrollZoom": True,
        "displaylogo":False,
        "toImageButtonOptions": {"format": "svg"}
    },
    style={"height": "100%"},
)

batches_layout =  html.Div(
    style = {"height": "100%"},
    children = [
        dcc.Store(id="batches-data-store", data={"data_batch_cycle": {}}),
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

#----------------------------- BATCHES CALLBACKS ------------------------------#

def register_batches_callbacks(app: Dash, config: dict) -> None:
    """Register all callbacks for the batches tab."""

    # Batch list has updated, update dropdowns
    @app.callback(
        Output("batches-dropdown", "options"),
        Input("batches-store", "data"),
    )
    def update_batches_dropdown(batches):
        return [{"label": name, "value": name} for name in batches]

    # Update the batches data store
    @app.callback(
        Output("batches-data-store", "data"),
        Output("batch-cycle-y", "options"),
        Output("batch-cycle-y", "value"),
        Output("batch-cycle-color", "options"),
        Output("batch-cycle-style", "options"),
        Output("batch-correlation-color", "options"),
        Input("batches-dropdown", "value"),
        State("batch-cycle-y", "value"),
        State("batches-data-store", "data"),
    )
    def update_batch_data(batches, y_val, data):
        # TODO add faster way to add/subtract data instead of re-reading all files
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
        print(y_vars and (not y_val or y_val not in y_vars))
        if y_vars and (not y_val or y_val not in y_vars):
            if "Specific discharge capacity (mAh/g)" in y_vars:
                y_val = "Specific discharge capacity (mAh/g)"
            elif "Discharge capacity (mAh)" in y_vars:
                y_val = "Discharge capacity (mAh)"
            else:
                y_val = y_vars[0]

        return data, y_vars, y_val, color_vars, color_vars, color_vars

    # Update the batch cycle graph
    @app.callback(
        Output("batch-cycle-graph", "figure"),
        State("batch-cycle-graph", "figure"),
        Input("batches-data-store", "data"),
        Input("batch-cycle-y", "value"),
        Input("batch-cycle-color", "value"),
        Input("batch-cycle-colormap", "value"),
        Input("batch-cycle-style", "value"),
    )
    def update_batch_cycle_graph(fig, data, variable, color, colormap, style):
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
        [d.update({"1/Formation C": 1 / d["Formation C"] if d["Formation C"] != 0 else 0}) for d in data["data_batch_cycle"] if "Formation C" in d]

        fig["layout"]["yaxis"]["title"] = variable
        fig["layout"]["title"] = f"{variable} vs cycle"

        # add trace for each sample
        for sample in data["data_batch_cycle"]:
            trace = go.Scattergl(
                x=sample["Cycle"],
                y=sample[variable],
                mode="lines+markers",
                name=sample["Sample ID"],
                line={"width": 1},
                marker={"size": 5},
            )
            fig["data"].append(trace)

        return fig

    # Update the correlation map
    @app.callback(
        Output("batch-correlation-map", "figure"),
        Output("batch-correlation-x", "options"),
        Output("batch-correlation-y", "options"),
        State("batch-correlation-map", "figure"),
        Input("batches-data-store", "data"),
    )
    def update_correlation_map(fig, data):
        # data is a list of dicts
        fig["data"] = []
        if not data["data_batch_cycle"]:
            return fig, [], []
        data_correlations = [{k:v for k,v in s.items() if v and not isinstance(v, list)} for s in data["data_batch_cycle"]]
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

        def customwrap(s,width=30):
            return "<br>".join(textwrap.wrap(s,width=width))

        options = df.columns
        df.columns = [customwrap(col) for col in df.columns]

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
    def update_correlation_vars(clickData):
        if not clickData:
            return no_update
        point = clickData["points"][0]
        xvar = point["x"].replace("<br>", " ")
        yvar = point["y"].replace("<br>", " ")
        return xvar, yvar

    # On changing x and y axes, update the correlation graph
    @app.callback(
        Output("batch-correlation-graph", "figure"),
        State("batch-correlation-graph", "figure"),
        Input("batches-data-store", "data"),
        Input("batch-correlation-x", "value"),
        Input("batch-correlation-y", "value"),
        Input("batch-correlation-color", "value"),
        Input("batch-correlation-colorscale", "value"),
    )
    def update_correlation_graph(fig, data, xvar, yvar, color, colormap):
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
        trace = go.Scatter(
            x=[s[xvar] for s in data["data_batch_cycle"]],
            y=[s[yvar] for s in data["data_batch_cycle"]],
            mode="markers",
            marker={
                "size": 10,
                "line": {"width": 1, "color": "black"},
            },
            customdata=customdata,
            hovertemplate="<br>".join(
                [
                    "Sample ID: %{customdata[0]}",
                    f"{xvar}: %{{x}}",
                    f"{yvar}: %{{y}}",
                ] +
                [f"{col}: %{{customdata[{i+1}]}}" for i, col in enumerate(hover_info[1:])] +
                ["<extra></extra>"],
            ),
        )
        fig["data"] = [trace]
        # TODO add colours
        return fig
