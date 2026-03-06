"""Copyright © 2026, Empa.

Plotting tab layout and callbacks for the visualiser app.
"""

import logging

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, dcc, html
from dash import callback_context as ctx
from dash.dependencies import MATCH
from dash.exceptions import PreventUpdate
from dash_resizable_panels import Panel, PanelGroup, PanelResizeHandle

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.visualiser.plots.cycles import make_cycles_graph, register_cycles_callbacks
from aurora_cycler_manager.visualiser.plots.eis import make_eis_graph, register_eis_callbacks
from aurora_cycler_manager.visualiser.plots.time_series import make_ts_graph, register_ts_callbacks

CONFIG = get_config()
logger = logging.getLogger(__name__)

# ----------------------------- SAMPLE SELECT ----------------------------- #

samples_modal = dmc.Modal(
    children=[
        dmc.Stack(
            children=[
                dmc.MultiSelect(
                    id="plotting-batch-select-dropdown",
                    label="Select batches",
                    data=[],  # Updated by callback
                    value=[],
                    clearable=True,
                    searchable=True,
                    checkIconPosition="right",
                ),
                dmc.MultiSelect(
                    id="plotting-samples-select-dropdown",
                    label="Select individual samples",
                    data=[],  # Updated by callback
                    value=[],
                    clearable=True,
                    searchable=True,
                    checkIconPosition="right",
                ),
                dmc.Button(
                    "Load",
                    id="plotting-samples-select-yes-close",
                    n_clicks=0,
                ),
            ],
        ),
    ],
    id="plotting-samples-select-modal",
    title="Select samples to plot",
    centered=True,
    opened=False,
    size="xl",
)

# ----------------------------- GRAPH BLOCKS ----------------------------- #


def make_graph_block(instance_id: str) -> html.Div:
    """Create a block that holds a graph selector and a graph."""
    return html.Div(
        id={"type": "graph-block", "index": instance_id},
        key=instance_id,
        style={
            "height": "100%",
            "width": "100%",
            "display": "flex",
            "flex-direction": "column",
            "overflow": "hidden",
        },
        children=[
            # Fixed height controls strip
            html.Div(
                dmc.Select(
                    id={"type": "graph-type-selector", "index": instance_id},
                    value=DEFAULT_PLOTS[instance_id],
                    data=[{"label": k, "value": k} for k in GRAPH_FACTORIES],
                    clearable=False,
                ),
                style={"flex-shrink": "0"},  # never shrink the controls
            ),
            # Graph takes all remaining space
            html.Div(
                id={"type": "graph-container", "index": instance_id},
                style={
                    "flex": "1",  # grow to fill remaining height
                    "min-height": "0",  # critical — allows flex child to shrink below content size
                    "overflow": "hidden",
                },
            ),
        ],
    )


# ---------------------------- GRAPH FACTORY ----------------------------- #

DEFAULT_PLOTS = {
    "A": "Time series",
    "B": "Cycles",
    "C": "EIS",
}

DEFAULT_AXES = {
    "Time series": {"xvar": "Total time (h)", "yvar": "V (V)"},
    "Cycles": {"xvar": "Cycle", "yvar": "Discharge capacity (mAh)"},
    "EIS": {"xvar": "Re(Z) (ohm)", "yvar": "Im(Z) (ohm)"},
}

GRAPH_FACTORIES = {
    "Time series": make_ts_graph,
    "Cycles": make_cycles_graph,
    "EIS": make_eis_graph,
    # TODO: correlation graph and map
}


def graph_factory(graph_type: str, instance_id: str) -> html.Div:
    """Make the right kind of graph."""
    if not graph_type:
        return html.Div()
    return GRAPH_FACTORIES[graph_type](instance_id)


# ----------------------------- MAIN LAYOUT ----------------------------- #
panels_container = PanelGroup(
    id="panels-container",
    direction="horizontal",
    style={"flex": "1", "min-height": "0"},
    children=[
        PanelGroup(
            direction="vertical",
            children=[
                Panel(
                    id={"type": "panel", "index": "A"},
                    children=make_graph_block("A"),
                ),
                PanelResizeHandle(
                    html.Div(className="resize-handle-vertical"),
                ),
                Panel(
                    children=[
                        PanelGroup(
                            direction="horizontal",
                            children=[
                                Panel(
                                    id={"type": "panel", "index": "B"},
                                    children=make_graph_block("B"),
                                ),
                                PanelResizeHandle(
                                    html.Div(className="resize-handle-horizontal"),
                                ),
                                Panel(
                                    id={"type": "panel", "index": "C"},
                                    children=make_graph_block("C"),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

plotting_layout = html.Div(
    style={
        "height": "100%",
        "width": "100%",
        "display": "flex",
        "flex-direction": "column",
        "overflow": "hidden",
    },
    children=[
        dmc.Button(
            "Select samples",
            id="select-samples-button",
            leftSection=html.I(className="bi bi-plus-circle-fill"),
            style={"flex-shrink": "0"},
        ),
        dcc.Store(
            id="panels-store",
            data=DEFAULT_PLOTS,
        ),
        dcc.Store(
            id="selected-samples",
            data=[],
        ),
        panels_container,
        samples_modal,
    ],
)

# ---------------------------- PLOTTING CALLBACKS ----------------------------- #


def register_plotting_callbacks(app: Dash) -> None:
    """Register all callbacks for plotting."""
    # Register individual graph callbacks
    register_ts_callbacks(app)
    register_cycles_callbacks(app)
    register_eis_callbacks(app)

    # Batch list has updated, update dropdowns
    @app.callback(
        Output("plotting-batch-select-dropdown", "data"),
        Input("batches-store", "data"),
        prevent_initial_call=True,
    )
    def update_batches_dropdown(batches: dict[str, dict]) -> list[dict]:
        return [{"label": b, "value": b} for b in batches]

    # When the user clicks the "Select samples to plot" button, open the modal
    @app.callback(
        Output("plotting-samples-select-modal", "opened", allow_duplicate=True),
        Input("plotting-samples-select-yes-close", "n_clicks"),
        Input("select-samples-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_samples_modal(_select_clicks: int, _yes: int) -> bool:
        if not ctx.triggered:
            return False
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return button_id == "select-samples-button"

    # When the user hits yes close, change the selected samples
    @app.callback(
        Output("selected-samples", "data"),
        Input("plotting-samples-select-yes-close", "n_clicks"),
        State("plotting-samples-select-dropdown", "value"),
        State("plotting-batch-select-dropdown", "value"),
        State("batches-store", "data"),
        running=[(Output("loading-message-store", "data"), "Loading data...", "")],
        prevent_initial_call=True,
    )
    def load_selected_samples(
        _n_clicks: int,
        samples: list,
        batches: list,
        batch_defs: dict[str, dict],
    ) -> list[str]:
        """Load the selected samples into the data store."""
        if not ctx.triggered:
            raise PreventUpdate
        # Add the samples from batches to samples
        sample_set = set(samples)
        for batch in batches:
            sample_set.update(batch_defs.get(batch, {}).get("samples", []))
        return sorted(sample_set)

    # Update graphs
    @app.callback(
        Output({"type": "graph-container", "index": MATCH}, "children"),
        Input({"type": "graph-type-selector", "index": MATCH}, "value"),
    )
    def update_graph(graph_type: str) -> html.Div:
        instance_id = ctx.outputs_list["id"]["index"]
        return graph_factory(graph_type, instance_id)
