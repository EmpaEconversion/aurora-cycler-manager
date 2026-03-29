"""Functions for determining colors and styles of plots."""

import logging

import plotly.express as px
from dash import Dash, Input, Output, State, dcc
from dash import callback_context as ctx
from plotly.colors import hex_to_rgb, label_rgb, sample_colorscale

logger = logging.getLogger(__name__)

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

# Define line styles
line_styles = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def to_rgba(color_str: str, alpha: float = 0.2) -> str:
    """Convert color in rgb, rgba, hex, or named to rgba with transparency."""
    if color_str.startswith("rgba"):
        return color_str
    if color_str.startswith("rgb"):
        rgb_vals = color_str[color_str.find("(") + 1 : color_str.find(")")].split(",")
        return f"rgba({int(rgb_vals[0])}, {int(rgb_vals[1])}, {int(rgb_vals[2])}, {alpha})"
    if color_str.startswith("#"):
        r, g, b = hex_to_rgb(color_str)
        return f"rgba({r}, {g}, {b}, {alpha})"
    r, g, b = hex_to_rgb(label_rgb(color_str))
    return f"rgba({r}, {g}, {b}, {alpha})"


def get_trace_colors(
    selected_samples: dict[str, dict],
    color_by: str,
    colormap: str,
    discrete_colormap: str,
) -> dict[str, str]:
    """Get trace colors."""
    none_color = "rgb(150, 150, 150)"

    sample_color: dict[str, str] = {}  # mapping sample ID to color

    color_value_set = {v for sample_data in selected_samples.values() if (v := sample_data.get(color_by)) is not None}

    # Try to figure out which coloring mode to use
    color_mode = "none"
    if len(color_value_set) == 0:
        color_mode = "none"
    elif len(color_value_set) == 1:
        color_mode = "single_value"
    elif not all(isinstance(v, (int, float)) or v is None for v in color_value_set):
        color_mode = "categorical"
    elif len(color_value_set) < 5:
        color_mode = "numerical_categorical"
    else:
        color_mode = "numerical"

    if color_mode == "none":
        colormap_list: list[str] = discrete_color_dict.get(discrete_colormap, discrete_color_dict["Plotly"])
        sample_color = dict.fromkeys(selected_samples, colormap_list[0])

    elif color_mode == "categorical":
        colormap_list: list[str] = discrete_color_dict.get(discrete_colormap, discrete_color_dict["Plotly"])
        val_to_col = {v: colormap_list[i % len(colormap_list)] for i, v in enumerate(color_value_set)}
        sample_color = {k: val_to_col.get(v.get(color_by), none_color) for k, v in selected_samples.items()}

    elif color_mode == "numerical_categorical":
        colormap_list: list[str] = discrete_color_dict.get(discrete_colormap, discrete_color_dict["Plotly"])
        val_to_col = {v: colormap_list[i] for i, v in enumerate(sorted(color_value_set))}
        sample_color = {k: val_to_col.get(v.get(color_by), none_color) for k, v in selected_samples.items()}

    elif color_mode == "numerical":
        cmin = min(color_value_set)
        cmax = max(color_value_set)
        colormap_list: list[str] = cont_color_dict.get(colormap, cont_color_dict["Viridis"])
        sample_color = {
            k: sample_colorscale(colormap_list, [(v.get(color_by) - cmin) / (cmax - cmin)])[0]
            if v.get(color_by)
            else none_color
            for k, v in selected_samples.items()
        }

    elif color_mode == "single_value":
        colormap_list = discrete_color_dict.get(discrete_colormap, discrete_color_dict["Plotly"])
        sample_color = {k: colormap_list[0] if v.get(color_by) else none_color for k, v in selected_samples.items()}

    return sample_color


color_stores = [
    dcc.Store(id="sample:color", data={}),
    dcc.Store(id="sample:style", data={}),
]


def register_color_callbacks(app: Dash) -> None:
    """Register callbacks for coloring plots."""

    @app.callback(
        Output("sample:color", "data", allow_duplicate=True),
        Output("redraw-trigger", "data"),
        State("selected-samples", "data"),
        Input("color-by", "value"),
        prevent_initial_call=True,
    )
    def update_colors(selected_samples: dict[str, dict], color_by: str) -> tuple[dict[str, str], str]:
        """Update color list."""
        return get_trace_colors(selected_samples, color_by, "Viridis", "Plotly"), ctx.triggered_id
