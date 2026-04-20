"""Copyright © 2026, Empa.

Functions and callbacks for the 'info' functionality in Database tab.
"""

import json
import logging
from datetime import datetime

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, callback, dcc, html
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate

import aurora_cycler_manager.database_funcs as dbf

logger = logging.getLogger(__name__)

info_modal = dmc.Modal(
    title=dmc.Group(
        [
            dmc.ActionIcon(
                html.I(className="bi bi-arrow-left"),
                id="info-back-button",
                variant="subtle",
                disabled=True,
            ),
            dmc.ActionIcon(
                html.I(className="bi bi-arrow-right"),
                id="info-forward-button",
                variant="subtle",
                disabled=True,
            ),
            html.Span(id="info-modal-title-text"),
        ],
        gap="xs",
        align="center",
    ),
    id="info-modal",
    children=dcc.Loading(
        children=html.Div(id="info-modal-content"),
        type="circle",
        color="var(--mantine-color-blue-6)",
        delay_show=200,
    ),
    size="xl",
)


def _nav_link(label: str, entity_type: str, entity_id: str) -> html.Span:
    """Generate a clickable link to a sample/job/pipeline/batch."""
    return html.Span(
        dmc.Badge(
            label,
            variant="light",
            color="blue",
            radius="sm",
            style={"cursor": "pointer", "textTransform": "none"},
        ),
        id={"type": "info-nav-link", "entity_type": entity_type, "id": entity_id},
        n_clicks=0,
    )


def _strip_datetimes(data: dict) -> dict:
    """Datetime -> ISO8601 string so dict can be JSON dumped."""
    return {k: v.isoformat() if isinstance(v, datetime) else v for k, v in data.items()}


def generate_sample_info(sample_id: str) -> dmc.Accordion:
    """Create info modal content from a Sample ID."""
    sample_data = dbf.get_sample_data(sample_id)
    results = dbf.get_results_from_sample(sample_id)
    jobs = dbf.get_jobs_from_sample(sample_id)
    pipeline = dbf.get_pipeline_from_sample(sample_id)
    batches = dbf.get_batches_from_sample(sample_id)

    if not jobs:
        job_element = [dmc.Text("No associated jobs")]
    elif len(jobs) == 1:
        job_element = [dmc.Text("Associated jobs:"), _nav_link(jobs[0], "Job ID", jobs[0])]
    else:
        job_element = [dmc.Text("Associated jobs:"), *[_nav_link(j, "Job ID", j) for j in jobs]]

    pipeline_id = pipeline["Pipeline"] if pipeline else None
    pipeline_element = (
        [dmc.Text("Loaded on pipeline:"), _nav_link(pipeline_id, "Pipeline", pipeline_id)]
        if pipeline_id
        else [dmc.Text("Not loaded on any pipeline")]
    )

    if not batches:
        batch_element = [dmc.Text("Not part of any batches")]
    elif len(batches) == 1:
        batch_element = [dmc.Text("Part of batch:"), _nav_link(batches[0], "Batch name", batches[0])]
    else:
        batch_element = [dmc.Text("Part of batches:"), *[_nav_link(b, "Batch name", b) for b in batches]]

    return dmc.Accordion(
        value="links",
        variant="separated",
        children=[
            dmc.AccordionItem(
                value="links",
                children=[
                    dmc.AccordionControl("Links"),
                    dmc.AccordionPanel(
                        dmc.Stack(
                            [
                                *batch_element,
                                *pipeline_element,
                                *job_element,
                            ]
                        )
                    ),
                ],
            ),
            dmc.AccordionItem(
                value="sample_data",
                children=[
                    dmc.AccordionControl("Sample data"),
                    dmc.AccordionPanel(
                        dmc.CodeHighlight(json.dumps(_strip_datetimes(sample_data), indent=4), language="json")
                    ),
                ],
            ),
            dmc.AccordionItem(
                value="results",
                children=[
                    dmc.AccordionControl("Results summary"),
                    dmc.AccordionPanel(
                        dmc.CodeHighlight(json.dumps(_strip_datetimes(results), indent=4), language="json")
                        if results
                        else dmc.Text("No results.")
                    ),
                ],
            ),
        ],
    )


def generate_pipeline_info(pipeline_id: str) -> dmc.Accordion:
    """Create info modal content from a Pipeline."""
    pipeline_data = dbf.get_pipeline(pipeline_id)
    sample_id = dbf.get_sample_from_pipeline(pipeline_id)
    sample_element = (
        [dmc.Text("Has sample loaded:"), _nav_link(sample_id, "Sample ID", sample_id)]
        if sample_id
        else [dmc.Text("Has no sample loaded")]
    )
    return dmc.Accordion(
        value="links",
        variant="separated",
        multiple=True,
        children=[
            dmc.AccordionItem(
                value="links",
                children=[
                    dmc.AccordionControl("Links"),
                    dmc.AccordionPanel(
                        dmc.Stack(
                            sample_element,
                        )
                    ),
                ],
            ),
            dmc.AccordionItem(
                value="pipeline_data",
                children=[
                    dmc.AccordionControl("Pipeline data"),
                    dmc.AccordionPanel(
                        dmc.CodeHighlight(json.dumps(_strip_datetimes(pipeline_data), indent=4), language="json")
                    ),
                ],
            ),
        ],
    )


def generate_job_info(job_id: str) -> dmc.Accordion:
    """Generate info modal content from a Job ID."""
    job_data = dbf.get_job_data(job_id)
    sample_id = job_data["Sample ID"]
    pipeline = job_data.get("Pipeline")

    sample_element = (
        [dmc.Text("Run on sample:"), _nav_link(sample_id, "Sample ID", sample_id)]
        if sample_id
        else [dmc.Text("Sample unknown.")]
    )
    pipeline_element = (
        [dmc.Text("Run on pipeline:"), _nav_link(pipeline, "Pipeline", pipeline)]
        if pipeline
        else [dmc.Text("Pipeline unknown.")]
    )

    return dmc.Accordion(
        value="links",
        variant="separated",
        multiple=True,
        children=[
            dmc.AccordionItem(
                value="links",
                children=[
                    dmc.AccordionControl("Links"),
                    dmc.AccordionPanel(
                        dmc.Stack(
                            [
                                *sample_element,
                                *pipeline_element,
                            ]
                        )
                    ),
                ],
            ),
            dmc.AccordionItem(
                value="job_data",
                children=[
                    dmc.AccordionControl("Job metadata"),
                    dmc.AccordionPanel(
                        dmc.CodeHighlight(json.dumps(_strip_datetimes(job_data), indent=4), language="json")
                    ),
                ],
            ),
        ],
    )


def generate_batch_info(batch_id: str) -> dmc.Stack:
    """Generate info modal content from a Batch ID."""
    batch = dbf.get_one_batch(batch_id)
    return dmc.Stack(
        [
            dmc.Button(
                "Edit on batch tab",
                leftSection=html.I(className="bi bi-pencil"),
                id="batch-info-edit-button",
            ),
            dmc.Text("Description:"),
            dmc.Text(batch["description"]),
            dmc.Divider(),
            dmc.Text("Contains samples:"),
            *[_nav_link(s, "Sample ID", s) for s in batch["samples"]],
        ]
    )


def register_db_info_callbacks(app: Dash) -> None:
    """Register info callbacks."""

    # When info button is pressed, open a modal and display the whole row
    @app.callback(
        Output("info-modal", "opened", allow_duplicate=True),
        Output("info-store", "data", allow_duplicate=True),
        Output("info-history-store", "data", allow_duplicate=True),
        Input("info-button", "n_clicks"),
        State("table-select", "value"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def open_info_modal(_n_clicks: int, table: str, selected_rows: list) -> tuple:
        if table in {"samples", "results"}:
            new_data = {"Sample ID": selected_rows[0]["Sample ID"]}
        elif table == "jobs":
            new_data = {"Job ID": selected_rows[0]["Job ID"]}
        elif table == "pipelines":
            new_data = {"Pipeline": selected_rows[0]["Pipeline"]}
        else:
            logger.warning("Table %s not understood", table)
            raise PreventUpdate
        return True, new_data, {"history": [new_data], "index": 0}

    # When a link is clicked, go to that info page
    @app.callback(
        Output("info-store", "data", allow_duplicate=True),
        Output("info-history-store", "data", allow_duplicate=True),
        Input({"type": "info-nav-link", "entity_type": ALL, "id": ALL}, "n_clicks"),
        State("info-history-store", "data"),
        prevent_initial_call=True,
    )
    def handle_nav_link(n_clicks_list: list[int], history_data: dict) -> tuple:
        if not ctx.triggered_id or not any(n for n in n_clicks_list if n):
            raise PreventUpdate
        new_data = {ctx.triggered_id["entity_type"]: ctx.triggered_id["id"]}
        history = history_data.get("history", [])
        index = history_data.get("index", -1)
        # Remove forward history, and append new
        history = [*history[: index + 1], new_data]
        return new_data, {"history": history, "index": len(history) - 1}

    # Back / forward navigation through history
    @app.callback(
        Output("info-store", "data", allow_duplicate=True),
        Output("info-history-store", "data", allow_duplicate=True),
        Input("info-back-button", "n_clicks"),
        Input("info-forward-button", "n_clicks"),
        State("info-history-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_history(_back: int, _forward: int, history_data: dict) -> tuple:
        if not ctx.triggered_id:
            raise PreventUpdate
        history = history_data.get("history", [])
        index = history_data.get("index", 0)
        new_index = index - 1 if ctx.triggered_id == "info-back-button" else index + 1
        if new_index < 0 or new_index >= len(history):
            raise PreventUpdate
        return history[new_index], {"history": history, "index": new_index}

    # Enable/disable back/forward buttons
    @app.callback(
        Output("info-back-button", "disabled"),
        Output("info-forward-button", "disabled"),
        Input("info-history-store", "data"),
    )
    def update_nav_buttons(history_data: dict) -> tuple:
        if not history_data:
            return True, True
        index = history_data.get("index", 0)
        length = len(history_data.get("history", []))
        return index <= 0, index >= length - 1

    # When the targeted info page changes, query db and update the actual modal children
    @app.callback(
        Output("info-modal-content", "children"),
        Input("info-store", "data"),
    )
    def render_info(data: dict[str, str]) -> dmc.Accordion | dmc.Stack | str:
        try:
            if sample_id := data.get("Sample ID"):
                return generate_sample_info(sample_id)
            if pipeline_id := data.get("Pipeline"):
                return generate_pipeline_info(pipeline_id)
            if job_id := data.get("Job ID"):
                return generate_job_info(job_id)
            if batch_id := data.get("Batch name"):
                return generate_batch_info(batch_id)
        except Exception as e:
            return "An error occurred: " + str(e)
        else:
            return "No information."

    @app.callback(
        Output("info-modal-title-text", "children"),
        Input("info-store", "data"),
    )
    def render_title(data: dict[str, str]) -> str:
        if sample_id := data.get("Sample ID"):
            return f"Sample: {sample_id}"
        if pipeline_id := data.get("Pipeline"):
            return f"Pipeline: {pipeline_id}"
        if job_id := data.get("Job ID"):
            return f"Job: {job_id}"
        if batch_id := data.get("Batch name"):
            return f"Batch: {batch_id}"
        return ""

    # When hitting edit batch from info, switch to batch tab
    @callback(
        Output("table-select", "value"),
        Output("batch-edit-batch", "value", allow_duplicate=True),
        Output("info-modal", "opened", allow_duplicate=True),
        Input("batch-info-edit-button", "n_clicks"),
        State("info-store", "data"),
        prevent_initial_call=True,
    )
    def switch_to_batch(_n_clicks: int, data: dict) -> tuple:
        """Switch to batch tab, select batch."""
        if _n_clicks:
            return "batches", data.get("Batch name"), False
        raise PreventUpdate
