"""Copyright © 2026, Empa.

Functions and callbacks for the 'info' functionality in Database tab.
"""

import json
import logging

import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, callback, html
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate

import aurora_cycler_manager.database_funcs as dbf

logger = logging.getLogger(__name__)

info_modal = dmc.Modal(
    title="Info",
    id="info-modal",
    centered=True,
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
                    dmc.AccordionPanel(dmc.CodeHighlight(json.dumps(sample_data, indent=4), language="json")),
                ],
            ),
            dmc.AccordionItem(
                value="results",
                children=[
                    dmc.AccordionControl("Results summary"),
                    dmc.AccordionPanel(
                        dmc.CodeHighlight(json.dumps(results, indent=4), language="json")
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
                    dmc.AccordionPanel(dmc.CodeHighlight(json.dumps(pipeline_data, indent=4), language="json")),
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
                    dmc.AccordionPanel(dmc.CodeHighlight(json.dumps(job_data, indent=4), language="json")),
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
        Input("info-button", "n_clicks"),
        State("table-select", "value"),
        State("selected-rows-store", "data"),
        prevent_initial_call=True,
    )
    def open_info_modal(_n_clicks: int, table: str, selected_rows: list) -> tuple:
        if table in {"samples", "results"}:
            return True, {"Sample ID": selected_rows[0]["Sample ID"]}
        if table == "jobs":
            return True, {"Job ID": selected_rows[0]["Job ID"]}
        if table == "pipelines":
            return True, {"Pipeline": selected_rows[0]["Pipeline"]}
        logger.warning("Table %s not understood", table)
        raise PreventUpdate

    # When a link is clicked, go to that info page
    @app.callback(
        Output("info-store", "data", allow_duplicate=True),
        Input({"type": "info-nav-link", "entity_type": ALL, "id": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_nav_link(n_clicks_list: list[int]) -> dict[str, str]:
        if not ctx.triggered_id or not any(n for n in n_clicks_list if n):
            raise PreventUpdate
        return {ctx.triggered_id["entity_type"]: ctx.triggered_id["id"]}

    # When the targeted info page changes, query db and update the actual modal children
    @app.callback(
        Output("info-modal", "title"),
        Output("info-modal", "children"),
        Input("info-store", "data"),
    )
    def render_info(data: dict[str, str]) -> tuple:
        try:
            if sample_id := data.get("Sample ID"):
                return f"Sample: {sample_id}", generate_sample_info(sample_id)
            if pipeline_id := data.get("Pipeline"):
                return f"Pipeline: {pipeline_id}", generate_pipeline_info(pipeline_id)
            if job_id := data.get("Job ID"):
                return f"Job: {job_id}", generate_job_info(job_id)
            if batch_id := data.get("Batch name"):
                return f"Batch: {batch_id}", generate_batch_info(batch_id)
        except Exception as e:
            return "An error occurred", str(e)
        else:
            return "An error occured", "Row missing ID or not understood"

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
