"""Copyright © 2025, Empa.

Batch editing sub-layout for the database tab.
"""

import logging

import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from aurora_cycler_manager.database_funcs import (
    remove_batch,
    save_or_overwrite_batch,
)

logger = logging.getLogger(__name__)

batch_edit_layout = dmc.Stack(
    id="batch-container",
    children=[
        ### Main layout ###
        dmc.Select(
            id="batch-edit-batch",
            label="Select batch to view",
            data=[],  # Filled in by callback
            searchable=True,
            pt="md",
        ),
        dmc.TextInput(
            id="batch-edit-name",
            label="Batch name",
            placeholder="Enter batch name",
            value="",
        ),
        dmc.Textarea(
            id="batch-edit-description",
            label="Description",
            minRows=4,
            placeholder="Write something about this batch",
            value="",
        ),
        dmc.MultiSelect(
            id="batch-edit-samples",
            label="Samples",
            data=[],  # Filled in by callback
            searchable=True,
            clearable=True,
            placeholder="Select samples",
        ),
        ### Buttons ###
        dmc.Group(
            children=[
                dmc.Button(
                    "Save",
                    leftSection=html.I(className="bi-save"),
                    id="batch-edit-save-button",
                    disabled=True,
                ),
                dmc.Button(
                    "Delete",
                    leftSection=html.I(className="bi-trash3"),
                    id="batch-edit-delete-button",
                    color="red",
                    disabled=True,
                ),
            ],
        ),
        ### Confirmation dialogs ###
        # Add sample confirmation
        dcc.ConfirmDialog(
            id="add-samples-confirm",
            message="This will overwrite samples. Are you sure you want to continue?",
        ),
        # Save batch confirmation
        dcc.ConfirmDialog(
            id="save-batch-confirm",
            message="Save as new batch?",
        ),
        # Overwrite batch confirmation
        dcc.ConfirmDialog(
            id="overwrite-batch-confirm",
            message="Warning: This will overwrite the existing batch. Are you sure?",
        ),
        # Delete batch confirmation
        dcc.ConfirmDialog(
            id="delete-batch-confirm",
            message="This will delete the batch. It will not remove the samples or data. Are you sure?",
        ),
        # A list of samples when creating a batch from other tab
        dcc.Store(id="create-batch-store", data=[]),
    ],
)


### Callbacks ###
def register_batch_edit_callbacks(app: Dash, database_access: bool):
    # When a batch is selected, show the samples in the batch
    @app.callback(
        Output("batch-edit-name", "value"),
        Output("batch-edit-description", "value"),
        Output("batch-edit-samples", "value"),
        Output("batch-edit-batch", "value", allow_duplicate=True),
        Output("create-batch-store", "data", allow_duplicate=True),
        Input("batch-edit-batch", "value"),
        Input("create-batch-store", "data"),
        State("batches-store", "data"),
        prevent_initial_call=True,
    )
    def update_batch_edit_samples(batch: str, samples: list, batch_defs: dict[str, dict]):
        # samples sent from elsewhere
        if samples:
            return "", "", samples, None, {}
        # normal batch selected
        if batch:
            description = str(batch_defs.get(batch, {}).get("description", ""))
            description = description if description else ""
            samples = batch_defs.get(batch, {}).get("samples", [])
            samples = samples if samples else []
            return batch, description, samples, no_update, no_update
        return "", "", [], no_update, no_update

    # When there is a valid batch name and at least one sample, enable the save button
    @app.callback(
        Output("batch-edit-save-button", "disabled"),
        Input("batch-edit-name", "value"),
        Input("batch-edit-samples", "value"),
        prevent_initial_call=True,
    )
    def enable_save_batch(name: str, samples: list[str]):
        return not (database_access and name and samples)

    # When the batch name is the same as an existing batch, enable the delete button
    @app.callback(
        Output("batch-edit-delete-button", "disabled"),
        Input("batch-edit-name", "value"),
        State("batches-store", "data"),
        prevent_initial_call=True,
    )
    def enable_delete_batch(name: str, batch_defs: dict[str, dict]):
        return not (database_access and (name in batch_defs))

    # When save batch is pressed, open a confirm dialog
    @app.callback(
        Output("save-batch-confirm", "displayed"),
        Output("overwrite-batch-confirm", "displayed"),
        Input("batch-edit-save-button", "n_clicks"),
        State("batch-edit-name", "value"),
        State("batches-store", "data"),
        prevent_initial_call=True,
    )
    def save_batch_button(n_clicks, name, batch_def):
        if n_clicks:
            if name in batch_def:
                return no_update, True
            return True, no_update
        raise PreventUpdate

    # When save batch is confirmed, save the batch
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Input("save-batch-confirm", "submit_n_clicks"),
        Input("overwrite-batch-confirm", "submit_n_clicks"),
        State("batch-edit-name", "value"),
        State("batch-edit-description", "value"),
        State("batch-edit-samples", "value"),
        prevent_initial_call=True,
    )
    def save_batch(save_click, overwrite_click, name, description, samples):
        logger.info("Saving batch %s", name)
        save_or_overwrite_batch(name, description, samples, overwrite=True)
        return 1

    # When delete batch is pressed, open confirm dialog
    @app.callback(
        Output("delete-batch-confirm", "displayed"),
        Input("batch-edit-delete-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def delete_batch_button(n_clicks):
        if n_clicks:
            return True
        raise PreventUpdate

    # When delete batch is confirmed, delete the batch
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Output("batch-edit-batch", "value", allow_duplicate=True),
        Input("delete-batch-confirm", "submit_n_clicks"),
        State("batch-edit-name", "value"),
        prevent_initial_call=True,
    )
    def delete_batch(delete_click, name: str):
        logger.info("Deleting batch %s", name)
        remove_batch(name)
        return 1, ""
