"""Batch edit sub-layout for the database tab."""
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update
from dash_mantine_components import MultiSelect, Select

from aurora_cycler_manager.database_funcs import (
    remove_batch,
    save_or_overwrite_batch,
)

batch_edit_layout = html.Div(
    id="batch-container",
    children = [
        ### Main layout ###
        "Select batch to view:",
        Select(
            id="batch-edit-batch",
            data=[], # Filled in by callback
            searchable=True,
        ),
        html.Div(style={"margin-top": "10px"}),
        "Batch name: ",
        dcc.Input(
            id="batch-edit-name",
            style={"width": "100%"},
            value="",
        ),
        html.Div(style={"margin-top": "10px"}),
        "Batch description:",
        dcc.Textarea(
            id="batch-edit-description",
            style={"width": "100%", "height": "200px"},
            value="",
        ),
        html.Div(style={"margin-top": "10px"}),
        "Samples:",
        MultiSelect(
            id="batch-edit-samples",
            data=[], # Filled in by callback
            searchable=True,
        ),
        html.Div(style={"margin-top": "10px"}),
        ### Buttons ###
        dbc.Button(
            [html.I(className="bi-save me-2"),"Save"],
            id="batch-edit-save-button",
            color="primary",
            className="me-1",
            disabled=True,
        ),
        dbc.Button(
            [html.I(className="bi-trash3 me-2"),"Delete"],
            id="batch-edit-delete-button",
            color="danger",
            className="me-1",
            disabled=True,
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
    ],
)

### Callbacks ###
def register_batch_edit_callbacks(app: Dash, database_access: bool):
    # When a batch is selected, show the samples in the batch
    @app.callback(
        Output("batch-edit-name", "value"),
        Output("batch-edit-description", "value"),
        Output("batch-edit-samples", "value", allow_duplicate=True),
        Input("batch-edit-batch", "value"),
        State("batches-store", "data"),
        prevent_initial_call=True,
    )
    def update_batch_edit_samples(batch: str, batch_defs: dict[str, dict]):
        if not batch:
            return "","",[]
        return (
            batch,
            batch_defs.get(batch, {}).get("description",""),
            batch_defs.get(batch, {}).get("samples",[]),
        )

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
        return no_update, no_update

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
        print(f"Saving batch '{name}'")
        save_or_overwrite_batch(name, description, samples)
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
        return no_update

    # When delete batch is confirmed, delete the batch
    @app.callback(
        Output("refresh-database", "n_clicks", allow_duplicate=True),
        Output("batch-edit-batch", "value"),
        Input("delete-batch-confirm", "submit_n_clicks"),
        State("batch-edit-name", "value"),
        prevent_initial_call=True,
    )
    def delete_batch(delete_click, name: str):
        print(f"Deleting batch '{name}'")
        remove_batch(name)
        return 1, ""
