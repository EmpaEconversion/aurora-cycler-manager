"""Unit tests for visualiser."""

from aurora_cycler_manager.visualiser.db_view import enable_buttons

# def test_app(app) -> None:
#     """Check app can be created without errors."""
#     assert app


def test_enable_buttons() -> None:
    """Test expected buttons are enabled on row selection."""
    assert enable_buttons([], "samples") == {"upload-button"}
    assert enable_buttons([], "pipelines") == {"upload-button"}
    assert enable_buttons([], "jobs") == {"upload-button"}
    assert enable_buttons([], "results") == {"upload-button"}

    assert enable_buttons([{"Sample ID": "samp1"}], "samples") == {
        "label-button",
        "view-button",
        "delete-button",
        "info-button",
        "create-batch-button",
        "upload-button",
        "download-button",
        "copy-button",
    }
    assert enable_buttons([{"Sample ID": "samp1"}, {"Sample ID": "samp2"}], "samples") == {
        "label-button",
        "view-button",
        "delete-button",
        "create-batch-button",
        "upload-button",
        "download-button",
        "copy-button",
    }
