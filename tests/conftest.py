"""Before any tests are run, set envionment variable PYTEST_RUNNING."""

import os

import pytest


@pytest.fixture(autouse=True)
def set_pytest_env() -> None:
    """Set envinonment variable for pytest."""
    os.environ["PYTEST_RUNNING"] = "1"
