"""Before any tests are run, set envionment variable PYTEST_RUNNING."""

import os


def pytest_configure(config) -> None:  # noqa: ANN001, ARG001
    """Set PYTEST_RUNNING env variable early, before any tests are collected or run."""
    os.environ["PYTEST_RUNNING"] = "1"
    print("PYTEST_RUNNING set to 1 in pytest_configure")