"""Before any tests are run, set envionment variable PYTEST_RUNNING."""

import json
import logging
import os
import shutil
import warnings
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from .mocks import MockSSHClient

logger = logging.getLogger(__name__)


def pytest_configure(config: pytest.Config) -> None:
    """Set PYTEST_RUNNING env variable early, before any tests are collected or run."""
    os.environ["PYTEST_RUNNING"] = "1"
    logger.info("PYTEST_RUNNING set to 1 in pytest_configure")


@pytest.fixture
def test_dir() -> Path:
    """Get test dir."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def ignore_sqlite_warns() -> None:
    """When using --cov, sqlite3 context managers don't close, coverage holds some ref and warns."""
    warnings.filterwarnings("ignore", message="unclosed database", category=ResourceWarning)


@pytest.fixture
def reset_all(test_dir: Path) -> Generator[None, None, None]:
    """Reset samples folders and database to original state."""
    db_path = test_dir / "database" / "test_database.db"
    snapshots_path = test_dir / "snapshots"
    batches_path = test_dir / "batches"

    # Make backup of database
    shutil.copyfile(db_path, db_path.with_suffix(".bak"))
    test_files = [
        "*.h5",
        "*.parquet",
        "cycles.*.json",
        "aux.*.jsonld",
        "battinfo.*.jsonld",
        "metadata.*.json",
        "overall.*.json",
        "batch.*.json",
        "batch.*.xlsx",
    ]
    for test_file in test_files:
        assert not any(snapshots_path.rglob(test_file)), f"Already {test_file} in snapshots folder!"
        assert not any(batches_path.rglob(test_file)), f"Already {test_file} in batches folder!"

    yield

    # Restore database
    shutil.copyfile(db_path.with_suffix(".bak"), db_path)
    # Remove sample files

    for test_file in test_files:
        for file in snapshots_path.rglob(test_file):
            file.unlink()
        for file in batches_path.rglob(test_file):
            file.unlink()
    # Reset config
    with (test_dir / "test_config.json").open("w") as f:
        f.write(
            json.dumps(
                {
                    "Shared config path": "shared_config.json",
                    "SSH private key path": "fake/private/key",
                    "Snapshots folder path": "local_snapshots",
                },
                indent=4,
            )
        )


@pytest.fixture
def mock_ssh() -> Generator[None, None, None]:
    """Mock SSH client."""
    with patch("aurora_cycler_manager.ssh.paramiko.SSHClient", return_value=MockSSHClient()):
        yield
