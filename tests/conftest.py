"""Before any tests are run, set envionment variable PYTEST_RUNNING."""

import json
import logging
import os
import shutil
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

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

    # Make backup of database
    shutil.copyfile(db_path, db_path.with_suffix(".bak"))
    assert not any(snapshots_path.rglob("*.h5")), "Already h5 files in snapshots folder!"
    assert not any(snapshots_path.rglob("*.parquet")), "Already parquet files in snapshots folder!"
    assert not any(snapshots_path.rglob("*.json")), "Already json files in snapshots folder!"
    assert not any(snapshots_path.rglob("*.jsonld")), "Already jsonld files in snapshots folder!"
    assert not any(test_dir.glob("temp_*")), "Already temp folders!"

    yield

    # Restore database
    shutil.copyfile(db_path.with_suffix(".bak"), db_path)
    # Remove sample files
    files_to_remove = [
        "*.h5",
        "*.parquet",
        "cycles.*.json",
        "aux.*.jsonld",
        "battinfo.*.jsonld",
        "metadata.*.json",
        "overall.*.json",
    ]
    for rem in files_to_remove:
        for file in snapshots_path.rglob(rem):
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
