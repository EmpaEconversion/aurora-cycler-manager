"""Test database_setup.py aurora-setup command line tool."""

import gc
import json
import os
import shutil
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_setup import connect_to_config, create_database, create_new_setup, get_status

# Double check you're not going to delete the prod database!
if os.getenv("PYTEST_RUNNING") != "1":
    msg = "This test should not run outside of pytest environment!"
    raise RuntimeError(msg)


@pytest.fixture
def setup_test_projects() -> Generator[tuple[Path, Path], None, None]:
    """Set up and teardown temporary folders for testing setup."""
    original_config_path = Path(__file__).resolve().parent / "test_data" / "test_config.json"
    backup_config_path = original_config_path.with_suffix(".bak")
    shutil.copy(original_config_path, backup_config_path)

    test_project_path_1 = Path(__file__).resolve().parent / "test_data" / "temp_project1"
    test_project_path_2 = Path(__file__).resolve().parent / "test_data" / "temp_project2"

    # Yield the paths for the test projects and config files
    yield test_project_path_1, test_project_path_2

    # Teardown happens at the end of the test, return nothing
    if backup_config_path.exists() and original_config_path.exists():
        original_config_path.unlink()
        backup_config_path.rename(original_config_path)
        get_config(reload=True)
    gc.collect()
    shutil.rmtree(test_project_path_1, ignore_errors=True)
    shutil.rmtree(test_project_path_2, ignore_errors=True)


class TestAnalysis:
    """Test the database_setup.py aurora-setup command line tool."""

    def test_project_init(self, setup_test_projects: tuple[Path, Path]) -> None:
        """Test connect command."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)

        test_project_path_1, test_project_path_2 = setup_test_projects
        shared_config_1 = test_project_path_1 / "database" / "shared_config.json"
        generated_files = [
            "database/shared_config.json",
            "database/database.db",
            "database",
            "protocols",
            "snapshots",
        ]

        # Check that all the files are made
        create_new_setup(test_project_path_1)
        for file in generated_files:
            assert (test_project_path_1 / file).exists(), f"File {file} was not created in {test_project_path_1}"

        # Not allowed to create a new setup in the same directory
        with pytest.raises(FileExistsError):
            create_new_setup(test_project_path_1)

        # Unless you force it
        with shared_config_1.open("w", encoding="utf-8") as f:
            json.dump({"This": "should not be in the next file"}, f)

        create_new_setup(test_project_path_1, overwrite=True)

        with shared_config_1.open(encoding="utf-8") as f:
            data = json.load(f)

        config = get_config(reload=True)
        assert "This" not in data
        assert config["Shared config path"] == shared_config_1

    def test_init_new_project(self, setup_test_projects: tuple[Path, Path]) -> None:
        """Test creating a new project and switching between projects."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)

        test_project_path_1, test_project_path_2 = setup_test_projects
        shared_config_1 = test_project_path_1 / "database" / "shared_config.json"
        shared_config_2 = test_project_path_2 / "database" / "shared_config.json"

        # Make a setup in one directory
        create_new_setup(test_project_path_1)

        # Make a new setup in a different directory
        create_new_setup(test_project_path_2)

        config = get_config(reload=True)
        assert config["Shared config path"] == shared_config_2

        # Switch back to the first project
        connect_to_config(test_project_path_1)
        config = get_config(reload=True)
        assert config["Shared config path"] == shared_config_1

        # Check the status
        status = get_status()
        assert Path(status["Shared config path"]) == shared_config_1

    def test_database_funcs(self, setup_test_projects: tuple[Path, Path]) -> None:
        """Test database functions."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)
        test_project_path_1, test_project_path_2 = setup_test_projects
        shared_config_1 = test_project_path_1 / "database" / "shared_config.json"

        # Initialise the setup
        create_new_setup(test_project_path_1)

        # First check we're pointing to the test database
        config = get_config(reload=True)
        assert config["Database path"] == test_project_path_1 / "database" / "database.db"

        # Update the config to remove all the columns
        with shared_config_1.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["Sample database"] = [
            {"Name": "Sample ID", "Alternative names": ["sampleid"], "Type": "VARCHAR(255) PRIMARY KEY"},
            {"Name": "Delete everything else", "Alternative names": [":)"], "Type": "VARCHAR(255)"},
        ]
        with shared_config_1.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        # This should fail without force
        get_config(reload=True)
        with pytest.raises(ValueError):
            create_database()

        # With force this should remove all the columns
        get_config(reload=True)
        create_database(force=True)
        with sqlite3.connect(test_project_path_1 / "database" / "database.db") as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(samples)")
            columns = cursor.fetchall()
            assert len(columns) == 2, "Columns were not deleted successfully"
