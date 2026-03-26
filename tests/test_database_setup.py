"""Test database_setup.py aurora-setup command line tool."""

import json
import os
from pathlib import Path

import pytest
from sqlalchemy import inspect

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import get_engine
from aurora_cycler_manager.database_setup import connect_to_config, create_database, create_new_setup, get_status

# Double check you're not going to delete the prod database!
if os.getenv("PYTEST_RUNNING") != "1":
    msg = "This test should not run outside of pytest environment!"
    raise RuntimeError(msg)


class TestAnalysis:
    """Test the database_setup.py aurora-setup command line tool."""

    def test_project_init(self, reset_all, tmp_path: Path) -> None:
        """Test connect command."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)

        test_project_path_1 = tmp_path / "temp_project1"
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

    def test_init_new_project(self, reset_all, tmp_path: Path) -> None:
        """Test creating a new project and switching between projects."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)

        test_project_path_1 = tmp_path / "temp_project1"
        test_project_path_2 = tmp_path / "temp_project2"
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

    def test_database_funcs(self, reset_all, tmp_path: Path) -> None:
        """Test database functions."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)
        test_project_path_1 = tmp_path / "temp_project1"
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

        # With force this should remove all the columns, sync_modified and sync_op must stay
        get_config(reload=True)
        create_database(force=True)
        engine = get_engine(config)
        inspector = inspect(engine)
        columns = inspector.get_columns("samples")
        assert len(columns) == 4, "Columns were not deleted successfully"
