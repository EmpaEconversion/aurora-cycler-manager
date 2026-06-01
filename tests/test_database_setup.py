"""Test database_setup.py aurora-setup command line tool."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import inspect, text, types
from sqlalchemy.exc import ProgrammingError

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import get_engine, patch_database
from aurora_cycler_manager.database_setup import (
    connect_to_config,
    create_database,
    create_new_setup,
    get_sa_type,
    main,
    print_config,
)

# Double check you're not going to delete the prod database!
if os.getenv("PYTEST_RUNNING") != "1":
    msg = "This test should not run outside of pytest environment!"
    raise RuntimeError(msg)


class TestDatabaseSetup:
    """Test the database_setup.py aurora-setup command line tool."""

    @staticmethod
    def assert_sa_type(result, expected) -> None:
        """Compare sqlalchemy types."""
        assert isinstance(result, type(expected))
        if hasattr(expected, "length"):
            assert result.length == expected.length
        if hasattr(expected, "precision"):
            assert result.precision == expected.precision
        if hasattr(expected, "scale"):
            assert result.scale == expected.scale

    def test_sa_types(self) -> None:
        """Check type mapping works."""
        self.assert_sa_type(get_sa_type("VARCHAR(123)"), types.String(123))
        self.assert_sa_type(get_sa_type("DECIMAL(5,1)"), types.Numeric(5, 1))
        self.assert_sa_type(get_sa_type("NUMERIC(3,2)"), types.Numeric(3, 2))
        self.assert_sa_type(get_sa_type("INT"), types.Integer())
        self.assert_sa_type(get_sa_type("FLOAT"), types.Float())
        with pytest.raises(ValueError, match="Valid types:"):
            get_sa_type("FOO")
        with pytest.raises(ValueError, match="Valid types:"):
            get_sa_type("NUMERIC(5)")

    def test_project_init(self, reset_all, tmp_path: Path) -> None:
        """Test connect command."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)

        test_project_path_1 = tmp_path / "temp_project1"
        shared_config_1 = test_project_path_1 / "shared_config.json"
        generated_files = [
            "shared_config.json",
            "aurora.db",
            "protocols",
            "data",
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
        shared_config_1 = test_project_path_1 / "shared_config.json"
        shared_config_2 = test_project_path_2 / "shared_config.json"

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
        status = print_config()
        assert Path(status["Shared config path"]) == shared_config_1

    def test_database_funcs(self, reset_all, tmp_path: Path) -> None:
        """Test database functions."""
        # Double check you're not going to delete the prod database!
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)
        test_project_path_1 = tmp_path / "temp_project1"
        shared_config_1 = test_project_path_1 / "shared_config.json"

        # Initialise the setup
        create_new_setup(test_project_path_1)

        # First check we're pointing to the test database
        config = get_config(reload=True)
        assert config["Database path"] == test_project_path_1 / "aurora.db"

        # Update the config to remove all the columns
        with shared_config_1.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["Sample database"] = [
            {"Name": "Sample ID", "Alternative names": ["sampleid"], "Type": "TEXT PRIMARY KEY"},
            {"Name": "Delete everything else", "Alternative names": [":)"], "Type": "TEXT"},
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
        # Should be left with 5 required cols + "delete everything else"
        assert len(columns) == 6, "Columns were not deleted successfully"

    def test_db_path(self, reset_all, tmp_path: Path) -> None:
        """Test running aurora-setup update without write permissions."""
        if os.getenv("PYTEST_RUNNING") != "1":
            msg = "This test should not run outside of pytest environment!"
            raise RuntimeError(msg)
        test_project_path_1 = tmp_path / "temp_project1"

        # Initialise the setup
        create_new_setup(test_project_path_1)
        config = get_config(reload=True)
        db_path = config["Database path"]
        assert db_path == test_project_path_1 / "aurora.db"

        # Delete some stuff so that a patch needs to run
        engine = get_engine(config)

        # Sanity check that we are using the test db
        assert "sqlite" in engine.url.drivername, "Safety check: expected SQLite engine"
        assert Path(engine.url.database).resolve() == db_path.resolve(), (
            f"Safety check: engine points to {engine.url.database}, expected {db_path}"
        )
        assert tmp_path.resolve() in Path(engine.url.database).resolve().parents, (
            "Safety check: DB must be inside tmp_path"
        )

        def drop_things() -> None:
            with engine.connect() as conn:
                for table in ["pipelines", "samples", "jobs", "results"]:
                    for col in ["sync_modified", "sync_op"]:
                        conn.execute(text(f'DROP INDEX IF EXISTS "idx_{table}_{col}"'))
                        conn.execute(text(f'ALTER TABLE "{table}" DROP COLUMN "{col}"'))
                conn.execute(text("DROP TABLE dataframes"))

        def assert_missing_things() -> None:
            inspector = inspect(engine)
            for table in ["pipelines", "samples", "jobs", "results"]:
                assert "sync_op" not in [c["name"] for c in inspector.get_columns(table)]
                assert "sync_modified" not in [c["name"] for c in inspector.get_columns(table)]
            tables = inspector.get_table_names()
            assert "dataframes" not in tables

        def assert_things_present() -> None:
            inspector = inspect(engine)
            for table in ["pipelines", "samples", "jobs", "results"]:
                assert "sync_op" in [c["name"] for c in inspector.get_columns(table)]
                assert "sync_modified" in [c["name"] for c in inspector.get_columns(table)]
            tables = inspector.get_table_names()
            assert "dataframes" in tables

        permission_denied = patch(
            "aurora_cycler_manager.database_funcs._update_db_schema",
            side_effect=ProgrammingError(
                statement="ALTER TABLE ...",
                params={},
                orig=Exception("Permission denied."),
            ),
        )

        assert_things_present()
        # patch_database should skip if things are present, so works even without pemissions
        with permission_denied:
            patch_database(engine)

        drop_things()  # Delete some cols/tables

        # patch_database sees there is an issue, errors because of permissions
        with (
            permission_denied,
            pytest.raises(
                PermissionError,
                match=r"Failed to update. An admin must run 'aurora-app' or 'aurora-setup update' first.",
            ),
        ):
            patch_database(engine)

        # Things should still be missing, patch_database (with permissions) adds them back
        assert_missing_things()
        patch_database(engine)
        assert_things_present()

        # Should also work with create_database, which calls patch_database
        drop_things()
        assert_missing_things()
        create_database()
        assert_things_present()

    def test_print_status(self, capsys: pytest.CaptureFixture, reset_all) -> None:
        """Check print status CLI works."""
        with patch("sys.argv", ["aurora-setup", "status"]):
            main()
        captured = capsys.readouterr()
        assert "User config path:" in captured.out
        assert "Shared config path:" in captured.out

    def test_print_status_verbose(self, capsys: pytest.CaptureFixture, reset_all) -> None:
        """Check print status CLI works."""
        with patch("sys.argv", ["aurora-setup", "status", "-v"]):
            main()
        captured = capsys.readouterr()
        res = json.loads(captured.out.strip())
        assert isinstance(res, dict)
        assert "User config path" in res

        with patch("sys.argv", ["aurora-setup", "status", "--verbose"]):
            main()
        captured2 = capsys.readouterr()
        assert captured == captured2
