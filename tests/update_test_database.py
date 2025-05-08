"""Utility to copy parts of the prod database for testing.

Intended to be run as a script to create a subset of the database for pytest.
"""

import os
import sqlite3
import sys
from pathlib import Path

from aurora_cycler_manager.config import get_config

CONFIG = get_config()
DB_PATH = CONFIG["Database path"]
TEST_DB_PATH = Path(__file__).parent / "test_data" / "database" / "test_database.db"
TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def copy_database_schema(source_db: Path, target_db: Path) -> None:
    """Create test database from prod database schema."""
    with sqlite3.connect(source_db) as conn_source, sqlite3.connect(target_db) as conn_target:
        cursor_source = conn_source.cursor()
        cursor_target = conn_target.cursor()

        # Get schema for all objects (tables, indexes, etc.)
        cursor_source.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'index', 'trigger', 'view') AND sql NOT NULL",
        )

        for (sql,) in cursor_source.fetchall():
            if "sqlite_sequence" in sql.lower():
                continue  # This is internal SQLite, skip
            cursor_target.execute(sql)

        conn_target.commit()

    print("Schema copied successfully.")


def copy_rows(source_db: Path, target_db: Path, table: str, id_col: str, row_ids: list[str]) -> None:
    """Copy the rows from one sqlite3 database to another."""
    with sqlite3.connect(source_db) as conn_source, sqlite3.connect(target_db) as conn_target:
        cursor_source = conn_source.cursor()
        cursor_target = conn_target.cursor()

        # Get column names dynamically
        cursor_source.execute(f"PRAGMA table_info({table})")
        columns_info = cursor_source.fetchall()
        column_names = [f"`{col[1]}`" for col in columns_info]  # col[1] is the column name

        # Build query strings
        columns_str = ", ".join(column_names)
        placeholders = ", ".join(["?"] * len(column_names))

        # Fetch the row from source
        for row_id in row_ids:
            cursor_source.execute(
                f"SELECT {columns_str} FROM {table} WHERE `{id_col}` = ?",  # noqa: S608
                (row_id,),
            )
            row = cursor_source.fetchone()
            if row:
                cursor_target.execute(
                    f"INSERT OR REPLACE INTO {table} ({columns_str}) VALUES ({placeholders})",  # noqa: S608
                    row,
                )
                print("Row copied:", row_id)
            else:
                print("Row not found:", row_id)
        conn_target.commit()


if __name__ == "__main__":
    # Check if the script is run directly
    if "PYTEST_RUNNING" in os.environ:
        print("This script is intended to be run outside of pytest.")
        sys.exit(1)

    # Remove old table
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    # Copy schema
    copy_database_schema(DB_PATH, TEST_DB_PATH)

    # Copy data
    samples = [
        "240606_svfe_gen1_15",
        "240606_svfe_gen1_16",
        "240701_svfe_gen6_01",
        "240709_svfe_gen8_01",
    ]
    jobs = [
        "tt1-68",
        "tt1-100",
        "tt1-69",
        "nw4-120-1-1-48",
    ]
    copy_rows(DB_PATH, TEST_DB_PATH, "samples", "Sample ID", samples)
    copy_rows(DB_PATH, TEST_DB_PATH, "jobs", "Job ID", jobs)
