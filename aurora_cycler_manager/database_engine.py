"""Copyright © 2025-2026, Empa.

Functions for interacting with the database.
"""

import os
from pathlib import Path

from sqlalchemy import (
    Engine,
    create_engine,
)

from aurora_cycler_manager.config import get_config

CONFIG = get_config()


def get_engine(config: dict) -> Engine:
    """Create sqlite3 or postgres db engine."""
    db_type = config.get("Database type", "sqlite")
    if db_type == "sqlite":
        return create_engine(f"sqlite:///{Path(config['Database path']).as_posix()}")
    if db_type == "postgresql":
        host = config["Database host"]
        port = config.get("Database port", 5432)
        name = config["Database name"]
        user = config.get("Database user") or os.environ.get("AURORA_DB_USER")
        password = config.get("Database password") or os.environ.get("AURORA_DB_PASSWORD")
        if password:
            connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
        else:  # Will use with .pgpass
            connection_string = f"postgresql+psycopg2://{user}@{host}:{port}/{name}"
        return create_engine(connection_string)
    msg = f"Unsupported database type: {db_type}"
    raise ValueError(msg)
