"""Copyright © 2025-2026, Empa.

Functions for interacting with the database.
"""

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
        user = config["Database user"]
        password = config["Database password"]
        return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}")
    msg = f"Unsupported database type: {db_type}"
    raise ValueError(msg)
