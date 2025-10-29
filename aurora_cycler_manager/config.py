"""Copyright © 2025, Empa.

Functions for getting the configuration settings.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)
CONFIG = None


def _read_config_file() -> dict:
    """Get the configuration data from the user and shared config files.

    Returns:
        dict: dictionary containing the configuration data

    """
    current_dir = Path(__file__).resolve().parent
    custom_config_path = os.getenv("AURORA_USER_CONFIG")
    # Check if the environment is set for pytest
    if os.getenv("PYTEST_RUNNING") == "1":
        config_dir = current_dir.parent / "tests" / "test_data"
        user_config_path = config_dir / "test_config.json"
    # Check if using a custom configuration, e.g. for testing
    elif custom_config_path:
        logger.warning("Using custom config file %s", custom_config_path)
        user_config_path = Path(custom_config_path)
        config_dir = user_config_path.parent
        if not user_config_path.exists():
            msg = f"User config file {user_config_path} does not exist."
            raise FileNotFoundError(msg)
    else:
        config_dir = Path(platformdirs.user_data_dir("aurora_cycler_manager", appauthor=False))
        user_config_path = config_dir / "config.json"
        # Legacy - might be in the current directory, move to user data directory
        if not user_config_path.exists():
            old_user_config_path = current_dir / "config.json"
            if old_user_config_path.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                old_user_config_path.rename(user_config_path)
                user_config_path = config_dir / "config.json"
                logger.warning("Moved config file from %s to %s", old_user_config_path, user_config_path)

    err_msg = f"""
        Please fill in the config file at {user_config_path}.

        REQUIRED:
        'Shared config path': Path to the shared config file on the network drive.

        OPTIONAL - if you want to interact directly with cyclers (e.g. load, eject, submit jobs):
        'SSH private key path': Path to the SSH private key file if not in standard location (e.g. '~/.ssh/id_rsa').
        'SSH known hosts path': Path to the SSH known hosts file if not in standard location ('~/.ssh/known_hosts')
        'Snapshots folder path': Path to a (local) folder to store unprocessed snapshots e.g. 'C:/aurora-shapshots'.

        You can set the 'Shared config path' by running 'aurora-setup connect --project-dir=<path>'.
    """

    # if there is no user config file, create one
    if not user_config_path.exists():
        with user_config_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "Shared config path": "",
                        "Snapshots folder path": platformdirs.user_data_dir("aurora_cycler_manager", appauthor=False),
                        "SSH private key path": "",
                        "SSH known hosts path": "",
                    },
                    indent=4,
                ),
            )
            logger.info(
                "Created new config file at %s.",
                user_config_path,
            )
            raise FileNotFoundError(err_msg)

    with user_config_path.open(encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            msg = f"Error reading config file {user_config_path}: {e}"
            raise ValueError(msg) from e

    if not config.get("Snapshots folder path"):
        config["Snapshots folder path"] = platformdirs.user_data_dir("aurora_cycler_manager", appauthor=False)
        with user_config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            logger.warning(
                "IMPORTANT: Added default 'Snapshots folder path' to config file at %s. ",
                user_config_path,
            )
            logger.warning("IMPORTANT: Snapshots can add up to many gigabytes if you have 100s of long experiments.")

    # Check for relative paths and convert to absolute paths
    for key in config:
        if "path" in key.lower() and config[key]:
            if not Path(config[key]).is_absolute():
                config[key] = Path(config_dir / config[key])
            else:
                config[key] = Path(config[key])

    # If there is a shared config file, update with settings from that file
    shared_config_path = config.get("Shared config path")
    if shared_config_path:
        with Path(shared_config_path).open(encoding="utf-8") as f:
            shared_config = json.load(f)

        # Check for relative paths and convert to absolute paths
        shared_config_dir = shared_config_path.parent
        for key in shared_config:
            if "path" in key.lower():
                if not Path(shared_config[key]).is_absolute():
                    shared_config[key] = Path(shared_config_dir / shared_config[key])
                else:
                    shared_config[key] = Path(shared_config[key])
        config.update(shared_config)

    if not config.get("Database path"):
        raise ValueError(err_msg)

    # Patch the database in case users are using an old version
    if config.get("Database path").exists():
        patch_database(config)

    config["User config path"] = user_config_path

    # For SSH connections, paths must be str | None, does not accept Path
    if config.get("SSH private key path"):
        config["SSH private key path"] = str(config["SSH private key path"])
    else:
        config["SSH private key path"] = None
    if config.get("SSH known hosts path"):
        config["SSH known hosts path"] = str(config["SSH known hosts path"])
    else:
        config["SSH known hosts path"] = Path("~/.ssh/known_hosts").expanduser()

    return config


def patch_database(config: dict) -> None:
    """Add missing columns to database, in case users are coming from an older version."""
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        # Make columns in the jobs table if it doesn't exist
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [col[1] for col in cursor.fetchall()]
        if "Capacity (mAh)" not in columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN `Capacity (mAh)` FLOAT")
        if "Unicycler protocol" not in columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN `Unicycler protocol` TEXT")
        conn.commit()


def get_config(*, reload: bool = False) -> dict:
    """Return global configuration dictionary.

    Only reads the config file once, unless reload is set to True.

    """
    global CONFIG
    if CONFIG is None or reload:
        CONFIG = _read_config_file()
    return CONFIG
