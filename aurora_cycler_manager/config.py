"""Copyright © 2025-2026, Empa.

Functions for getting the configuration settings.
"""

import json
import logging
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import platformdirs
from tzlocal import get_localzone_name

from aurora_cycler_manager.stdlib_utils import check_illegal_text

logger = logging.getLogger(__name__)
CONFIG = None


def _assert_required_keys(config: dict) -> None:
    """Check if the config has required info."""
    if config["Database type"] not in ["sqlite", "postgresql"]:
        msg = f"Unknown database type {config['Database type']}. Supported: 'sqlite' and 'postgresql'"
        raise ValueError(msg)
    if config["Database type"] == "sqlite" and "Database path" not in config:
        msg = "sqlite requires a 'Database path' in the config"
        raise ValueError(msg)
    if config["Database type"] == "postgresql" and "Database host" not in config:
        msg = "postgresql requires at least 'Database host', 'Database name', 'Database user' in the config"
        raise ValueError(msg)
    if not config.get("Data folder path"):
        msg = "Config missing 'Data folder path'"
        raise ValueError(msg)
    if not config.get("Protocols folder path"):
        msg = "Config missing 'Protocols folder path"
        raise ValueError(msg)


def _fixup_config(config: dict) -> dict:
    """Add any missing config info, defaults, fix types."""
    # sqlite by default
    if "Database type" not in config:
        config["Database type"] = "sqlite"

    # Servers should be transformed to key: dict with valid labels
    config["Servers"] = _convert_legacy_servers(config)

    # Set timezone
    if config.get("Time zone"):
        config["tz"] = ZoneInfo(config["Time zone"])
    else:
        config["tz"] = ZoneInfo(get_localzone_name())

    # Add a raw snapshots folder path to USER config if it doesn't exist
    if not config.get("Snapshots folder path"):
        config["Snapshots folder path"] = platformdirs.user_data_dir("aurora_cycler_manager", appauthor=False)
        user_config_path = config["User config path"]
        with user_config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            logger.warning(
                "IMPORTANT: Added default 'Snapshots folder path' to config file at %s. ",
                user_config_path,
            )
            logger.warning("IMPORTANT: Snapshots can add up to many gigabytes if you have 100s of long experiments.")

    # Use "Data folder path" - rename legacy "Processed snapshots folder path"
    if not config.get("Data folder path"):
        config["Data folder path"] = config.get("Processed snapshots folder path")

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

    # If there is no user config file, create one
    if not user_config_path.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
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
            logger.critical(
                "Created new config file at %s.",
                user_config_path,
            )

    with user_config_path.open(encoding="utf-8") as f:
        try:
            config = json.load(f)
            config["User config path"] = user_config_path
        except json.JSONDecodeError as e:
            msg = f"Error reading config file {user_config_path}: {e}"
            raise ValueError(msg) from e

    # Check for USER CONFIG relative paths and convert to absolute paths
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

        # Check for SHARED CONFIG relative paths and convert to absolute paths
        shared_config_dir = shared_config_path.parent
        for key in shared_config:
            if "path" in key.lower():
                if not Path(shared_config[key]).is_absolute():
                    shared_config[key] = Path(shared_config_dir / shared_config[key])
                else:
                    shared_config[key] = Path(shared_config[key])
        config.update(shared_config)

    # Fill in any missing details in the config
    config = _fixup_config(config)

    # Check that the config is complete
    if shared_config_path:
        _assert_required_keys(config)
    else:
        try:
            _assert_required_keys(config)
        except ValueError as e:
            msg = (
                "Not connected to any Aurora project."
                'Use `aurora-setup init "path/to/my/project` to create a new project, '
                'or `aurora-setup connect "path/to/my/project"` to connect to an existing project.'
            )
            raise ValueError(msg) from e

    return config


def _convert_legacy_servers(config: dict) -> dict:
    """Convert servers from older config styles to single dict."""
    servers = _convert_servers_to_dict(config.get("Servers", {}))

    # Also convert old harvester lists to new server dict
    neware_harvesters = _convert_servers_to_dict(config.get("Neware harvester", {}).get("Servers", {}))
    for server_config in neware_harvesters.values():
        server_config["server_type"] = "neware_harvester"
    biologic_harvesters = _convert_servers_to_dict(config.get("EC-lab harvester", {}).get("Servers", {}))
    for server_config in biologic_harvesters.values():
        server_config["server_type"] = "biologic_harvester"

    # Merge, new server style takes priority over old harvester style
    servers = {**neware_harvesters, **biologic_harvesters, **servers}

    for server_label in servers:
        check_illegal_text(server_label)

    # Drop example-server if it still exists
    if servers.get("example-label") and servers["example-label"].get("hostname") == "example-hostname":
        servers.pop("example-label")
    return servers


def _convert_servers_to_dict(servers: list | dict) -> dict:
    """Convert list of servers to dict."""
    if isinstance(servers, list):
        return {s["label"]: s for s in servers}
    for server_label, server_config in servers.items():
        server_config["label"] = server_label
    return servers


def get_config(*, reload: bool = False) -> dict:
    """Return global configuration dictionary.

    Only reads the config file once, unless reload is set to True.

    """
    global CONFIG
    if CONFIG is None or reload:
        CONFIG = _read_config_file()
    return CONFIG
