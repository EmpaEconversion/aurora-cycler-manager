"""Copyright © 2025, Empa.

Command line utility for setting up the Aurora Cycler Manager.

Connect to an existing configuration:
    aurora-setup connect --config=<path>

Create a new setup with a shared config file and database:
    aurora-setup init --base-dir=<path> [--overwrite]

Update the existing database from the config:
    aurora-setup update [--force]

Get the status of the setup:
    aurora-setup status [--verbose]
"""

import argparse
import contextlib
import json
import logging
import os
import sqlite3
from pathlib import Path

import platformdirs

from aurora_cycler_manager.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if the environment is set for pytest
root_dir = Path(__file__).resolve().parent
if os.getenv("PYTEST_RUNNING") == "1":
    root_dir = root_dir.parent / "tests" / "test_data"
    USER_CONFIG_PATH = root_dir / "test_config.json"
else:
    user_config_dir = Path(platformdirs.user_data_dir("aurora_cycler_manager", appauthor=False))
    USER_CONFIG_PATH = user_config_dir / "config.json"


def default_config(base_dir: Path) -> dict:
    """Create default shared config file."""
    return {
        "Database path": str(base_dir / "database" / "database.db"),
        "Database backup folder path": str(base_dir / "database" / "backup"),
        "Samples folder path": str(base_dir / "samples"),
        "Protocols folder path": str(base_dir / "protocols"),
        "Processed snapshots folder path": str(base_dir / "snapshots"),
        "Servers": [
            {
                "label": "example-server",
                "hostname": "example-hostname",
                "username": "username on remote server",
                "server_type": "tomato (only supported type at the moment)",
                "shell_type": "powershell or cmd - changes some commands",
                "command_prefix": "this is put before any command, e.g. conda activate tomato ; ",
                "command_suffix": "",
                "tomato_scripts_path": "tomato-specific: this is put before ketchup in the command",
                "tomato_data_path": "tomato-specific: the folder where data is stored, usually AppData/local/dgbowl/tomato/version/jobs",
            },
        ],
        "EC-lab harvester": {
            "Servers": [
                {
                    "label": "example-server",
                    "hostname": "example-hostname",
                    "username": "username on remote server",
                    "shell_type": "powershell or cmd",
                    "EC-lab folder location": "C:/where/data/is/saved",
                },
            ],
            "run_id_lookup": {
                "folder name on server": "run_id in database",
            },
        },
        "Neware harvester": {
            "Servers": [
                {
                    "label": "example-server",
                    "hostname": "example-hostname",
                    "username": "username on remote server",
                    "shell_type": "cmd",
                    "Neware folder location": "C:/where/data/is/saved/",
                },
            ],
        },
        "User mapping": {
            "short_name": "full_name",
        },
        "Sample database": [
            {"Name": "Sample ID", "Alternative names": ["sampleid"], "Type": "VARCHAR(255) PRIMARY KEY"},
            {"Name": "Run ID", "Type": "VARCHAR(255)"},
            {"Name": "Cell number", "Alternative names": ["Battery_Number"], "Type": "INT"},
            {"Name": "Rack position", "Alternative names": ["Rack_Position"], "Type": "INT"},
            {"Name": "N:P ratio", "Alternative names": ["Actual N:P Ratio"], "Type": "FLOAT"},
            {"Name": "N:P ratio overlap factor", "Type": "FLOAT"},
            {"Name": "Anode rack position", "Alternative names": ["Anode Position"], "Type": "INT"},
            {"Name": "Anode type", "Type": "VARCHAR(255)"},
            {"Name": "Anode description", "Type": "TEXT"},
            {"Name": "Anode diameter (mm)", "Alternative names": ["Anode_Diameter", "Anode Diameter"], "Type": "FLOAT"},
            {"Name": "Anode mass (mg)", "Alternative names": ["Anode Weight (mg)", "Anode Weight"], "Type": "FLOAT"},
            {
                "Name": "Anode current collector mass (mg)",
                "Alternative names": ["Anode Current Collector Weight (mg)"],
                "Type": "FLOAT",
            },
            {
                "Name": "Anode active material mass fraction",
                "Alternative names": ["Anode active material weight fraction", "Anode AM Content"],
                "Type": "FLOAT",
            },
            {
                "Name": "Anode active material mass (mg)",
                "Alternative names": ["Anode Active Material Weight (mg)", "Anode AM Weight (mg)"],
                "Type": "FLOAT",
            },
            {"Name": "Anode C-rate definition areal capacity (mAh/cm2)", "Type": "FLOAT"},
            {"Name": "Anode C-rate definition specific capacity (mAh/g)", "Type": "FLOAT"},
            {
                "Name": "Anode balancing specific capacity (mAh/g)",
                "Alternative names": ["Anode Practical Capacity (mAh/g)", "Anode Nominal Specific Capacity (mAh/g)"],
                "Type": "FLOAT",
            },
            {"Name": "Anode balancing capacity (mAh)", "Alternative names": ["Anode Capacity (mAh)"], "Type": "FLOAT"},
            {"Name": "Cathode rack position", "Alternative names": ["Cathode Position"], "Type": "INT"},
            {"Name": "Cathode type", "Type": "VARCHAR(255)"},
            {"Name": "Cathode description", "Type": "TEXT"},
            {
                "Name": "Cathode diameter (mm)",
                "Alternative names": ["Cathode_Diameter", "Cathode Diameter"],
                "Type": "FLOAT",
            },
            {"Name": "Cathode mass (mg)", "Alternative names": ["Cathode Weight (mg)"], "Type": "FLOAT"},
            {
                "Name": "Cathode current collector mass (mg)",
                "Alternative names": ["Cathode Current Collector Weight (mg)"],
                "Type": "FLOAT",
            },
            {
                "Name": "Cathode active material mass fraction",
                "Alternative names": ["Cathode Active Material Weight Fraction", "Cathode AM Content"],
                "Type": "FLOAT",
            },
            {
                "Name": "Cathode active material mass (mg)",
                "Alternative names": ["Cathode Active Material Weight (mg)", "Cathode AM Weight (mg)"],
                "Type": "FLOAT",
            },
            {"Name": "Cathode C-rate definition areal capacity (mAh/cm2)", "Type": "FLOAT"},
            {"Name": "Cathode C-rate definition specific capacity (mAh/g)", "Type": "FLOAT"},
            {
                "Name": "Cathode balancing specific capacity (mAh/g)",
                "Alternative names": [
                    "Cathode Practical Capacity (mAh/g)",
                    "Cathode Nominal Specific Capacity (mAh/g)",
                ],
                "Type": "FLOAT",
            },
            {
                "Name": "Cathode balancing capacity (mAh)",
                "Alternative names": ["Cathode Capacity (mAh)"],
                "Type": "FLOAT",
            },
            {"Name": "Separator type", "Alternative names": ["Separator"], "Type": "VARCHAR(255)"},
            {"Name": "Separator diameter (mm)", "Type": "FLOAT"},
            {"Name": "Separator thickness (mm)", "Type": "FLOAT"},
            {"Name": "Electrolyte name", "Alternative names": ["Electrolyte"], "Type": "VARCHAR(255)"},
            {"Name": "Electrolyte description", "Type": "TEXT"},
            {"Name": "Electrolyte position", "Type": "INT"},
            {"Name": "Electrolyte amount (uL)", "Alternative names": ["Electrolyte Amount"], "Type": "FLOAT"},
            {"Name": "Electrolyte dispense order", "Type": "VARCHAR(255)"},
            {
                "Name": "Electrolyte amount before separator (uL)",
                "Alternative names": ["Electrolyte Amount Before Seperator (uL)"],
                "Type": "FLOAT",
            },
            {
                "Name": "Electrolyte amount after separator (uL)",
                "Alternative names": ["Electrolyte Amount After Seperator (uL)"],
                "Type": "FLOAT",
            },
            {
                "Name": "C-rate definition capacity (mAh)",
                "Alternative names": ["Capacity (mAh)", "C-rate Capacity (mAh)"],
                "Type": "FLOAT",
            },
            {"Name": "Casing type", "Type": "VARCHAR(255)"},
            {"Name": "Casing material", "Type": "VARCHAR(255)"},
            {"Name": "Top spacer type", "Type": "VARCHAR(255)"},
            {"Name": "Top spacer thickness (mm)", "Alternative names": ["Spacer (mm)"], "Type": "FLOAT"},
            {"Name": "Top spacer diameter (mm)", "Alternative names": [], "Type": "FLOAT"},
            {"Name": "Top spacer material", "Alternative names": [], "Type": "VARCHAR(255)"},
            {"Name": "Bottom spacer type", "Type": "VARCHAR(255)"},
            {"Name": "Bottom spacer thickness (mm)", "Alternative names": [], "Type": "FLOAT"},
            {"Name": "Bottom spacer diameter (mm)", "Alternative names": [], "Type": "FLOAT"},
            {"Name": "Bottom spacer material", "Alternative names": [], "Type": "VARCHAR(255)"},
            {"Name": "Label", "Type": "VARCHAR(255)"},
            {"Name": "Comment", "Alternative names": ["Comments"], "Type": "TEXT"},
            {"Name": "Barcode", "Type": "VARCHAR(255)"},
            {"Name": "Assembly history", "Type": "TEXT"},
        ],
    }


def create_database(force: bool = False) -> None:
    """Create/update a database file."""
    # Load the configuration
    config = get_config()
    database_path = Path(config["Database path"])

    # Check if database file already exists
    if database_path.exists() and database_path.suffix == ".db":
        db_existed = True
        logger.info("Found database at %s", database_path)
    else:
        db_existed = False
        database_path.parent.mkdir(exist_ok=True)
        logger.info("Creating new database at %s", database_path)

    # Get the list of columns from the configuration
    columns = config["Sample database"]
    column_definitions = [f"`{col['Name']}` {col['Type']}" for col in columns]

    # Connect to database, create tables
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS samples ({', '.join(column_definitions)})")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "`Job ID` VARCHAR(255) PRIMARY KEY, "
            "`Sample ID` VARCHAR(255), "
            "`Pipeline` VARCHAR(50), "
            "`Status` VARCHAR(3), "
            "`Jobname` VARCHAR(50), "
            "`Server label` VARCHAR(255), "
            "`Server hostname` VARCHAR(255), "
            "`Job ID on server` VARCHAR(255), "
            "`Submitted` DATETIME, "
            "`Payload` TEXT, "
            "`Comment` TEXT, "
            "`Last checked` DATETIME, "
            "`Snapshot status` VARCHAR(3), "
            "`Last snapshot` DATETIME, "
            "FOREIGN KEY(`Sample ID`) REFERENCES samples(`Sample ID`),"
            "FOREIGN KEY(`Pipeline`) REFERENCES pipelines(`Pipeline`)"
            ")",
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS pipelines ("
            "`Pipeline` VARCHAR(50) PRIMARY KEY, "
            "`Sample ID` VARCHAR(255),"
            "`Job ID` VARCHAR(255), "
            "`Ready` BOOLEAN, "
            "`Flag` VARCHAR(10), "
            "`Last checked` DATETIME, "
            "`Server label` VARCHAR(255), "
            "`Server type` VARCHAR(50), "
            "`Server hostname` VARCHAR(255), "
            "`Job ID on server` VARCHAR(255), "
            "FOREIGN KEY(`Sample ID`) REFERENCES samples(`Sample ID`), "
            "FOREIGN KEY(`Job ID`) REFERENCES jobs(`Job ID`)"
            ")",
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "`Sample ID` VARCHAR(255) PRIMARY KEY,"
            "`Pipeline` VARCHAR(50),"
            "`Status` VARCHAR(3),"
            "`Flag` VARCHAR(10),"
            "`Number of cycles` INT,"
            "`Capacity loss (%)` FLOAT,"
            "`First formation efficiency (%)` FLOAT,"
            "`Initial specific discharge capacity (mAh/g)` FLOAT,"
            "`Initial efficiency (%)` FLOAT,"
            "`Last specific discharge capacity (mAh/g)` FLOAT,"
            "`Last efficiency (%)` FLOAT,"
            "`Max voltage (V)` FLOAT,"
            "`Formation C` FLOAT,"
            "`Cycling C` FLOAT,"
            "`Last snapshot` DATETIME,"
            "`Last analysis` DATETIME,"
            "`Last plotted` DATETIME,"
            "`Snapshot status` VARCHAR(3),"
            "`Snapshot pipeline` VARCHAR(50),"
            "FOREIGN KEY(`Sample ID`) REFERENCES samples(`Sample ID`), "
            "FOREIGN KEY(`Pipeline`) REFERENCES pipelines(`Pipeline`)"
            ")",
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS harvester ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "`Server label` TEXT, "
            "`Server hostname` TEXT, "
            "`Folder` TEXT, "
            "UNIQUE(`Server label`, `Server hostname`, `Folder`)"
            ")",
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS batches ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "label TEXT UNIQUE NOT NULL, "
            "description TEXT"
            ")",
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS batch_samples ("
            "batch_id INT, "
            "sample_id TEXT, "
            "FOREIGN KEY(batch_id) REFERENCES batches(id), "
            "FOREIGN KEY(sample_id) REFERENCES samples(`Sample ID`), "
            "UNIQUE(batch_id, sample_id)"
            ")",
        )
        conn.commit()

        # Check if there are new columns to add in samples table
        if db_existed:
            cursor.execute("PRAGMA table_info(samples)")
            existing_columns = cursor.fetchall()
            existing_columns = [col[1] for col in existing_columns]
            new_columns = [col["Name"] for col in config["Sample database"]]
            added_columns = [col for col in new_columns if col not in existing_columns]
            removed_columns = [col for col in existing_columns if col not in new_columns]
            if removed_columns:
                # Ask user to double confirm
                if not force:
                    msg = (
                        "WARNING: Operation would remove columns.\n"
                        "Use '--force' to proceed.\n"
                        f"Would remove columns: {', '.join(removed_columns)}"
                    )
                    raise ValueError(msg)
                for col in removed_columns:
                    cursor.execute(f'ALTER TABLE samples DROP COLUMN "{col}"')
                conn.commit()
                logger.warning("Columns %s removed", ", ".join(removed_columns))
            if added_columns:
                # Add new columns
                for col in config["Sample database"]:
                    if col["Name"] in added_columns:
                        cursor.execute(f'ALTER TABLE samples ADD COLUMN "{col["Name"]}" {col["Type"]}')
                conn.commit()
                logger.info("Adding new columns to database: %s", ", ".join(added_columns))
            if not added_columns and not removed_columns:
                logger.info("No changes to database configuration")


def create_new_setup(base_dir: str | Path, overwrite: bool = False) -> None:
    """Create a new aurora setup with a shared config file and database."""
    base_dir = Path(base_dir).resolve()
    shared_config_path = base_dir / "database" / "shared_config.json"
    if shared_config_path.exists():
        if overwrite:
            logger.warning("Overwriting existing shared config file at %s", shared_config_path)
        else:
            msg = "Shared config file already exists. Use --overwrite to overwrite it."
            raise FileExistsError(msg)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "database").mkdir(exist_ok=True)
    (base_dir / "snapshots").mkdir(exist_ok=True)
    (base_dir / "protocols").mkdir(exist_ok=True)

    logger.info("Created folder structure at %s", base_dir)

    with (shared_config_path).open("w") as f:
        json.dump(default_config(base_dir), f, indent=4)

    # Read the user_config file, if it didn't exist before, get_config will create it
    with contextlib.suppress(Exception):
        get_config(reload=True)
    with (USER_CONFIG_PATH).open("r") as f:
        user_config = json.load(f)

    # Add the shared config path to the user config file
    user_config["Shared config path"] = str(shared_config_path)
    with (USER_CONFIG_PATH).open("w") as f:
        json.dump(user_config, f, indent=4)

    # Reload the configuration with the new path
    get_config(reload=True)

    create_database(force=False)

    logger.critical(
        "YOU MUST FILL IN THE DETAILS AT %s",
        shared_config_path,
    )


def connect_to_config(shared_config_folder: str | Path) -> None:
    """Connect to an existing configuration."""
    shared_config_path = Path(shared_config_folder).resolve()
    # Try to find the shared config file in a few different locations
    confirmed_shared_config_path = None

    # Maybe they provided a full path to the shared config file
    if shared_config_path.suffix == ".json" and shared_config_path.exists():
        confirmed_shared_config_path = shared_config_path

    # Maybe they provided a parent folder or parent parent folder
    if not confirmed_shared_config_path and shared_config_path.is_dir():
        potential_paths = [
            shared_config_path / "database" / "shared_config.json",
            shared_config_path / "shared_config.json",
        ]
        for path in potential_paths:
            if path.exists():
                confirmed_shared_config_path = path
                break

    # If not, give up searching
    if not confirmed_shared_config_path:
        msg = "Could not find a valid shared config file. Check that shared_config.json exists in the provided folder."
        raise FileNotFoundError(msg)

    logger.info("Using shared config file at %s", str(confirmed_shared_config_path))

    # Check that the shared config has the required keys
    required_keys = [
        "Database path",
        "Database backup folder path",
        "Samples folder path",
        "Protocols folder path",
        "Processed snapshots folder path",
    ]
    with confirmed_shared_config_path.open("r") as f:
        shared_config = json.load(f)
    for key in required_keys:
        if key not in shared_config:
            msg = f"Shared config file at {confirmed_shared_config_path} is missing required key: {key}"
            raise ValueError(msg)

    # get_config will generate a default file if it doesn't exist
    with contextlib.suppress(Exception):
        get_config(reload=True)
    # Update the user config file with the shared config path
    logger.info("Updating user config file at %s", str(USER_CONFIG_PATH))
    with (USER_CONFIG_PATH).open("r") as f:
        user_config = json.load(f)
    user_config["Shared config path"] = str(confirmed_shared_config_path)
    with (USER_CONFIG_PATH).open("w") as f:
        json.dump(user_config, f, indent=4)

    # If this runs successfully, the user can now run the app
    get_config(reload=True)
    logger.info("You can now start the app with aurora-app")


def get_status(verbose: bool = False) -> dict:
    """Print the status of the aurora cycler manager setup."""
    if not USER_CONFIG_PATH.exists():
        logger.error("User config file does not exist at %s", USER_CONFIG_PATH)
        raise FileNotFoundError

    with USER_CONFIG_PATH.open("r") as f:
        user_config = json.load(f)

    shared_config_path = user_config.get("Shared config path")
    if not shared_config_path or not Path(shared_config_path).exists():
        logger.error(
            "Shared config path is not set or does not exist. "
            "Use 'aurora-setup connect' to connect to a config, "
            "or 'aurora-setup init' to create a new one."
        )
        raise FileNotFoundError
    logger.info("User config file: %s", USER_CONFIG_PATH)
    logger.info("Shared config file: %s", shared_config_path)

    config = get_config()
    if verbose:
        logger.info("Current configuration:")
        config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        logger.info(json.dumps(config, indent=4))
    return config


def main() -> None:
    """CLI entry point for aurora cycler manager setup utility."""
    parser = argparse.ArgumentParser(description="aurora-cycler-manager setup utility.")
    subparsers = parser.add_subparsers(dest="command")

    connect_parser = subparsers.add_parser("connect", help="Connect to existing config")
    connect_parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Path to Aurora project directory containing configuration, database, data folders",
    )

    create_parser = subparsers.add_parser("init", help="Create new config and database")
    create_parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Path to Aurora project directory - subfolders, configuration files and a database will be placed here",
    )
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing config and database")

    update_parser = subparsers.add_parser("update", help="Update the database from the config")
    update_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow permanent deletion of database columns if config removes columns",
    )

    status_parser = subparsers.add_parser("status", help="Get the status of the setup")
    status_parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    if args.command == "connect":
        connect_to_config(args.project_dir)
    elif args.command == "init":
        create_new_setup(args.project_dir, args.overwrite)
    elif args.command == "update":
        create_database(force=args.force)
    elif args.command == "status":
        get_status(verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
