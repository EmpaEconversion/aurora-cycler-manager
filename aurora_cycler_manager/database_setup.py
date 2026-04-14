"""Copyright © 2025-2026, Empa.

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
from pathlib import Path

import platformdirs
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    inspect,
    text,
    types,
)

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_engine import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if the environment is set for pytest
root_dir = Path(__file__).resolve().parent
custom_config_path = os.getenv("AURORA_USER_CONFIG")
if os.getenv("PYTEST_RUNNING") == "1":
    root_dir = root_dir.parent / "tests" / "test_data"
    USER_CONFIG_PATH = root_dir / "test_config.json"
elif custom_config_path:
    USER_CONFIG_PATH = Path(custom_config_path)
else:
    user_config_dir = Path(platformdirs.user_data_dir("aurora_cycler_manager", appauthor=False))
    USER_CONFIG_PATH = user_config_dir / "config.json"

TYPE_MAP = {
    "TEXT": Text,
    "VARCHAR(255)": String(255),
    "INTEGER": Integer,
    "FLOAT": Float,
    "BOOLEAN": Boolean,
    "DATETIME": DateTime,
    "TIMESTAMP": DateTime,
}


def get_sa_type(type_str: str) -> types.TypeEngine:
    """Convert SQLITE to sqalchemy types."""
    return TYPE_MAP.get(type_str.upper(), Text)


def default_config(base_dir: Path) -> dict:
    """Create default shared config file."""
    return {
        "Database type": "sqlite",
        "Database path": str(base_dir / "database" / "database.db"),
        "Samples folder path": str(base_dir / "samples"),
        "Protocols folder path": str(base_dir / "protocols"),
        "Data folder path": str(base_dir / "data"),
        "Servers": {
            "example-label": {
                "hostname": "example-hostname",
                "username": "username on remote server",
                "server_type": "neware or biologic or neware_harvester or biologic_harvester",
                "shell_type": "powershell or cmd - changes some commands",
                "command_prefix": "this is put before any command, e.g. conda activate my_env ; ",
                "command_suffix": "this is put after any command",
                "harvester_folders": ["path/to/additional/passive/folders/to/scrape"],
                "data_path": "C:/aurora/data/",
                "neware_raw_data_path": "neware-specific raw ndc path, usually in install folder /BTSServer80/NdcFile/",
            },
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
    """Create/update sqlite3 or postgres database.

    For sqlite3, just a database path is needed in the config.
    For postgres, postgres must already be installed, running, and an empty database must be
    provided, with host, name, user, and password in config.
    """
    config = get_config()
    db_type = config.get("Database type", "sqlite")

    if db_type == "sqlite":
        database_path = Path(config["Database path"])
        db_existed = database_path.exists()
        if not db_existed:
            database_path.parent.mkdir(exist_ok=True)
            logger.info("Creating new database at %s", database_path)
        else:
            logger.info("Found database at %s", database_path)
    else:
        db_existed = True  # assume postgres db already exists, we just create tables

    columns = config["Sample database"]
    sample_columns = [
        Column(col["Name"], get_sa_type(col["Type"]), primary_key=(col["Name"] == "Sample ID")) for col in columns
    ]

    engine = get_engine(config)
    meta = MetaData()

    samples_table = Table(  # noqa: F841
        "samples",
        meta,
        *sample_columns,
        Column("sync_modified", Float),
        Column("sync_op", Text),
    )

    jobs_table = Table(
        "jobs",
        meta,
        Column("Job ID", String(255), primary_key=True),
        Column("Sample ID", String(255)),
        Column("Pipeline", String(50)),
        Column("Status", String(3)),
        Column("Jobname", String(50)),
        Column("Server label", String(255)),
        Column("Server hostname", String(255)),
        Column("Job ID on server", String(255)),
        Column("Submitted", DateTime),
        Column("Payload", Text),
        Column("Unicycler protocol", Text),
        Column("Capacity (mAh)", Float),
        Column("Comment", Text),
        Column("Last checked", DateTime),
        Column("Snapshot status", String(3)),
        Column("Last snapshot", DateTime),
        Column("sync_modified", Float),
        Column("sync_op", Text),
    )

    pipelines_table = Table(
        "pipelines",
        meta,
        Column("Pipeline", String(50), primary_key=True),
        Column("Sample ID", String(255)),
        Column("Job ID", String(255)),
        Column("Ready", Boolean),
        Column("Flag", String(10)),
        Column("Last checked", DateTime),
        Column("Server label", String(255)),
        Column("Server type", String(50)),
        Column("Server hostname", String(255)),
        Column("Job ID on server", String(255)),
        Column("sync_modified", Float),
        Column("sync_op", Text),
    )

    results_table = Table(  # noqa: F841
        "results",
        meta,
        Column("Sample ID", String(255), primary_key=True),
        Column("Pipeline", String(50)),
        Column("Status", String(3)),
        Column("Flag", String(10)),
        Column("Number of cycles", Integer),
        Column("Capacity loss (%)", Float),
        Column("First formation efficiency (%)", Float),
        Column("Initial specific discharge capacity (mAh/g)", Float),
        Column("Initial efficiency (%)", Float),
        Column("Last specific discharge capacity (mAh/g)", Float),
        Column("Last efficiency (%)", Float),
        Column("Max voltage (V)", Float),
        Column("Formation C", Float),
        Column("Cycling C", Float),
        Column("Last snapshot", DateTime),
        Column("Last analysis", DateTime),
        Column("Snapshot status", String(3)),
        Column("Snapshot pipeline", String(50)),
        Column("sync_modified", Float),
        Column("sync_op", Text),
    )

    dataframes_table = Table(  # noqa: F841
        "dataframes",
        meta,
        Column("Sample ID", Text, nullable=False),
        Column("File stem", Text, nullable=False),
        Column("Job ID", Text),
        Column("From known source", Boolean),
        Column("Data start", DateTime),
        Column("Data end", DateTime),
        Column("Modified", DateTime),
        PrimaryKeyConstraint("Sample ID", "File stem"),
    )

    harvester_table = Table(  # noqa: F841
        "harvester",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("Server label", Text),
        Column("Server hostname", Text),
        Column("Folder", Text),
        Column("Last snapshot", DateTime),
        UniqueConstraint("Server label", "Server hostname", "Folder"),
    )

    batches_table = Table(  # noqa: F841
        "batches",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("label", Text, unique=True, nullable=False),
        Column("description", Text),
    )

    batch_samples_table = Table(  # noqa: F841
        "batch_samples",
        meta,
        Column("batch_id", Integer),
        Column("sample_id", Text),
        UniqueConstraint("batch_id", "sample_id"),
    )

    # Indexes
    Index("idx_jobs_job_on_server", jobs_table.c["Job ID on server"], jobs_table.c["Server label"])
    Index("idx_jobs_sample", jobs_table.c["Sample ID"])
    Index("idx_pipelines_sample_id", pipelines_table.c["Sample ID"])
    Index("idx_pipelines_job_id", pipelines_table.c["Job ID"])
    if db_type == "sqlite":
        logger.info("Updating sqlite database at %s...", str(config["Database path"]))
    else:
        logger.info("Update postgresql database '%s'...", config["Database name"])
    meta.create_all(engine, checkfirst=True)
    logger.info("Done. Tables: %s", ", ".join(meta.tables.keys()))

    # Handle added/removed columns in samples
    if db_existed:
        inspector = inspect(engine)
        existing_columns = [col["name"] for col in inspector.get_columns("samples")]
        new_columns = [col["Name"] for col in columns] + ["sync_modified", "sync_op"]
        added = [c for c in new_columns if c not in existing_columns]
        removed = [c for c in existing_columns if c not in new_columns]

        with engine.begin() as conn:
            if removed:
                if not force:
                    msg = f"Operation would remove columns: {', '.join(removed)}. Use '--force' to proceed."
                    raise ValueError(msg)
                for col in removed:
                    conn.execute(text(f'ALTER TABLE samples DROP COLUMN "{col}"'))
                logger.warning("Columns %s removed", ", ".join(removed))

            if added:
                for col in columns:
                    if col["Name"] in added:
                        conn.execute(text(f'ALTER TABLE samples ADD COLUMN "{col["Name"]}" {col["Type"]}'))
                logger.info("Adding new columns: %s", ", ".join(added))

            if not added and not removed:
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
        "Samples folder path",
        "Protocols folder path",
        "Data folder path",
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
            "or 'aurora-setup init' to create a new one.",
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
