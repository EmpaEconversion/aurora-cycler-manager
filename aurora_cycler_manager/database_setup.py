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
import re
from pathlib import Path

import platformdirs
from sqlalchemy import (
    Column,
    Index,
    MetaData,
    PrimaryKeyConstraint,
    Table,
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
    # Strings
    "TEXT": types.Text,
    "VARCHAR": types.String,
    "CHAR": types.String,
    # Integers
    "INTEGER": types.Integer,
    "INT": types.Integer,
    "BIGINT": types.BigInteger,
    "SMALLINT": types.SmallInteger,
    # Floats / Decimals
    "FLOAT": types.Float,
    "REAL": types.Float,
    "DOUBLE PRECISION": types.Float,
    "NUMERIC": types.Numeric,
    "DECIMAL": types.Numeric,
    # Boolean
    "BOOLEAN": types.Boolean,
    "BOOL": types.Boolean,
    # Date / Time
    "DATE": types.Date,
    "TIME": types.Time,
    "DATETIME": types.DateTime,
    "TIMESTAMP": types.DateTime,
    # Other
    "JSON": types.JSON,
}


def get_sa_type(type_str: str) -> types.TypeEngine:
    """Convert types in config to sqlalchemy types."""
    type_upper = type_str.strip().upper()

    if type_upper in TYPE_MAP:
        t = TYPE_MAP[type_upper]
        # Return instance if it's a class, not already instantiated
        return t() if isinstance(t, type) else t

    # Parameterised types
    if m := re.match(r"VARCHAR\((\d+)\)", type_upper):
        return types.String(int(m.group(1)))
    if m := re.match(r"(NUMERIC|DECIMAL)\((\d+),\s*(\d+)\)", type_upper):
        return types.Numeric(int(m.group(2)), int(m.group(3)))

    valid = list(TYPE_MAP.keys())
    msg = (
        f"SQL type '{type_upper}' not known. Valid types: {valid}, "
        "or VARCHAR(x), NUMERIC(y,z), DECIMAL(y,z), where x=length, y=precision, and z=scale."
    )
    raise ValueError(msg)


def default_config(base_dir: Path) -> dict:
    """Create default shared config file."""
    return {
        "Database type": "sqlite",
        "Database path": str(base_dir / "aurora.db"),
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
            {"Name": "Cell number", "Alternative names": ["Battery_Number"], "Type": "INT"},
            {"Name": "Rack position", "Alternative names": ["Rack_Position"], "Type": "INT"},
            {"Name": "N:P ratio", "Alternative names": ["Actual N:P Ratio"], "Type": "FLOAT"},
            {"Name": "N:P ratio overlap factor", "Type": "FLOAT"},
            {"Name": "Anode rack position", "Alternative names": ["Anode Position"], "Type": "INT"},
            {"Name": "Anode type", "Type": "TEXT"},
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
            {"Name": "Cathode type", "Type": "TEXT"},
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
            {"Name": "Separator type", "Alternative names": ["Separator"], "Type": "TEXT"},
            {"Name": "Separator diameter (mm)", "Type": "FLOAT"},
            {"Name": "Separator thickness (mm)", "Type": "FLOAT"},
            {"Name": "Electrolyte name", "Alternative names": ["Electrolyte"], "Type": "TEXT"},
            {"Name": "Electrolyte description", "Type": "TEXT"},
            {"Name": "Electrolyte position", "Type": "INT"},
            {"Name": "Electrolyte amount (uL)", "Alternative names": ["Electrolyte Amount"], "Type": "FLOAT"},
            {"Name": "Electrolyte dispense order", "Type": "TEXT"},
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
            {"Name": "Casing type", "Type": "TEXT"},
            {"Name": "Casing material", "Type": "TEXT"},
            {"Name": "Top spacer type", "Type": "TEXT"},
            {"Name": "Top spacer thickness (mm)", "Alternative names": ["Spacer (mm)"], "Type": "FLOAT"},
            {"Name": "Top spacer diameter (mm)", "Alternative names": [], "Type": "FLOAT"},
            {"Name": "Top spacer material", "Alternative names": [], "Type": "TEXT"},
            {"Name": "Bottom spacer type", "Type": "TEXT"},
            {"Name": "Bottom spacer thickness (mm)", "Alternative names": [], "Type": "FLOAT"},
            {"Name": "Bottom spacer diameter (mm)", "Alternative names": [], "Type": "FLOAT"},
            {"Name": "Bottom spacer material", "Alternative names": [], "Type": "TEXT"},
            {"Name": "Comment", "Alternative names": ["Comments"], "Type": "TEXT"},
            {"Name": "Barcode", "Type": "TEXT"},
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

    engine = get_engine(config)
    meta = MetaData()

    samples_required_cols = [
        Column("Sample ID", types.Text, primary_key=True),
        Column("Run ID", types.Text),
        Column("Label", types.Text),
    ]
    columns = config["Sample database"]
    sample_columns = [
        Column(col["Name"], get_sa_type(col["Type"]))
        for col in columns
        if col["Name"] not in {c.name for c in samples_required_cols} | {"sync_modified", "sync_op"}
    ]

    samples_table = Table(  # noqa: F841
        "samples",
        meta,
        *samples_required_cols,
        *sample_columns,
        Column("sync_modified", types.Float),
        Column("sync_op", types.Text),
    )

    jobs_table = Table(
        "jobs",
        meta,
        Column("Job ID", types.Text, primary_key=True),
        Column("Sample ID", types.Text),
        Column("Pipeline", types.Text),
        Column("Status", types.Text),
        Column("Jobname", types.Text),
        Column("Server label", types.Text),
        Column("Server hostname", types.Text),
        Column("Job ID on server", types.Text),
        Column("Submitted", types.DateTime),
        Column("Payload", types.Text),
        Column("Unicycler protocol", types.Text),
        Column("Capacity (mAh)", types.Float),
        Column("Comment", types.Text),
        Column("Last checked", types.DateTime),
        Column("Snapshot status", types.Text),
        Column("Last snapshot", types.DateTime),
        Column("sync_modified", types.Float),
        Column("sync_op", types.Text),
    )

    pipelines_table = Table(
        "pipelines",
        meta,
        Column("Pipeline", types.Text, primary_key=True),
        Column("Sample ID", types.Text),
        Column("Job ID", types.Text),
        Column("Ready", types.Boolean),
        Column("Flag", types.Text),
        Column("Last checked", types.DateTime),
        Column("Server label", types.Text),
        Column("Server type", types.Text),
        Column("Server hostname", types.Text),
        Column("Job ID on server", types.Text),
        Column("sync_modified", types.Float),
        Column("sync_op", types.Text),
    )

    results_table = Table(  # noqa: F841
        "results",
        meta,
        Column("Sample ID", types.Text, primary_key=True),
        Column("Pipeline", types.Text),
        Column("Status", types.Text),
        Column("Flag", types.Text),
        Column("Number of cycles", types.Integer),
        Column("Capacity loss (%)", types.Float),
        Column("First formation efficiency (%)", types.Float),
        Column("Initial specific discharge capacity (mAh/g)", types.Float),
        Column("Initial efficiency (%)", types.Float),
        Column("Last specific discharge capacity (mAh/g)", types.Float),
        Column("Last efficiency (%)", types.Float),
        Column("Max voltage (V)", types.Float),
        Column("Formation C", types.Float),
        Column("Cycling C", types.Float),
        Column("Last snapshot", types.DateTime),
        Column("Last analysis", types.DateTime),
        Column("Snapshot status", types.String(3)),
        Column("Snapshot pipeline", types.String(50)),
        Column("sync_modified", types.Float),
        Column("sync_op", types.Text),
    )

    dataframes_table = Table(  # noqa: F841
        "dataframes",
        meta,
        Column("Sample ID", types.Text, nullable=False),
        Column("File stem", types.Text, nullable=False),
        Column("Job ID", types.Text),
        Column("From known source", types.Boolean),
        Column("Data start", types.DateTime),
        Column("Data end", types.DateTime),
        Column("Modified", types.DateTime),
        PrimaryKeyConstraint("Sample ID", "File stem"),
    )

    harvester_table = Table(  # noqa: F841
        "harvester",
        meta,
        Column("id", types.Integer, primary_key=True, autoincrement=True),
        Column("Server label", types.Text),
        Column("Server hostname", types.Text),
        Column("Folder", types.Text),
        Column("Last snapshot", types.DateTime),
        UniqueConstraint("Server label", "Server hostname", "Folder"),
    )

    batches_table = Table(  # noqa: F841
        "batches",
        meta,
        Column("id", types.Integer, primary_key=True, autoincrement=True),
        Column("label", types.Text, unique=True, nullable=False),
        Column("description", types.Text),
    )

    batch_samples_table = Table(  # noqa: F841
        "batch_samples",
        meta,
        Column("batch_id", types.Integer),
        Column("sample_id", types.Text),
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
        existing_columns = (
            {col["name"] for col in inspector.get_columns("samples")}
            - {c.name for c in samples_required_cols}
            - {"sync_modified", "sync_op"}
        )
        new_columns = {col.name for col in sample_columns}
        added = new_columns - existing_columns
        removed = existing_columns - new_columns

        with engine.begin() as conn:
            if removed:
                if not force:
                    msg = f"Operation would remove columns: {', '.join(removed)}. Use '--force' to proceed."
                    raise ValueError(msg)
                for col in removed:
                    conn.execute(text(f'ALTER TABLE samples DROP COLUMN "{col}"'))
                logger.warning("Columns %s removed", ", ".join(removed))

            if added:
                for col in sample_columns:
                    if col.name in added:
                        type_str = col.type.compile(dialect=engine.dialect)
                        conn.execute(text(f'ALTER TABLE samples ADD COLUMN "{col.name}" {type_str}'))
                logger.info("Adding new columns: %s", ", ".join(added))

            if not added and not removed:
                logger.info("No changes to database configuration")


def create_new_setup(base_dir: str | Path, overwrite: bool = False) -> None:
    """Create a new aurora setup with a shared config file and database."""
    base_dir = Path(base_dir).resolve()
    shared_config_path = base_dir / "shared_config.json"
    if shared_config_path.exists():
        if overwrite:
            logger.warning("Overwriting existing project config file at %s", shared_config_path)
        else:
            msg = "A project shared config file already exists at this location. Use --overwrite to overwrite it."
            raise FileExistsError(msg)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "data").mkdir(exist_ok=True)
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
