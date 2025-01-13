"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Script to create the shared config and database files.

Config and database files are created if they do not exist during server-manager
initialisation. The config file is created with some default values for
file paths and server information. The database samples table is created
with columns specified in the config file, with alternative names for handling
different naming conventions in output files.
"""
import json
import sqlite3
import sys
from pathlib import Path

from aurora_cycler_manager.config import get_config

base_dir = Path(__file__).resolve().parents[1]
config_path = base_dir / "config.json"
shared_config_path = base_dir / "shared_config.json"

def default_config(base_dir: Path) -> dict:
    """Create default shared config file."""
    return {
        "Database path": str(base_dir/"database"/"database.db"),
        "Database backup folder path": str(base_dir/"database"/"backup"),
        "Samples folder path": str(base_dir/"samples"),
        "Processed snapshots folder path": str(base_dir/"snapshots"),
        "Batches folder path": str(base_dir/"batches"),
        "Graphs folder path": str(base_dir/"snapshots"),

        "Servers" : [
            {
                "label": "example-server",
                "hostname": "example-hostname",
                "username": "username on remote server",
                "server_type": "tomato (only supported type at the moment)",
                "shell_type": "powershell or cmd - changes some commands",
                "command_prefix" : "this is put before any command, e.g. conda activate tomato ; ",
                "command_suffix" : "",
                "tomato_scripts_path": "tomato-specific: this is put before ketchup in the command",
                "tomato_data_path": "tomato-specific: the folder where data is stored, usually AppData/local/dgbowl/tomato/version/jobs",
            },
        ],

        "EC-lab harvester" : {
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

        "Neware harvester" : {
            "Servers": [
                {
                    "label" : "example-server",
                    "hostname": "example-hostname",
                    "username": "username on remote server",
                    "shell_type": "cmd",
                    "Neware folder location": "C:/where/data/is/saved/",
                },
            ],
        },

        "Sample database" : [
            {"Name" : "Sample ID", "Alternative names" : ["sampleid"], "Type" : "VARCHAR(255) PRIMARY KEY"},
            {"Name" : "Run ID", "Alternative names" : [], "Type" : "VARCHAR(255)"},
            {"Name" : "Cell number", "Alternative names" : ["Cell Number","Battery_Number"], "Type" : "INT"},
            {"Name" : "Actual N:P ratio", "Alternative names" : ["Actual N:P Ratio"], "Type" : "FLOAT"},
            {"Name" : "Rack position", "Alternative names" : ["Rack Position","Rack_Position"], "Type" : "INT"},
            {"Name" : "Separator", "Alternative names" : [], "Type" : "VARCHAR(255)"},
            {"Name" : "Electrolyte name", "Alternative names" : ["Electrolyte Name","Electrolyte"], "Type" : "VARCHAR(255)"},
            {"Name" : "Electrolyte description", "Alternative names" : ["Electrolyte Description"], "Type" : "TEXT"},
            {"Name" : "Electrolyte position", "Alternative names" : ["Electrolyte Position"], "Type" : "INT"},
            {"Name" : "Electrolyte amount (uL)", "Alternative names" : ["Electrolyte Amount (uL)","Electrolyte Amount"], "Type" : "FLOAT"},
            {"Name" : "Electrolyte dispense order", "Alternative names" : ["Electrolyte Dispense Order"], "Type" : "VARCHAR(255)"},
            {"Name" : "Electrolyte amount before separator (uL)", "Alternative names" : ["Electrolyte Amount Before Separator (uL)", "Electrolyte Amount Before Seperator (uL)"], "Type" : "FLOAT"},
            {"Name" : "Electrolyte amount after separator (uL)", "Alternative names" : ["Electrolyte Amount After Separator (uL)","Electrolyte Amount After Seperator (uL)"], "Type" : "FLOAT"},
            {"Name" : "Anode rack position", "Alternative names" : ["Anode Rack Position","Anode Position"], "Type" : "INT"},
            {"Name" : "Anode type", "Alternative names" : ["Anode Type"], "Type" : "VARCHAR(255)"},
            {"Name" : "Anode description", "Alternative names" : ["Anode Description"], "Type" : "TEXT"},
            {"Name" : "Anode diameter (mm)", "Alternative names" : ["Anode Diameter (mm)", "Anode_Diameter", "Anode Diameter"], "Type" : "FLOAT"},
            {"Name" : "Anode mass (mg)", "Alternative names" : ["Anode Weight (mg)", "Anode Weight"], "Type" : "FLOAT"},
            {"Name" : "Anode current collector mass (mg)", "Alternative names" : ["Anode Current Collector Weight (mg)"], "Type" : "FLOAT"},
            {"Name" : "Anode active material mass fraction", "Alternative names" : ["Anode Active Material Weight Fraction", "Anode AM Content"], "Type" : "FLOAT"},
            {"Name" : "Anode active material mass (mg)", "Alternative names" : ["Anode Active Material Weight (mg)", "Anode AM Weight (mg)"], "Type" : "FLOAT"},
            {"Name" : "Anode C-rate definition areal capacity (mAh/cm2)", "Alternative names" : ["Anode C-rate Definition Areal Capacity (mAh/cm2)"], "Type" : "FLOAT"},
            {"Name" : "Anode C-rate definition specific capacity (mAh/g)", "Alternative names" : ["Anode C-rate Definition Specific Capacity (mAh/g)"], "Type" : "FLOAT"},
            {"Name" : "Anode balancing specific capacity (mAh/g)", "Alternative names" : ["Anode Balancing Specific Capacity (mAh/g)","Anode Practical Capacity (mAh/g)","Anode Nominal Specific Capacity (mAh/g)"], "Type" : "FLOAT"},
            {"Name" : "Anode balancing capacity (mAh)", "Alternative names" : ["Anode Balancing Capacity (mAh)","Anode Capacity (mAh)"], "Type" : "FLOAT"},
            {"Name" : "Cathode rack position", "Alternative names" : ["Cathode Rack Position","Cathode Position"], "Type" : "INT"},
            {"Name" : "Cathode type", "Alternative names" : ["Cathode Type"], "Type" : "VARCHAR(255)"},
            {"Name" : "Cathode description", "Alternative names" : ["Cathode Description"], "Type" : "TEXT"},
            {"Name" : "Cathode diameter (mm)", "Alternative names" : ["Cathode Diameter (mm)", "Cathode_Diameter", "Cathode Diameter"], "Type" : "FLOAT"},
            {"Name" : "Cathode mass (mg)", "Alternative names" : ["Cathode Weight (mg)"], "Type" : "FLOAT"},
            {"Name" : "Cathode current collector mass (mg)", "Alternative names" : ["Cathode Current Collector Weight (mg)"], "Type" : "FLOAT"},
            {"Name" : "Cathode active material mass fraction", "Alternative names" : ["Cathode Active Material Weight Fraction","Cathode AM Content"], "Type" : "FLOAT"},
            {"Name" : "Cathode active material mass (mg)", "Alternative names" : ["Cathode Active Material Weight (mg)","Cathode AM Weight (mg)"], "Type" : "FLOAT"},
            {"Name" : "Cathode C-rate definition areal capacity (mAh/cm2)", "Alternative names" : ["Cathode C-rate Definition Areal Capacity (mAh/cm2)"], "Type" : "FLOAT"},
            {"Name" : "Cathode C-rate definition specific capacity (mAh/g)", "Alternative names" : ["Cathode C-rate Definition Specific Capacity (mAh/g)"], "Type" : "FLOAT"},
            {"Name" : "Cathode balancing specific capacity (mAh/g)", "Alternative names" : ["Cathode Balancing Specific Capacity (mAh/g)","Cathode Practical Capacity (mAh/g)","Cathode Nominal Specific Capacity (mAh/g)"], "Type" : "FLOAT"},
            {"Name" : "Cathode balancing capacity (mAh)", "Alternative names" : ["Cathode Balancing Capacity (mAh)","Cathode Capacity (mAh)"], "Type" : "FLOAT"},
            {"Name" : "C-rate definition capacity (mAh)", "Alternative names" : ["C-rate Definition Capacity (mAh)","Capacity (mAh)", "C-rate Capacity (mAh)"], "Type" : "FLOAT"},
            {"Name" : "Target N:P ratio", "Alternative names" : ["Target N:P Ratio"], "Type" : "FLOAT"},
            {"Name" : "Minimum N:P ratio", "Alternative names" : ["Minimum N:P Ratio"], "Type" : "FLOAT"},
            {"Name" : "Maximum N:P ratio", "Alternative names" : ["Maximum N:P Ratio"], "Type" : "FLOAT"},
            {"Name" : "N:P ratio overlap factor", "Alternative names" : [], "Type" : "FLOAT"},
            {"Name" : "Casing type", "Alternative names" : ["Casing Type"], "Type" : "VARCHAR(255)"},
            {"Name" : "Separator diameter (mm)", "Alternative names" : ["Separator Diameter (mm)"], "Type" : "FLOAT"},
            {"Name" : "Spacer (mm)", "Alternative names" : [], "Type" : "FLOAT"},
            {"Name" : "Comment", "Alternative names" : ["Comments"], "Type" : "TEXT"},
            {"Name" : "Barcode", "Alternative names" : [], "Type" : "VARCHAR(255)"},
            {"Name" : "Subbatch number", "Alternative names" : ["Batch Number","Subbatch"], "Type" : "INT"},
            {"Name" : "Timestamp step 1", "Alternative names" : ["Timestamp Step 1"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 2", "Alternative names" : ["Timestamp Step 2"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 3", "Alternative names" : ["Timestamp Step 3"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 4", "Alternative names" : ["Timestamp Step 4"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 5", "Alternative names" : ["Timestamp Step 5"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 6", "Alternative names" : ["Timestamp Step 6"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 7", "Alternative names" : ["Timestamp Step 7"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 8", "Alternative names" : ["Timestamp Step 8"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 9", "Alternative names" : ["Timestamp Step 9"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 10", "Alternative names" : ["Timestamp Step 10"], "Type" : "DATETIME"},
            {"Name" : "Timestamp step 11", "Alternative names" : ["Timestamp Step 11"], "Type" : "DATETIME"},
        ],
    }

def create_database(config_path: Path) -> None:
    """Create/update a database file."""
    # Load the configuration
    with config_path.open("r") as f:
        config = json.load(f)
    database_path = Path(config["Database path"])

    # Check if database file already exists
    if (database_path/"database.db").exists():
        db_existed=True
    else:
        db_existed=False
        database_path.parent.mkdir(exist_ok=True)

    # Get the list of columns from the configuration
    columns = config["Sample database"]
    column_definitions = [f'`{col["Name"]}` {col["Type"]}' for col in columns]
    column_definitions += [
        "`Pipeline` VARCHAR(50)",
        "`Job ID` VARCHAR(255)",
        "FOREIGN KEY(`Pipeline`) REFERENCES pipelines(`Pipeline`)",
        "FOREIGN KEY(`Job ID`) REFERENCES jobs(`Job ID`)",
    ]

    # Connect to database, create tables
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(f'CREATE TABLE IF NOT EXISTS samples ({", ".join(column_definitions)})')
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "`Job ID` VARCHAR(255) PRIMARY KEY, "
            "`Sample ID` VARCHAR(255), "
            "`Pipeline` VARCHAR(50), "
            "`Status` VARCHAR(3), "
            "`Jobname` VARCHAR(50), "
            "`Server label` VARCHAR(255), "
            "`Server hostname` VARCHAR(255), "
            "`Job ID on server` INT, "
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
        conn.commit()

        # Check if there are new columns to add in samples table
        if db_existed:
            cursor.execute("PRAGMA table_info(samples)")
            existing_columns = cursor.fetchall()
            existing_columns = [col[1] for col in existing_columns]
            new_columns = [
                col["Name"]
                for col in config["Sample database"]
                if col["Name"] not in existing_columns
            ]
            removed_columns = [
                col
                for col in existing_columns
                if col not in [col["Name"] for col in config["Sample database"]]+["Pipeline", "Job ID"]
            ]
            if removed_columns:
                # Ask user to double confirm
                print(f"Database config would remove columns: {', '.join(removed_columns)}")
                msg1 = "Are you sure you want to delete these columns? Type 'yes' to confirm: "
                msg2 = "Are you really sure? This will permanently delete all data in these columns. Type 'really' to confirm: "
                if input(msg1) == "yes" and input(msg2) == "really":
                    for col in removed_columns:
                        cursor.execute(f'ALTER TABLE samples DROP COLUMN "{col}"')
                    conn.commit()
                    print(f"Columns {', '.join(removed_columns)} removed")
            if new_columns:
                # Add new columns
                for col in config["Sample database"]:
                    if col["Name"] in new_columns:
                        cursor.execute(f'ALTER TABLE samples ADD COLUMN "{col["Name"]}" {col["Type"]}')
                conn.commit()
                print(f"Adding new columns to database: {', '.join(new_columns)}")
            else:
                print("No changes to database configuration")
        else:
            print(f"Created database at {database_path}")

def main() -> None:
    """Create the shared config and database files."""
    try:
        config = get_config()
        if config.get("Shared config path") and config.get("Database path"):
            print("Set up already exists:")
            print(f"User config file: {config_path}")
            print(f"Shared config file: {config['Shared config path']}")
            print(f"Database: {config['Database path']}")
            if input("Continue with another set up? (yes/no): ") != "yes":
                print("Exiting")
                sys.exit()
    except (ValueError,FileNotFoundError):
        pass

    root_dir = Path(__file__).resolve().parents[1]

    choice = input("Connect to an existing configuration and database? (yes/no):")
    if choice in ["yes","y"]:
        shared_config_path = input("Please enter the path to the shared_config.json file:")
        # Check if shared_config already exists
        if not Path(shared_config_path).exists():
            print("File does not exist, exiting.")
            sys.exit()
        # Update the user config file with the shared config path
        with (root_dir/"config.json").open("w") as f:
            config = json.load(f)
            config["Shared config path"] = shared_config_path
            json.dump(config, f, indent=4)
        # If this runs successfully, the user can now run the app
        get_config()
        print("Successfully connected to existing configuration. Run the app with 'aurora-app'")
    elif choice in ["no", "n"]:
        choice = input("Create a new config, database and file structure? (yes/no):")
        if choice not in ["yes","y"]:
            print("Exiting")
            sys.exit()
        base_dir = Path(input("Please enter the folder path to be used for Aurora database and data storage:")).resolve()
        # If it doesn't exist, ask user if they want to create the folder
        if not base_dir.exists():
            if input(f"Folder {base_dir} does not exist. Create it? (yes/no): ") == "yes":
                base_dir.mkdir(parents=True)
            else:
                print("Exiting")
                sys.exit()

        # Create the folder structure
        (base_dir / "database").mkdir(exist_ok=True)
        (base_dir / "samples").mkdir(exist_ok=True)
        (base_dir / "snapshots").mkdir(exist_ok=True)
        (base_dir / "batches").mkdir(exist_ok=True)
        (base_dir / "payloads").mkdir(exist_ok=True)
        print(f"Created folder structure in {base_dir}")

        # Create the config file, if it already exists warn the user
        config_path = base_dir/"database"/"shared_config.json"
        if config_path.exists() and input(f"Config file {config_path} already exists. Overwrite it? (yes/no): ") != "yes":
                print("Exiting")
        with (base_dir/"database"/"shared_config.json").open("w") as f:
            json.dump(default_config(base_dir), f, indent=4)
        # Read the user config file and update with the shared config file
        try:
            config = get_config()
        except (FileNotFoundError, ValueError):
            # If it didn't exist before, get_config will have created a blank file
            with (root_dir/"config.json").open("r") as f:
                config = json.load(f)
        config["Shared config path"] = str(config_path)
        with (root_dir/"config.json").open("w") as f:
            json.dump(config, f, indent=4)
        print(f"Created shared config file at {config_path}")
        print(f"Updated user config at {root_dir/"config.json"} to point to shared config file")

        # Create the database
        create_database(config_path)

if __name__ == "__main__":
    main()
