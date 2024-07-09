""" Functions to create config and database files.

Config and database files are created if they do not exist during Cucumber
initialisation. The config file is created with some default values for
file paths and server information. The database samples table is created
with columns specified in the config file, with alternative names for handling
different naming conventions in output files.
"""
import os
import json
import sqlite3

base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, 'config.json')

def create_config() -> None:
    """ Create a config file if it doesn't exist. """
    if not os.path.exists(config_path):
        print(
            "Config file not found. Creating config.json file in the same directory as this script."
        )
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "Database Path": os.path.join(base_dir, "database", "database.db"),
                "Database Backup Folder Path": os.path.join(base_dir, "database", "backup"),
                "Samples Folder Path": os.path.join(base_dir, "samples"),
                "Snapshots Folder Path": os.path.join(base_dir, "snapshots"),
                "Processed Snapshots Folder Path": os.path.join(base_dir, "snapshots"),
                "Graphs Folder Path": os.path.join(base_dir, "snapshots"),

                "Servers" : [
                    {
                        "label": "example-server",
                        "hostname": "example-hostname",
                        "username": "user name on remote server",
                        "server_type": "tomato",
                        "command_prefix" : "this is put before any command, e.g. conda activate tomato ; ",
                        "tomato_scripts_path": "tomato-specific: this is put before ketchup in the command",
                    }
                ],
                "Sample Database" : [
                    {"Name" : "Sample ID", "Alternative Names" : ["sampleid"], "Type" : "VARCHAR(255) PRIMARY KEY"},
                    {"Name" : "Run ID", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Cell Number", "Alternative Names" : ["Battery_Number"], "Type" : "INT"},
                    {"Name" : "Actual N:P Ratio", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Rack Position", "Alternative Names" : ["Rack_Position"], "Type" : "INT"},
                    {"Name" : "Separator", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Electrolyte Name", "Alternative Names" : ["Electrolyte"], "Type" : "VARCHAR(255)"},
                    {"Name" : "Electrolyte Description", "Alternative Names" : [], "Type" : "TEXT"},
                    {"Name" : "Electrolyte Position", "Alternative Names" : [], "Type" : "INT"},
                    {"Name" : "Electrolyte Amount (uL)", "Alternative Names" : ["Electrolyte Amount"], "Type" : "FLOAT"},
                    {"Name" : "Electrolyte Dispense Order", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Electrolyte Amount Before Separator (uL)", "Alternative Names" : ["Electrolyte Amount Before Seperator (uL)"], "Type" : "FLOAT"},
                    {"Name" : "Electrolyte Amount After Separator (uL)", "Alternative Names" : ["Electrolyte Amount After Seperator (uL)"], "Type" : "FLOAT"},
                    {"Name" : "Anode Rack Position", "Alternative Names" : ["Anode Position"], "Type" : "INT"},
                    {"Name" : "Anode Type", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Anode Description", "Alternative Names" : [], "Type" : "TEXT"},
                    {"Name" : "Anode Diameter (mm)", "Alternative Names" : ["Anode_Diameter", "Anode Diameter"], "Type" : "FLOAT"},
                    {"Name" : "Anode Weight (mg)", "Alternative Names" : ["Anode Weight"], "Type" : "FLOAT"},
                    {"Name" : "Anode Current Collector Weight (mg)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Anode Active Material Weight Fraction", "Alternative Names" : ["Anode AM Content"], "Type" : "FLOAT"},
                    {"Name" : "Anode Active Material Weight (mg)", "Alternative Names" : ["Anode AM Weight (mg)"], "Type" : "FLOAT"},
                    {"Name" : "Anode C-rate Definition Areal Capacity (mAh/cm2)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Anode C-rate Definition Specific Capacity (mAh/g)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Anode Balancing Specific Capacity (mAh/g)", "Alternative Names" : ["Anode Practical Capacity (mAh/g)","Anode Nominal Specific Capacity (mAh/g)"], "Type" : "FLOAT"},
                    {"Name" : "Anode Balancing Capacity (mAh)", "Alternative Names" : ["Anode Capacity (mAh)"], "Type" : "FLOAT"},
                    {"Name" : "Cathode Rack Position", "Alternative Names" : ["Cathode Position"], "Type" : "INT"},
                    {"Name" : "Cathode Type", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Cathode Description", "Alternative Names" : [], "Type" : "TEXT"},
                    {"Name" : "Cathode Diameter (mm)", "Alternative Names" : ["Cathode_Diameter", "Cathode Diameter"], "Type" : "FLOAT"},
                    {"Name" : "Cathode Weight (mg)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Cathode Current Collector Weight (mg)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Cathode Active Material Weight Fraction", "Alternative Names" : ["Cathode AM Content"], "Type" : "FLOAT"},
                    {"Name" : "Cathode Active Material Weight (mg)", "Alternative Names" : ["Cathode AM Weight (mg)"], "Type" : "FLOAT"},
                    {"Name" : "Cathode C-rate Definition Areal Capacity (mAh/cm2)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Cathode C-rate Definition Specific Capacity (mAh/g)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Cathode Balancing Specific Capacity (mAh/g)", "Alternative Names" : ["Cathode Practical Capacity (mAh/g)","Cathode Nominal Specific Capacity (mAh/g)"], "Type" : "FLOAT"},
                    {"Name" : "Cathode Balancing Capacity (mAh)", "Alternative Names" : ["Cathode Capacity (mAh)"], "Type" : "FLOAT"},
                    {"Name" : "C-rate Definition Capacity (mAh)", "Alternative Names" : ["Capacity (mAh)", "C-rate Capacity (mAh)"], "Type" : "FLOAT"},
                    {"Name" : "Target N:P Ratio", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Minimum N:P Ratio", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Maximum N:P Ratio", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "N:P ratio overlap factor", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Casing Type", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Separator Diameter (mm)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Spacer (mm)", "Alternative Names" : [], "Type" : "FLOAT"},
                    {"Name" : "Comment", "Alternative Names" : ["Comments"], "Type" : "TEXT"},
                    {"Name" : "Barcode", "Alternative Names" : [], "Type" : "VARCHAR(255)"},
                    {"Name" : "Batch Number", "Alternative Names" : ["Subbatch"], "Type" : "INT"},
                    {"Name" : "Timestamp Step 1", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 2", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 3", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 4", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 5", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 6", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 7", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 8", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 9", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 10", "Alternative Names" : [], "Type" : "DATETIME"},
                    {"Name" : "Timestamp Step 11", "Alternative Names" : [], "Type" : "DATETIME"}
                ],
            }, f, indent=4)
    return


def create_database() -> None:
    """ Create a database file if it doesn't exist. """
    # Load the configuration
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    database_path = config["Database Path"]
    database_folder, _ = os.path.split(database_path)

    os.makedirs(database_folder, exist_ok=True)

    # Get the list of columns from the configuration
    columns = config["Sample Database"]
    column_definitions = [f'`{col["Name"]}` {col["Type"]}' for col in columns]
    column_definitions += [
        '`Pipeline` VARCHAR(50)',
        '`Job ID` VARCHAR(255)',
        'FOREIGN KEY(`Pipeline`) REFERENCES pipelines(`Pipeline`)',
        'FOREIGN KEY(`Job ID`) REFERENCES jobs(`Job ID`)'
    ]

    # Connect to database, create tables
    with sqlite3.connect(config["Database Path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(f'CREATE TABLE IF NOT EXISTS samples ({", ".join(column_definitions)})')
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS jobs ('
            '`Job ID` VARCHAR(255) PRIMARY KEY, '
            '`Sample ID` VARCHAR(255), '
            '`Pipeline` VARCHAR(50), '
            '`Status` VARCHAR(3), '
            '`Jobname` VARCHAR(50), '
            '`Job ID on Server` INT, '
            '`Submitted` DATETIME, '
            '`Payload` TEXT, '
            '`Comment` TEXT, '
            '`Last Checked` DATETIME, '
            '`Snapshot Status` VARCHAR(3), '
            '`Last Snapshot` DATETIME, '
            'FOREIGN KEY(`Sample ID`) REFERENCES samples(`Sample ID`),'
            'FOREIGN KEY(`Pipeline`) REFERENCES pipelines(`Pipeline`)'
            ')'
        )
        conn.commit()
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS pipelines ('
            '`Pipeline` VARCHAR(50) PRIMARY KEY, '
            '`Sample ID` VARCHAR(255),'
            '`Job ID` VARCHAR(255), '
            '`Flag` VARCHAR(10), '
            '`Last Checked` DATETIME, '
            '`Server Label` VARCHAR(255), '
            '`Server Hostname` VARCHAR(255), '
            'FOREIGN KEY(`Sample ID`) REFERENCES samples(`Sample ID`), '
            'FOREIGN KEY(`Job ID`) REFERENCES jobs(`Job ID`)'
            ')'
        )
        conn.commit()
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS results ('
            '`Sample ID` VARCHAR(255) PRIMARY KEY,'
            '`Pipeline` VARCHAR(50),'
            '`Status` VARCHAR(3),'
            '`Flag` VARCHAR(10),'
            '`Number of cycles` INT,'
            '`Capacity loss (%)` FLOAT,'
            '`First formation efficiency (%)` FLOAT,'
            '`Initial discharge specific capacity (mAh/g)` FLOAT,'
            '`Initial efficiency (%)` FLOAT,'
            '`Last discharge specific capacity (mAh/g)` FLOAT,'
            '`Last efficiency (%)` FLOAT,'
            '`Max voltage (V)` FLOAT,'
            '`Formation C` FLOAT,'
            '`Cycling C` FLOAT,'
            '`Last snapshot` DATETIME,'
            '`Last analysis` DATETIME,'
            '`Last plotted` DATETIME,'
            '`Snapshot Status` VARCHAR(3),'
            '`Snapshot Pipeline` VARCHAR(50),'
            'FOREIGN KEY(`Sample ID`) REFERENCES samples(`Sample ID`), '
            'FOREIGN KEY(`Pipeline`) REFERENCES pipelines(`Pipeline`)'
            ')'
        )
        conn.commit()

        # Check if there are new columns to add in samples table
        cursor.execute("PRAGMA table_info(samples)")
        existing_columns = cursor.fetchall()
        existing_columns = [col[1] for col in existing_columns]
        new_columns = [
            col["Name"]
            for col in config["Sample Database"]
            if col["Name"] not in existing_columns
        ]
        removed_columns = [
            col
            for col in existing_columns
            if col not in [col["Name"] for col in config["Sample Database"]]
        ]
        if removed_columns:
            print(f"Database config would remove columns: {', '.join(removed_columns)}")
            # Ask user to type 'yes' to confirm
            if input(
                "Are you sure you want to delete these columns? Type 'yes' to confirm: "
            ) == "yes":
                if input(
                    "Are you really sure? This will delete all data in these columns. "
                    "Type 'really' to confirm: "
                ) == "really":
                    # Remove columns
                    for col in removed_columns:
                        cursor.execute(f'ALTER TABLE samples DROP COLUMN "{col}"')
                    conn.commit()
                    print(f"Columns {', '.join(removed_columns)} removed")
        if new_columns:
            print(f"Adding new columns: {', '.join(new_columns)}")
            # Add new columns
            for col in config["Sample Database"]:
                if col["Name"] in new_columns:
                    cursor.execute(f'ALTER TABLE samples ADD COLUMN "{col["Name"]}" {col["Type"]}')
            conn.commit()
        else:
            print("No changes to configuration")


if __name__ == "__main__":
    create_config()
    create_database()
