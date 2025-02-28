"""Functions for interacting with the database."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.utils import run_from_sample

config = get_config()

def add_samples_from_object(samples: list[dict], overwrite: bool = False) -> None:
    """Add a samples to database from a list of dicts."""
    df = pd.DataFrame(samples)
    sample_df_to_db(df, overwrite)

def add_samples_from_file(json_file: str | Path, overwrite: bool = False) -> None:
    """Add a samples to database from a JSON file."""
    json_file = Path(json_file)
    _pre_check_sample_file(json_file)
    df = pd.read_json(json_file, orient="records")
    sample_df_to_db(df, overwrite)

def sample_df_to_db(df: pd.DataFrame, overwrite: bool = False) -> None:
    """Upload the sample dataframe to the database."""
    sample_ids = df["Sample ID"].tolist()
    if len(sample_ids) != len(set(sample_ids)):
        msg = "File contains duplicate 'Sample ID' keys"
        raise ValueError(msg)
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        msg = "File contains non-string 'Sample ID' keys"
        raise TypeError(msg)

    # Check if any sample already exists
    existing_sample_ids = get_all_sampleids()
    if not overwrite and any(sample_id in existing_sample_ids for sample_id in sample_ids):
        msg = "Sample IDs already exist in the database"
        raise ValueError(msg)

    # Recalculate some values
    df = _recalculate_sample_data(df)

    # Insert into database
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        for _, raw_row in df.iterrows():
            # Remove empty columns from the row
            row = raw_row.dropna()
            if row.empty:
                continue
            # Check if the row has sample ID
            if "Sample ID" not in row:
                continue
            placeholders = ", ".join("?" * len(row))
            columns = ", ".join(f"`{key}`" for key in row.keys())
            # Insert or ignore the row
            sql = f"INSERT OR IGNORE INTO samples ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(row))
            # Update the row
            updates = ", ".join(f"`{column}` = ?" for column in row.keys())
            sql = f"UPDATE samples SET {updates} WHERE `Sample ID` = ?"
            cursor.execute(sql, (*tuple(row), row["Sample ID"]))
        conn.commit()

def add_samples(json_file: str | Path, overwrite: bool = False) -> None:
    """Add a sample file to the database."""
    if isinstance(json_file, Path):
        _pre_check_sample_file(json_file)
        df = pd.read_json(json_file, orient="records")
    elif isinstance(json_file, str):
        # it is a binary string
        json_string = json_file.decode("utf-8")
        df = pd.read_json(json_string, orient="records")
    else:
        msg = "json_file must be a Path or a json string"
        raise TypeError(msg)
    sample_ids = df["Sample ID"].tolist()
    if len(sample_ids) != len(set(sample_ids)):
        msg = "File contains duplicate 'Sample ID' keys"
        raise ValueError(msg)
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        msg = "File contains non-string 'Sample ID' keys"
        raise TypeError(msg)

    # Check if any sample already exists
    existing_sample_ids = get_all_sampleids()
    if not overwrite and any(sample_id in existing_sample_ids for sample_id in sample_ids):
        msg = "Sample IDs already exist in the database"
        raise ValueError(msg)

    # Recalculate some values
    df = _recalculate_sample_data(df)

    # Insert into database
    with sqlite3.connect(config["Database path"]) as conn:
        df.to_sql("samples", conn, if_exists="append", index=False)

def _pre_check_sample_file(json_file: Path):
    """Raise error if file is not a sensible JSON file."""
    json_file = Path(json_file)
    # If csv is over 2 MB, do not insert
    if json_file.suffix != ".json":
        msg = f"File '{json_file}' is not a json file"
        raise ValueError(msg)
    if not json_file.exists():
        msg = f"File '{json_file}' does not exist"
        raise FileNotFoundError(msg)
    if json_file.stat().st_size > 2e6:
        msg = f"File {json_file} is over 2 MB, skipping"
        raise ValueError(msg)

    df = pd.read_json(json_string, orient="records")
def _recalculate_sample_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate some values for sample data before inserting into database."""
    # Pre-checks
    if "Sample ID" not in df.columns:
        msg = f"File '{json_file}' does not contain a 'Sample ID' column"
        raise ValueError(msg)
    if any(df["Sample ID"].duplicated()):
        msg = f"File '{json_file}' contains duplicate 'Sample ID' keys"
        raise ValueError(msg)
    if any(df["Sample ID"].isna()):
        msg = f"File '{json_file}' contains NaN 'Sample ID' keys"
        raise ValueError(msg)

    # Load the config file
    column_config = config["Sample database"]

    # Create a dictionary for lookup of alternative and case insensitive names
    col_names = [col["Name"] for col in column_config]
    alt_name_dict = {
        alt_name.lower(): item["Name"] for item in column_config for alt_name in item.get("Alternative names", [])
    }
    # Add on the main names in lower case
    alt_name_dict.update({col.lower(): col for col in col_names})

    # Rename columns to match the database
    rename = {}
    drop = []
    for column in df.columns:
        new_col_name = alt_name_dict.get(column.lower(), None)
        if new_col_name:
            rename[column] = new_col_name
        else:
            drop.append(column)
    df = df.rename(columns=rename)
    if drop:
        df = df.drop(columns=drop)

    # Change sample history to a JSON string
    if "Assembly history" in df.columns:
        df["Assembly history"] = df["Assembly history"].apply(json.dumps)

    # Calculate/overwrite certain columns
    # Active material masses
    required_columns = [
        "Anode mass (mg)",
        "Anode current collector mass (mg)",
        "Anode active material mass fraction",
    ]
    if all(col in df.columns for col in required_columns):
        df["Anode active material mass (mg)"] = (
            (df["Anode mass (mg)"] - df["Anode current collector mass (mg)"])
            * df["Anode active material mass fraction"]
        )
    required_columns = ["Cathode mass (mg)", "Cathode current collector mass (mg)", "Cathode active material mass fraction"]
    if all(col in df.columns for col in required_columns):
        df["Cathode active material mass (mg)"] = (
            (df["Cathode mass (mg)"] - df["Cathode current collector mass (mg)"])
            * df["Cathode active material mass fraction"]
        )
    # Capacities
    required_columns = ["Anode active material mass (mg)", "Anode balancing specific capacity (mAh/g)"]
    if all(col in df.columns for col in required_columns):
        df["Anode balancing capacity (mAh)"] = (
            1e-3 * df["Anode active material mass (mg)"] * df["Anode balancing specific capacity (mAh/g)"]
        )
    required_columns = ["Cathode active material mass (mg)", "Cathode balancing specific capacity (mAh/g)"]
    if all(col in df.columns for col in required_columns):
        df["Cathode balancing capacity (mAh)"] = (
            1e-3 * df["Cathode active material mass (mg)"] * df["Cathode balancing specific capacity (mAh/g)"]
        )
    # N:P ratio overlap factor
    required_columns = ["Anode diameter (mm)", "Cathode diameter (mm)"]
    if all(col in df.columns for col in required_columns):
        df["N:P ratio overlap factor"] = (df["Cathode diameter (mm)"]**2 / df["Anode diameter (mm)"]**2).fillna(0)
    # N:P ratio
    required_columns = ["Anode balancing capacity (mAh)", "Cathode balancing capacity (mAh)", "N:P ratio overlap factor"]
    if all(col in df.columns for col in required_columns):
        df["N:P ratio"] = (
            df["Anode balancing capacity (mAh)"] * df["N:P ratio overlap factor"]
            / df["Cathode balancing capacity (mAh)"]
        )
    # Run ID - if column is missing or where it is empty, find from the sample ID
    if "Run ID" not in df.columns:
        df["Run ID"] = df["Sample ID"].apply(lambda x: run_from_sample(x))
    else:
        df["Run ID"] = df["Run ID"].fillna(df["Sample ID"].apply(lambda x: run_from_sample(x)))
    return df

def get_all_sampleids() -> list[str]:
    """Get a list of all sample IDs in the database."""
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM samples")
        return [row[0] for row in cursor.fetchall()]

def get_job_data(job_id: str) -> dict:
    """Get all data about a job from the database."""
    with sqlite3.connect(config["Database path"]) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE `Job ID`=?", (job_id,))
        result = cursor.fetchone()
        if not result:
            msg = f"Job ID '{job_id}' not found in the database"
            raise ValueError(msg)
        job_data = dict(result)
        # Convert json strings to python objects
        payload = job_data.get("Payload")
        if payload:
            job_data["Payload"] = json.loads(payload)
    return job_data

def get_sample_data(sample_id: str) -> dict:
    """Get all data about a sample from the database."""
    with sqlite3.connect(config["Database path"]) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE `Sample ID`=?", (sample_id,))
        result = cursor.fetchone()
        if not result:
            msg = f"Sample ID '{sample_id}' not found in the database"
            raise ValueError(msg)
        sample_data = dict(result)
        # Convert json strings to python objects
        history = sample_data.get("Assembly history")
        if history:
            sample_data["Assembly history"] = json.loads(history)
    return sample_data
