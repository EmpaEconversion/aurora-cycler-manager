"""Copyright © 2025, Empa.

Harvest EC-lab .mpr files and convert to aurora-compatible hdf5 files.

Define the machines to grab files from in the config.json file.

get_mpr will copy all files from specified folders on a remote machine, if they
have been modified since the last time the function was called.

get_all_mprs does this for all machines defined in the config.

convert_mpr converts an mpr to a dataframe and optionally saves it as a hdf5
file. This file contains all cycling data as well as metadata from the mpr and
information about the sample from the database.

convert_all_mprs does this for all mpr files in the local snapshot folder, and
saves them to the processed snapshot folder.

Run the script to harvest and convert all mpr files.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import paramiko
import yadg
from dgbowl_schemas.yadg.dataschema import ExtractorFactory

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import add_data_to_db, get_sample_data
from aurora_cycler_manager.setup_logging import setup_logging
from aurora_cycler_manager.utils import parse_datetime, run_from_sample, ssh_connect
from aurora_cycler_manager.version import __url__, __version__

CONFIG = get_config()
logger = logging.getLogger(__name__)


def get_eclab_snapshot_folder() -> Path:
    """Get the path to the snapshot folder for eclab files."""
    snapshot_parent = CONFIG.get("Snapshots folder path")
    if not snapshot_parent:
        msg = (
            "No 'Snapshots folder path' in config file. "
            f"Please fill in the config file at {CONFIG.get('User config path')}.",
        )
        raise ValueError(msg)
    snapshot_path = Path(snapshot_parent) / "eclab_snapshots"
    snapshot_path.mkdir(parents=True, exist_ok=True)
    return snapshot_path


def get_mprs(
    server_label: str,
    server_hostname: str,
    server_username: str,
    server_shell_type: str,
    server_copy_folder: str,
    local_folder: Path | str,
    *,
    force_copy: bool = False,
) -> list[str]:
    """Get .mpr files from subfolders of specified folder.

    Args:
        server_label (str): Label of the server
        server_hostname (str): Hostname of the server
        server_username (str): Username to login with
        server_shell_type (str): Type of shell to use (powershell or cmd)
        server_copy_folder (str): Folder to search and copy .mpr and .mpl files
        local_folder (Path | str): Folder to copy files to
        force_copy (bool, optional): Copy all files regardless of modification date

    """
    # Set default cutoff date, use local timezone
    cutoff_datetime = datetime.fromtimestamp(0, tz=CONFIG["tz"])
    if not force_copy:  # Set cutoff date to last snapshot from database
        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT `Last snapshot` FROM harvester WHERE `Server label`=? AND `Server hostname`=? AND `Folder`=?",
                (server_label, server_hostname, server_copy_folder),
            )
            result = cursor.fetchone()
            cursor.close()
        if result:
            cutoff_datetime = parse_datetime(result[0])
    # Cannot use timezone or ISO8061 - not supported in PowerShell 5.1
    cutoff_date_str = cutoff_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Connect to the server and copy the files
    with paramiko.SSHClient() as ssh:
        ssh_connect(ssh, server_username, server_hostname)

        # Shell commands to find files modified since cutoff date
        # TODO: grab all the filenames and modified dates, copy if they are newer than local files not just cutoff date
        if server_shell_type == "powershell":
            command = (
                f"Get-ChildItem -Path '{server_copy_folder}' -Recurse "
                "| Where-Object { "
                f"$_.LastWriteTime -gt '{cutoff_date_str}' -and "
                "($_.Extension -eq '.mpl' -or $_.Extension -eq '.mpr')} "
                "| Select-Object -ExpandProperty FullName"
            )
        elif server_shell_type == "cmd":
            command = (
                'powershell.exe -Command "'
                f"Get-ChildItem -Path '{server_copy_folder}' -Recurse "
                "| Where-Object { "
                f"$_.LastWriteTime -gt '{cutoff_date_str}' -and "
                "($_.Extension -eq '.mpl' -or $_.Extension -eq '.mpr') "
                "} | Select-Object -ExpandProperty FullName"
                '"'
            )
        else:
            msg = f"Unknown shell type {server_shell_type} for server {server_label}"
            raise ValueError(msg)
        _stdin, stdout, stderr = ssh.exec_command(command)

        # Parse the output
        output = stdout.read().decode("utf-8").strip()
        error = stderr.read().decode("utf-8").strip()
        if error:
            msg = f"Error finding modified files: {error}"
            raise RuntimeError(msg)
        modified_files = output.splitlines()
        logger.info("Found %d modified files since %s", len(modified_files), cutoff_date_str)

        # Copy the files using SFTP
        current_datetime = datetime.now(timezone.utc)  # Keep time of copying for database
        new_files = []
        with ssh.open_sftp() as sftp:
            for file in modified_files:
                # Maintain the folder structure when copying
                relative_path = os.path.relpath(file, server_copy_folder)
                local_path = os.path.join(local_folder, relative_path)
                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                logger.info("Copying %s to %s", file, local_path)
                sftp.get(file, local_path)
                new_files.append(local_path)

    # Update the database
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO harvester (`Server label`, `Server hostname`, `Folder`) VALUES (?, ?, ?)",
            (server_label, server_hostname, server_copy_folder),
        )
        cursor.execute(
            "UPDATE harvester "
            "SET `Last snapshot` = ? "
            "WHERE `Server label` = ? AND `Server hostname` = ? AND `Folder` = ?",
            (current_datetime.isoformat(), server_label, server_hostname, server_copy_folder),
        )
        cursor.close()

        return new_files


def get_all_mprs(*, force_copy: bool = False) -> list[str]:
    """Get all MPR files from the folders specified in the config.

    The config file needs a key "EC-lab harvester" with a key "Snapshots folder
    path" with a location to save to, and a key "Servers" containing a list of
    dictionaries with the keys "label" and "EC-lab folder location".
    The "label" must match a server in the "Servers" list in the main config.
    """
    all_new_files = []
    snapshot_folder = get_eclab_snapshot_folder()

    # Check active biologic servers
    for server in CONFIG.get("Servers", {}):
        if server.get("server_type") == "biologic":
            new_files = get_mprs(
                server["label"],
                server["hostname"],
                server["username"],
                server["shell_type"],
                server["data_path"],
                snapshot_folder,
                force_copy=force_copy,
            )
            all_new_files.extend(new_files)

    # Check passive EC-lab harvesters
    for server in CONFIG.get("EC-lab harvester", {}).get("Servers", []):
        new_files = get_mprs(
            server["label"],
            server["hostname"],
            server["username"],
            server["shell_type"],
            server["EC-lab folder location"],
            snapshot_folder,
            force_copy=force_copy,
        )
        all_new_files.extend(new_files)
    return all_new_files


def get_mpr_data(
    mpr_file: str | Path | bytes,
    mpl_file: str | Path | bytes | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Convert mpr file to dataframe."""
    if isinstance(mpr_file, (str, Path)):
        mpr_file = Path(mpr_file)
        data = yadg.extractors.extract("eclab.mpr", mpr_file)
    elif isinstance(mpr_file, bytes):
        extractor = ExtractorFactory(extractor={"filetype": "eclab.mpr"}).extractor
        data = yadg.extractors.extract_from_bytes(
            source=mpr_file,
            extractor=extractor,
        )
    else:
        msg = "mpr_file must be str, Path, or raw bytes of mpr file."
        raise TypeError(msg)

    # Create dataframe
    df = pd.DataFrame()
    df["uts"] = data.coords["uts"].to_numpy()

    # Check if the time is incorrect and fix it
    df = check_mpr_uts(df, mpr_file, mpl_file)

    # Only keep certain columns in dataframe
    try:
        voltage_col = next(col for col in ("Ewe", "<Ewe>") if col in data.data_vars)
    except StopIteration as e:
        msg = "No voltage column found in data"
        raise KeyError(msg) from e
    df["V (V)"] = data.data_vars[voltage_col].to_numpy()
    I_units = {"A": 1, "mA": 1e-3, "uA": 1e-6}
    dq_units = {"mA·h": 3600 / 1000, "A·h": 3600}
    if "I" in data.data_vars:
        multiplier = I_units.get(data.data_vars["I"].attrs.get("units"), 1)
        df["I (A)"] = data.data_vars["I"].to_numpy() * multiplier
    elif "dq" in data.data_vars:  # If no current, calculate from dq
        multiplier = dq_units.get(data.data_vars["dq"].attrs.get("units"), 1)
        df["I (A)"] = (
            multiplier * data.data_vars["dq"].to_numpy() / np.diff(data.coords["uts"].to_numpy(), prepend=[np.inf])
        )
    else:
        df["I (A)"] = 0.0
    df["cycle_number"] = data.data_vars["half cycle"].to_numpy() // 2 if "half cycle" in data.data_vars else 0
    df["technique"] = data.data_vars["mode"].to_numpy() if "mode" in data.data_vars else 0
    df = df.astype({"V (V)": "float32", "I (A)": "float32"})
    df = df.astype({"technique": "int16", "cycle_number": "int32"})

    # Get metadata
    mpr_metadata = json.loads(data.attrs["original_metadata"])
    mpr_metadata["job_type"] = "eclab_mpr"
    yadg_metadata = {k: v for k, v in data.attrs.items() if k.startswith("yadg")}
    return df, mpr_metadata, yadg_metadata


def check_mpr_uts(
    df: pd.DataFrame, mpr_file: str | Path | bytes, mpl_file: str | Path | bytes | None = None
) -> pd.DataFrame:
    """Check if the unix timestamp is correct, attempts to find it from .mpl if not."""
    if df["uts"].to_numpy()[0] < 1000000000:  # The measurement started before 2001, assume wrong
        if not isinstance(mpr_file, (str, Path)) and not mpl_file:
            msg = "Incorrect start time in mpr file, reading from bytes, cannot auto-find mpl and no mpl provided."
            raise ValueError(msg)
        # Try to find the start datetime from the mpl
        if isinstance(mpr_file, (str, Path)) and not mpl_file:
            mpl_file = Path(mpr_file).with_suffix(".mpl")
            if not mpl_file.exists():
                msg = "Could not get acquisition start time from mpr, cannot find associated mpl file."
                raise ValueError(msg)
        if isinstance(mpl_file, (str, Path)):
            with Path(mpl_file).open(encoding="cp1252") as f:
                lines = f.readlines()
        elif isinstance(mpl_file, bytes):  # file-like object
            text = mpl_file.decode("cp1252", errors="replace")
            lines = text.splitlines()
        else:
            msg = "Cannot get start time from mpr or mpl file."
            raise ValueError(msg)
        try:
            for line in lines:
                if line.startswith("Acquisition started on : "):
                    datetime_str = line.split(":", 1)[1].strip()
                    # EC-lab mpl has no timezone info - assume it is in the same timezone
                    datetime_object = datetime.strptime(datetime_str, "%m/%d/%Y %H:%M:%S.%f")  # noqa: DTZ007
                    uts_timestamp = datetime_object.replace(tzinfo=CONFIG["tz"]).timestamp()
                    df["uts"] = df["uts"] + uts_timestamp
                    break
            else:
                msg = f"Incorrect start time in {mpr_file} and no start time in found {mpl_file}"
                raise ValueError(msg)
        except FileNotFoundError as e:
            msg = "Incorrect start time in mpr file, cannot find associated mpl file"
            raise ValueError(msg) from e
    return df


def convert_mpr(
    mpr_file: str | Path | bytes,
    mpl_file: str | Path | bytes | None = None,
    sample_id: str | None = None,
    job_id: str | None = None,
    modified_date: datetime | None = None,
    file_name: str | None = None,
    *,
    update_database: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Convert a ec-lab mpr to dataframe, optionally update database.

    Args:
        mpr_file (str, Path, bytes): path to the mpr file, or raw bytes
        mpl_file (str, Path, bytes, optional): path to the associated mpl file, or raw bytes
        update_database (bool, optional): whether to save data and update tables in database
        sample_id (str, optional): Sample ID as in database, REQUIRED if reading from bytes
        job_id (str, optional): Job ID as in the database, will check dataframe hash if not used
        modified_date (datetime, optional): Used for last snapshot time, inferred from mpr_file if str/path
        file_name (str, optional): Filename uploaded to the database, inferred from mpr_file if str/path

    Returns:
        pd.DataFrame: DataFrame containing the time-series cycling data
        dict: metadata

    Columns in output DataFrame:
    - uts: unix timestamp in seconds
    - V (V): Cell voltage in volts
    - I (A): Current in amps
    - cycle_number: number of cycles within one technique
    - technique: code of technique using Biologic convention

    """
    # Ensure there is a Sample ID, Run ID, and optionally creation date
    if isinstance(mpr_file, (str, Path)):
        mpr_file = Path(mpr_file).resolve()
        if not sample_id:
            run_id, sample_id = get_sampleid_from_mpr(mpr_file)
        else:
            run_id = run_from_sample(sample_id)
        if not modified_date:
            modified_date = datetime.fromtimestamp(mpr_file.stat().st_mtime, tz=timezone.utc)
    elif isinstance(mpr_file, bytes):  # file-like object
        if not sample_id:
            msg = "Sample ID is required if reading from bytes"
            raise ValueError(msg)
        if not file_name:
            msg = "File name is required if reading from bytes"
            raise ValueError(msg)
        run_id = run_from_sample(sample_id)
    else:
        msg = "mpr_file must be str, Path, or file-like object"
        raise TypeError(msg)

    # Get data and metadata from mpr (and mpl) files
    df, mpr_metadata, yadg_metadata = get_mpr_data(mpr_file, mpl_file)

    # get sample data from database
    try:
        sample_data = get_sample_data(sample_id)
    except ValueError:
        sample_data = {}

    # Metadata to add to file
    technique_codes = {1: "Constant current", 2: "Constant voltage", 3: "Open circuit voltage"}
    metadata = {
        "provenance": {
            "snapshot_file": str(mpr_file) if isinstance(mpr_file, (str, Path)) else None,
            "yadg_metadata": yadg_metadata,
            "aurora_metadata": {
                "mpr_conversion": {
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "eclab_harvester.convert_mpr",
                    "datetime": datetime.now(timezone.utc).isoformat(),
                },
            },
        },
        "job_data": mpr_metadata,
        "sample_data": sample_data,
        "glossary": {
            "uts": "Unix time stamp in seconds",
            "V (V)": "Cell voltage in volts",
            "I (A)": "Current across cell in amps",
            "cycle_number": "Number of cycles based on EC-lab half cycles",
            "technique": "code of technique using mpr conventions, see technique codes",
            "technique codes": technique_codes,
        },
    }

    # Save and update database
    if update_database:
        if not sample_id:
            logger.warning("Not saving %s, no valid Sample ID found", mpr_file)
            return df, metadata
        folder = Path(CONFIG["Processed snapshots folder path"]) / run_id / sample_id
        if not folder.exists():
            folder.mkdir(parents=True)

        # Get the file stem and path
        if file_name:
            file_stem = Path(file_name).stem
        else:
            assert isinstance(mpr_file, (str, Path))  # noqa: S101
            file_stem = Path(mpr_file).stem
        hdf5_filepath = folder / f"snapshot.{file_stem}.h5"

        # Add the file/job information to the database
        add_data_to_db(sample_id, file_stem, df, job_id)

        # Create the 'data' hdf5 dataset
        df.to_hdf(
            hdf5_filepath,
            key="data",
            mode="w",
            complib="blosc",
            complevel=9,
        )

        # Create a dataset called metadata and json dump the metadata
        with h5py.File(hdf5_filepath, "a") as f:
            f.create_dataset("metadata", data=json.dumps(metadata))
        logger.info("Saved %s", hdf5_filepath)

        # Update the database
        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)",
                (sample_id,),
            )
            cursor.execute(
                "UPDATE results SET `Last snapshot` = ? WHERE `Sample ID` = ?",
                (modified_date.isoformat() if modified_date else None, sample_id),
            )
            cursor.close()
    return df, metadata


def get_sampleid_from_mpr(mpr_rel_path: str | Path) -> tuple[str, str]:
    """Try to get the sample ID based on the remote file path."""
    # split the relative path into parts
    parts = Path(mpr_rel_path).parts
    run_id: str | None = None
    sample_number: int | None = None
    sample_id: str | None = None

    # From aurora-biologic API, files are stored base_folder/run_id/sample_id/job_id/files.mpr
    try:
        if parts[-4] == run_from_sample(parts[-3]):  # if this works, it was stored correctly
            run_id = parts[-4]
            sample_id = parts[-3]
            return run_id, sample_id
    except IndexError:
        pass

    # If that did not work, this may be 'out of bounds' data

    # A run ID lookup table is defined in case run IDs are wrong on the server
    run_id_lookup = CONFIG.get("EC-lab harvester", {}).get("Run ID lookup", {})

    # Get valid run IDs from the database
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT `Run ID` FROM samples")
        valid_run_ids = {row[0] for row in cursor.fetchall()}

    # This is usually stored as
    # base_folder/run_id/sample_number/files.mpr (run id index -3)
    # base_folder/run_id/sample_number/another_folder/files.mpr (run id index -4)
    for i in [-3, -4]:
        try:
            run_folder = parts[i]
            sample_folder = parts[i + 1]
        except IndexError as e:
            if i == -2:
                msg = f"Could not get run and sample number components for {mpr_rel_path}"
                raise ValueError(msg) from e
            continue
        # Check if the run_id is in the database or in the lookup table
        run_id = run_folder if run_folder in valid_run_ids else run_id_lookup.get(run_folder, None)

        if run_id:  # A run ID folder was found, now try to get the sample number
            try:
                sample_number = int(sample_folder)
                sample_id = f"{run_id}_{sample_number:02d}"
                break
            except ValueError:
                pass

    if not run_id or not sample_id:
        msg = f"Could not get Sample ID and Run ID for {mpr_rel_path}"
        raise ValueError(msg)
    return run_id, sample_id


def convert_all_mprs() -> None:
    """Convert all raw .mpr files to .h5.

    The config file needs a key "EC lab harvester" with the keys "Snapshots folder path".

    "EC lab harvester" can optionally contain "Run ID lookup" with a dictionary of run ID
    lookups for folders that are named differently to run ID on the server.
    """
    # walk through raw_folder and get the sample ID
    snapshot_folder = get_eclab_snapshot_folder()
    for dirpath, _dirnames, filenames in os.walk(snapshot_folder):
        for filename in filenames:
            if filename.endswith(".mpr"):
                full_path = Path(dirpath) / filename
                try:
                    convert_mpr(
                        full_path,
                        update_database=True,
                    )
                    logger.info("Converted %s", full_path)
                except (ValueError, IndexError, KeyError, RuntimeError):
                    logger.exception("Error converting %s", full_path)
                    continue


def main() -> None:
    """Harverst and convert all new mpr files."""
    new_files = get_all_mprs()
    for mpr_path in new_files:
        try:
            convert_mpr(
                mpr_path,
                update_database=True,
            )
        except (ValueError, IndexError, KeyError, RuntimeError):
            logger.exception("Error converting %s", mpr_path)
            continue


if __name__ == "__main__":
    setup_logging()
    main()
