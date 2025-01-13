"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Harvest EC-lab .mpr files and convert to aurora-compatible .json.gz / .h5 files.

Define the machines to grab files from in the config.json file.

get_mpr will copy all files from specified folders on a remote machine, if they
have been modified since the last time the function was called.

get_all_mprs does this for all machines defined in the config.

convert_mpr converts an mpr to a dataframe and optionally saves it as a hdf5
file or a gzipped json file. This file contains all cycling data as well as
metadata from the mpr and information about the sample from the database.

convert_all_mprs does this for all mpr files in the local snapshot folder, and
saves them to the processed snapshot folder.

Run the script to harvest and convert all mpr files.
"""
import gzip
import json
import os
import re
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import paramiko
import pytz
import yadg

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.version import __url__, __version__

# Load configuration
config = get_config()
snapshot_folder = Path(config["Snapshots folder path"]) / "eclab_snapshots"

def get_mprs(
    server_label: str,
    server_hostname: str,
    server_username: str,
    server_shell_type: str,
    server_copy_folder: str,
    local_private_key: str,
    local_folder: str,
    force_copy: bool = False,
) -> None:
    """Get .mpr files from subfolders of specified folder.

    Args:
        server_label (str): Label of the server
        server_hostname (str): Hostname of the server
        server_username (str): Username to login with
        server_shell_type (str): Type of shell to use (powershell or cmd)
        server_copy_folder (str): Folder to search and copy .mpr and .mpl files
        local_private_key (str): Local private key for ssh
        local_folder (str): Folder to copy files to
        force_copy (bool): Copy all files regardless of modification date

    """
    if force_copy:  # Set cutoff date to 1970
        cutoff_datetime = datetime.fromtimestamp(0)
    else:  # Set cutoff date to last snapshot from database
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT `Last snapshot` FROM harvester WHERE `Server label`=? AND `Server hostname`=? AND `Folder`=?",
                (server_label, server_hostname, server_copy_folder),
            )
            result = cursor.fetchone()
            cursor.close()
        cutoff_datetime = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S") if result else datetime.fromtimestamp(0)
    cutoff_date_str = cutoff_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Connect to the server and copy the files
    with paramiko.SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"Connecting to host {server_hostname} user {server_username}")
        ssh.connect(server_hostname, username=server_username, pkey=local_private_key)

        # Shell commands to find files modified since cutoff date
        # TODO: grab all the filenames and modified dates, copy if they are newer than local files not just cutoff date
        if server_shell_type == "powershell":
            command = (
                f'Get-ChildItem -Path \'{server_copy_folder}\' -Recurse '
                f'| Where-Object {{ $_.LastWriteTime -gt \'{cutoff_date_str}\' -and ($_.Extension -eq \'.mpl\' -or $_.Extension -eq \'.mpr\')}} '
                f'| Select-Object -ExpandProperty FullName'
            )
        elif server_shell_type == "cmd":
            command = (
                f'powershell.exe -Command "Get-ChildItem -Path \'{server_copy_folder}\' -Recurse '
                f'| Where-Object {{ $_.LastWriteTime -gt \'{cutoff_date_str}\' -and ($_.Extension -eq \'.mpl\' -or $_.Extension -eq \'.mpr\')}} '
                f'| Select-Object -ExpandProperty FullName"'
            )
        stdin, stdout, stderr = ssh.exec_command(command)

        # Parse the output
        output = stdout.read().decode("utf-8").strip()
        error = stderr.read().decode("utf-8").strip()
        if error:
            msg = f"Error finding modified files: {error}"
            raise RuntimeError(msg)
        modified_files = output.splitlines()
        print(f"Found {len(modified_files)} files modified since {cutoff_date_str}")

        # Copy the files using SFTP
        current_datetime = datetime.now()  # Keep time of copying for database
        new_files = []
        with ssh.open_sftp() as sftp:
            for file in modified_files:
                # Maintain the folder structure when copying
                relative_path = os.path.relpath(file, server_copy_folder)
                local_path = os.path.join(local_folder, relative_path)
                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                print(f"Copying {file} to {local_path}")
                sftp.get(file, local_path)
                new_files.append(local_path)

    # Update the database
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO harvester (`Server label`, `Server hostname`, `Folder`) "
            "VALUES (?, ?, ?)",
            (server_label, server_hostname, server_copy_folder),
        )
        cursor.execute(
            "UPDATE harvester "
            "SET `Last snapshot` = ? "
            "WHERE `Server label` = ? AND `Server hostname` = ? AND `Folder` = ?",
            (current_datetime.strftime("%Y-%m-%d %H:%M:%S"), server_label, server_hostname, server_copy_folder),
        )
        cursor.close()

        return new_files

def get_all_mprs(force_copy: bool = False) -> None:
    """Get all MPR files from the folders specified in the config.

    The config file needs a key "EC-lab harvester" with a key "Snapshots folder
    path" with a location to save to, and a key "Servers" containing a list of
    dictionaries with the keys "label" and "EC-lab folder location".
    The "label" must match a server in the "Servers" list in the main config.
    """
    all_new_files = []
    for server in config["EC-lab harvester"]["Servers"]:
        new_files = get_mprs(
            server["label"],
            server["hostname"],
            server["username"],
            server["shell_type"],
            server["EC-lab folder location"],
            paramiko.RSAKey.from_private_key_file(config["SSH private key path"]),
            snapshot_folder,
            force_copy,
        )
        all_new_files.extend(new_files)
    return all_new_files

def convert_mpr(
        mpr_file: str,
        output_jsongz_file: bool = False,
        output_hdf5_file: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
    """Convert a ec-lab mpr to dataframe, optionally save as hdf5 or zipped json file.

    Args:
        sampleid (str): sample ID from robot output
        mpr_file (str): path to the raw mpr file
        output_hdf_file (str, optional): path to save the output hdf5 file
        output_jsongz_file (str, optional): path to save the output zipped json file

    Returns:
        pd.DataFrame: DataFrame containing the cycling data

    Columns in output DataFrame:
    - uts: unix timestamp in seconds
    - V (V): Cell voltage in volts
    - I (A): Current in amps
    - cycle_number: number of cycles within one technique
    - technique: code of technique using Biologic convention

    TODO: use capacity to define C rate

    """
    # Normalize paths to avoid escape character issues
    mpr_file = os.path.normpath(mpr_file)

    # Try to get the sample ID and run ID from the path
    run_id, sample_id = get_sampleid_from_mpr(mpr_file)

    creation_date = datetime.fromtimestamp(
        os.path.getmtime(mpr_file),
    ).strftime("%Y-%m-%d %H:%M:%S")

    # Extract data from mpr file
    data = yadg.extractors.extract("eclab.mpr", mpr_file)

    df = pd.DataFrame()
    df["uts"] = data.coords["uts"].to_numpy()

    # Check if the time is incorrect and fix it
    if df["uts"].to_numpy()[0] < 1000000000:  # The measurement started before 2001, assume wrong
        # Grab the start time from mpl file
        mpl_file = mpr_file.replace(".mpr",".mpl")
        try:
            with open(mpl_file, encoding="ANSI") as f:
                lines = f.readlines()
            for line in lines:
                # Find the start datetime from the mpl
                found_start_time = False
                if line.startswith("Acquisition started on : "):
                    datetime_str = line.split(":",1)[1].strip()
                    datetime_object = datetime.strptime(datetime_str, "%m/%d/%Y %H:%M:%S.%f")
                    timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
                    uts_timestamp = timezone.localize(datetime_object).timestamp()
                    df["uts"] = df["uts"] + uts_timestamp
                    found_start_time = True
                    break
            if not found_start_time:
                warnings.warn(f"Incorrect start time in {mpr_file} and no start time in found {mpl_file}", stacklevel=2)
        except FileNotFoundError:
            warnings.warn(f"Incorrect start time in {mpr_file} and no mpl file found.", stacklevel=2)

    # Only keep certain columns in dataframe
    df["V (V)"] = data.data_vars["Ewe"].to_numpy()
    df["I (A)"] = (
        (3600 / 1000) * data.data_vars["dq"].to_numpy() /
        np.diff(data.coords["uts"].to_numpy(),prepend=[np.inf])
    )
    df["cycle_number"] = data.data_vars["half cycle"].to_numpy()//2
    df["technique"] = data.data_vars["mode"].to_numpy()

    # get sample data from database
    try:
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM samples WHERE `Sample ID`=?", (sample_id,))
            row = cursor.fetchone()
            columns = [column[0] for column in cursor.description]
            sample_data = dict(zip(columns, row))
    except Exception as e:
        print(f"Error getting job and sample data from database: {e}")
        sample_data = None

    # Get job data from the snapshot file
    mpr_metadata = json.loads(data.attrs["original_metadata"])
    mpr_metadata["job_type"] = "eclab_mpr"
    yadg_metadata = {k: v for k, v in data.attrs.items() if k.startswith("yadg")}

    # Metadata to add
    timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
    metadata = {
        "provenance": {
            "snapshot_file": mpr_file,
            "yadg_metadata": yadg_metadata,
            "aurora_metadata": {
                "mpr_conversion" : {
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "eclab_harvester.convert_mpr",
                    "datetime": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %z"),
                },
            },
        },
        "job_data": mpr_metadata,
        "sample_data": sample_data if sample_data is not None else {},
    }

    if output_hdf5_file or output_jsongz_file:  # Save and update database
        if not sample_id:
            print(f"Not saving {mpr_file}, no valid Sample ID found")
            return df, metadata
        folder = Path(config["Processed snapshots folder path"]) / run_id / sample_id
        if not folder.exists():
            folder.mkdir(parents=True)

        mpr_filename = Path(mpr_file).name
        if output_jsongz_file:  # Save as zipped json
            jsongz_filepath = folder / ("snapshot." + mpr_filename.replace(".mpr", ".json.gz"))
            with gzip.open(jsongz_filepath, "wt") as f:
                json.dump({"data": df.to_dict(orient="list"), "metadata": metadata}, f)
            print(f"Saved {jsongz_filepath}")

        if output_hdf5_file:  # Save as hdf5
            hdf5_filepath = folder / ("snapshot." + mpr_filename.replace(".mpr", ".h5"))
            # Ensure smallest data types are used
            df = df.astype({"V (V)": "float32", "I (A)": "float32"})
            df = df.astype({"technique": "int16", "cycle_number": "int32"})
            df.to_hdf(
                hdf5_filepath,
                key="data",
                mode="w",
                complib="blosc",
                complevel=9,
            )
            # create a dataset called metadata and json dump the metadata
            with h5py.File(hdf5_filepath, "a") as f:
                f.create_dataset("metadata", data=json.dumps(metadata))
            print(f"Saved {hdf5_filepath}")

        # Update the database
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)",
                (sample_id,),
            )
            cursor.execute(
                "UPDATE results "
                "SET `Last snapshot` = ? "
                "WHERE `Sample ID` = ?",
                (creation_date, sample_id),
            )
            cursor.close()
    return df, metadata

def get_sampleid_from_mpr(mpr_rel_path: str) -> str:
    """Try to get the sample ID based on the remote file path."""
    # split the relative path into parts
    parts = Path(mpr_rel_path).parts

    run_id_lookup = config["EC-lab harvester"].get("Run ID lookup", {})

    # Usually the run_ID is the 2nd parent folder and the sample number is the level above
    # If this is not the case, try the 3rd parent folder and 1st parent folder
    for i in [-3, -4, -2]:
        try:
            run_folder = parts[i]
            sample_folder = parts[i+1]
        except IndexError as e:
            if i == -2:
                msg = f"Could not get run and sample number components for {mpr_rel_path}"
                raise ValueError(msg) from e
            continue
        run_id = run_id_lookup.get(run_folder, None)
        if not run_id:
            with sqlite3.connect(config["Database path"]) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT `Run ID` FROM samples WHERE `Sample ID` LIKE ?", (f"%{run_folder}%",))
                result = cursor.fetchone()
                cursor.close()
            if result:
                run_id = result[0]
        if run_id:
            try:
                sample_number = int(sample_folder)
            except ValueError:
                match = re.search(r"cell(\d+)[_(]?", sample_folder)
                sample_number = int(match.group(1)) if match else None
            if sample_number:
                break

    if not run_id:
        msg = f"Could not find a Run ID for {mpr_rel_path}"
        raise ValueError(msg)
    if not sample_number:
        msg = f"Could not find a sample number for {mpr_rel_path}"
        raise ValueError(msg)
    sample_id = f"{run_id}_{sample_number:02d}"
    return run_id, sample_id

def convert_all_mprs() -> None:
    """Convert all raw .mpr files to gzipped json files.

    The config file needs a key "EC lab harvester" with the keys "Snapshots folder path".

    "EC lab harvester" can optionally contain "Run ID lookup" with a dictionary of run ID
    lookups for folders that are named differently to run ID on the server.
    """
    # walk through raw_folder and get the sample ID
    for dirpath, _dirnames, filenames in os.walk(snapshot_folder):
        for filename in filenames:
            if filename.endswith(".mpr"):
                full_path = Path(dirpath) / filename
                try:
                    convert_mpr(
                        full_path,
                        output_jsongz_file=False,
                        output_hdf5_file=True,
                    )
                    print(f"Converted {full_path}")
                except (ValueError, IndexError, KeyError, RuntimeError) as e:
                    print(f"Error converting {full_path}: {e}")
                    continue

def main() -> None:
    new_files = get_all_mprs()
    for mpr_path in new_files:
        try:
            convert_mpr(
                mpr_path,
                output_jsongz_file=False,
                output_hdf5_file=True,
            )
        except (ValueError, IndexError, KeyError, RuntimeError) as e:  # noqa: PERF203
            print(f"Error converting {mpr_path}: {e}")
            continue

if __name__ == "__main__":
    main()
