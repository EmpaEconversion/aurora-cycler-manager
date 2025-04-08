"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Harvest Neware data files and convert to aurora-compatible .json.gz / .h5 files.

Define the machines to grab files from in the config.json file.

get_neware_data will copy all files from specified folders on a remote machine,
if they have been modified since the last time the function was called.

get_all_neware_data does this for all machines defined in the config.

convert_neware_data converts the file to a pandas dataframe and metadata
dictionary, and optionally saves as a hdf5 file or gzipped json file. This file
contains all cycling data as well as metadata and information about the sample
from the database.

convert_all_neware_data does this for all files in the local snapshot folder,
and saves them to the processed snapshot folder.

Run the script to harvest and convert all neware files.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import NewareNDA
import pandas as pd
import paramiko
import pytz
import xmltodict

from aurora_cycler_manager.config import CONFIG
from aurora_cycler_manager.database_funcs import get_sample_data
from aurora_cycler_manager.utils import run_from_sample
from aurora_cycler_manager.version import __url__, __version__

# Load configuration
tz = pytz.timezone(CONFIG.get("Time zone", "Europe/Zurich"))


def get_snapshot_folder() -> Path:
    """Get the path to the snapshot folder for neware files."""
    snapshot_parent = CONFIG.get("Snapshots folder path")
    if not snapshot_parent:
        msg = (
            "No 'Snapshots folder path' in config file. "
            f"Please fill in the config file at {CONFIG.get('User config path')}.",
        )
        raise ValueError(msg)
    return Path(snapshot_parent) / "neware_snapshots"


def harvest_neware_files(
    server_label: str,
    server_hostname: str,
    server_username: str,
    server_shell_type: str,
    server_copy_folder: str,
    local_folder: str | Path,
    local_private_key_path: str | None = None,
    force_copy: bool = False,
) -> list[Path]:
    """Get Neware files from subfolders of specified folder.

    Args:
        server_label (str): Label of the server
        server_hostname (str): Hostname of the server
        server_username (str): Username to login with
        server_shell_type (str): Type of shell to use (powershell or cmd)
        server_copy_folder (str): Folder to search and copy TODO file types
        local_folder (str): Folder to copy files to
        local_private_key_path (str, optional): Local private key path for ssh
        force_copy (bool): Copy all files regardless of modification date

    Returns:
        list of new files copied

    """
    cutoff_datetime = datetime.fromtimestamp(0)  # Set default cutoff date
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
            cutoff_datetime = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
    cutoff_date_str = cutoff_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Connect to the server and copy the files
    with paramiko.SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"Connecting to host {server_hostname} user {server_username}")
        ssh.connect(server_hostname, username=server_username, key_filename=local_private_key_path)

        # Shell commands to find files modified since cutoff date
        # TODO: grab all the filenames and modified dates, copy if they are newer than local files not just cutoff date
        if server_shell_type == "powershell":
            command = (
                f"Get-ChildItem -Path '{server_copy_folder}' -Recurse "
                f"| Where-Object {{ $_.LastWriteTime -gt '{cutoff_date_str}' -and ($_.Extension -eq '.xlsx' -or $_.Extension -eq '.ndax')}} "
                f"| Select-Object -ExpandProperty FullName"
            )
        elif server_shell_type == "cmd":
            command = (
                f"powershell.exe -Command \"Get-ChildItem -Path '{server_copy_folder}' -Recurse "
                f"| Where-Object {{ $_.LastWriteTime -gt '{cutoff_date_str}' -and ($_.Extension -eq '.xlsx' -or $_.Extension -eq '.ndax')}} "
                f'| Select-Object -ExpandProperty FullName"'
            )
        _stdin, stdout, stderr = ssh.exec_command(command)

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
                local_path = Path(local_folder) / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)  # Create local directory if it doesn't exist
                # Prepend the server label to the filename
                local_path = local_path.with_name(f"{server_label}-{local_path.name.replace("_","-").replace(" ","-")}")
                print(f"Copying {file} to {local_path}")
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
            (current_datetime.strftime("%Y-%m-%d %H:%M:%S"), server_label, server_hostname, server_copy_folder),
        )
        cursor.close()

    return new_files


def harvest_all_neware_files(force_copy: bool = False) -> list[Path]:
    """Get neware files from all servers specified in the config."""
    all_new_files = []
    snapshots_folder = get_snapshot_folder()
    for server in CONFIG.get("Neware harvester", {}).get("Servers", []):
        new_files = harvest_neware_files(
            server_label=server["label"],
            server_hostname=server["hostname"],
            server_username=server["username"],
            server_shell_type=server["shell_type"],
            server_copy_folder=server["Neware folder location"],
            local_folder=snapshots_folder,
            local_private_key_path=CONFIG["SSH private key path"],
            force_copy=force_copy,
        )
        all_new_files.extend(new_files)
    return all_new_files


def get_neware_xlsx_metadata(file_path: Path) -> dict:
    """Get metadata from a neware xlsx file.

    Args:
        file_path (Path): Path to the neware xlsx file

    Returns:
        dict: Metadata from the file

    """
    # Get the test info, including barcode / remarks
    df = pd.read_excel(file_path, sheet_name="test", header=None, engine="calamine")

    # In first column, find index where value is "Test information" and "Step plan"
    test_idx = df[df.iloc[:, 0] == "Test information"].index[0]
    step_idx = df[df.iloc[:, 0] == "Step plan"].index[0]

    # Get test info, remove empty columns and rows
    test_settings = df.iloc[test_idx + 1 : step_idx, :]
    test_settings = test_settings.dropna(axis=1, how="all")
    test_settings = test_settings.dropna(axis=0, how="all")

    # Flatten and convert to dict
    flattened = test_settings.to_numpy().flatten().tolist()
    flattened = [str(x) for x in flattened if str(x) != "nan"]
    test_info = {
        flattened[i]: flattened[i + 1] for i in range(0, len(flattened), 2) if flattened[i] and flattened[i] != "-"
    }
    test_info = {k: v for k, v in test_info.items() if (k and k not in ("-", "nan")) or (v and v not in ("-", "nan"))}

    # Payload
    payload = df.iloc[step_idx + 2 :, :]
    payload.columns = df.iloc[step_idx + 1]
    payload_dict = payload.to_dict(orient="records")

    payload_dict = [{k: v for k, v in record.items() if str(v) != "nan"} for record in payload_dict]

    # In Neware step information, 'Cycle' steps have different columns defined within the row
    # E.g. the "Voltage (V)" column has a value like "Cycle count:2"
    # We find these entires, and rename the key e.g. "Voltage (V)": "Cycle count:2" becomes "Cycle count": 2
    rename = {
        "Voltage(V)": "Voltage (V)",
        "Current(A)": "Current (A)",
        "Time(s)": "Time (s)",
        "Cut-off curr.(A)": "Cut-off current (A)",
    }
    for record in payload_dict:
        # change Voltage(V) to Voltage (V) if it exists
        for k, v in rename.items():
            if k in record:
                record[v] = record.pop(k)
        if record.get("Step Name") == "Cycle":
            # find values with ":" in them, and split them into key value pairs, delete the original key
            bad_key_vals = {k: v for k, v in record.items() if ":" in str(v)}
            for k, v in bad_key_vals.items():
                del record[k]
                new_k, new_v = v.split(":")
                record[new_k] = new_v

    # Add to test_info
    test_info["Payload"] = payload_dict
    return test_info


# change every dict with {'Value': 'some value'} to 'some value'
def _scrub_dict(d: dict) -> dict:
    """Scrub 'Value' and 'Main' keys from dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            if "Value" in v:
                d[k] = v["Value"]
            elif "Main" in v:
                d[k] = v["Main"]
            _scrub_dict(v)
    return d


def _convert_dict_values(d: dict) -> dict:
    """Convert values in step.xml dict to floats and scale to SI units."""
    for key, value in d.items():
        if isinstance(value, dict):
            if key == "Volt":
                for sub_key in value:
                    value[sub_key] = float(value[sub_key]) / 10000
            elif key in ("Curr", "Time"):
                for sub_key in value:
                    value[sub_key] = float(value[sub_key]) / 1000
            else:
                _convert_dict_values(value)
        elif isinstance(value, str):
            if key in ("Volt", "Stop_Volt"):
                d[key] = float(value) / 10000
            elif key in ("Curr", "Stop_Curr", "Time"):
                d[key] = float(value) / 1000
    return d


state_dict = {
    "1": "CC Chg",
    "2": "CC DChg",
    "3": "CV Chg",
    "4": "Rest",
    "5": "Cycle",
    "6": "End",
    "7": "CCCV Chg",
    "8": "CP DChg",
    "9": "CP Chg",
    "10": "CR DChg",
    "13": "Pause",
    "16": "Pulse",
    "17": "SIM",
    "19": "CV DChg",
    "20": "CCCV DChg",
    "21": "Control",
    "26": "CPCV DChg",
    "27": "CPCV Chg",
}
# For switching back to ints from NewareNDA
state_dict_rev = {v: int(k) for k, v in state_dict.items()}
state_dict_rev_underscored = {v.replace(" ", "_"): int(k) for k, v in state_dict.items()}


def _clean_ndax_step(d: dict) -> dict:
    """Extract useful info from dict from step.xml inside .ndax file."""
    # get rid of 'root' 'config' keys
    d = d["root"]["config"]
    # scrub 'Value' and 'Main' keys
    d = _scrub_dict(d)
    # put 'Head_Info' dict into the main dict
    d.update(d.pop("Head_Info"))

    # convert all values to floats
    _convert_dict_values(d)

    # convert 'Step_Info' to a more readable 'Payload' list
    step_list = []
    step_info = d.pop("Step_Info")
    for k, v in step_info.items():
        if k == "Num":
            continue
        new_step: dict[str, int | float | str] = {}
        new_step["Step Index"] = int(v.get("Step_ID"))
        new_step["Step Name"] = state_dict.get(v.get("Step_Type"), "Unknown")
        record = v.get("Record")
        if record:
            new_step["Record settings"] = (
                str(record.get("Time"))
                + "s/"
                + str(record.get("Curr", "0"))
                + "A/"
                + str(record.get("Volt", "0"))
                + "V"
            )
        limit = v.get("Limit")
        if limit:
            new_step["Current (A)"] = limit.get("Curr", 0)
            new_step["Voltage (V)"] = limit.get("Volt", 0)
            new_step["Time (s)"] = limit.get("Time", 0)
            new_step["Cut-off voltage (V)"] = limit.get("Stop_Volt", 0)
            new_step["Cut-off current (A)"] = limit.get("Stop_Curr", 0)
            other = limit.get("Other", {})
            if other:
                new_step["Cycle count"] = int(other.get("Cycle_Count", 0))
                new_step["Start step ID"] = int(other.get("Start_Step_ID", 0))
        # remove keys where value is 0 and add to list
        new_step = {k: v for k, v in new_step.items() if v != 0}
        step_list.append(new_step)
    d["Payload"] = step_list
    # Change some keys for consistency with xlsx
    d["Remarks"] = d.pop("Remark", "")
    d["Start step ID"] = int(d.pop("Start_Step", 1))
    # Get rid of keys that are not useful
    unwanted_keys = ["SMBUS", "Whole_Prt", "Guid", "Operate", "type", "version", "SCQ", "SCQ_F", "RateType", "Scale"]
    for k in unwanted_keys:
        d.pop(k, None)
    return d


def get_neware_ndax_metadata(file_path: Path) -> dict:
    """Extract metadata from Neware .ndax file.

    Args:
        file_path (Path): Path to the .ndax file

    Returns:
        dict: Metadata from the file

    """
    # Get step.xml and testinfo.xml from the .ndax file
    # get the step info from step.xml
    zf = zipfile.PyZipFile(str(file_path))
    step = zf.read("Step.xml")
    step_parsed = xmltodict.parse(step.decode(), attr_prefix="")
    metadata = _clean_ndax_step(step_parsed)

    # add test info
    testinfo = xmltodict.parse(zf.read("TestInfo.xml").decode(), attr_prefix="")
    testinfo = testinfo.get("root", {}).get("config", {}).get("TestInfo", {})
    metadata["Barcode"] = testinfo.get("Barcode")
    metadata["Start time"] = testinfo.get("StartTime")
    metadata["Step name"] = testinfo.get("StepName")
    metadata["Device type"] = testinfo.get("DevType")
    metadata["Device ID"] = testinfo.get("DevID")
    metadata["Subdevice ID"] = testinfo.get("UnitID") # Seems like this doesn't work from Neware's side
    metadata["Channel ID"] = testinfo.get("ChlID")
    metadata["Test ID"] = testinfo.get("TestID")
    metadata["Voltage range (V)"] = float(testinfo.get("VoltRange", 0))
    metadata["Current range (mA)"] = float(testinfo.get("CurrRange", 0))
    return metadata


def get_sampleid_from_metadata(metadata: dict) -> str | None:
    """Get sample ID from Remarks or Barcode in the Neware metadata."""
    # Get sampleid from test_info
    barcode_sampleid = metadata.get("Barcode", "")
    remark_sampleid = metadata.get("Remarks", "")
    sampleid = None

    # Check against known samples
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM samples")
        rows = cursor.fetchall()
        cursor.close()
    known_samples = [row[0] for row in rows]
    for possible_sampleid in [remark_sampleid, barcode_sampleid]:
        if possible_sampleid not in known_samples:
            # If sampleid is not known, it might be using the convention 'date-number-other'
            # Extract date and number
            sampleid_parts = re.split("_|-", possible_sampleid)
            if len(sampleid_parts) > 1:
                sampleid_date = sampleid_parts[0]
                sampleid_number = sampleid_parts[1].zfill(2)  # pad with zeros
                # Check if this is consistent with any known samples
                possible_samples = [
                    s for s in known_samples if s.startswith(sampleid_date) and s.endswith(sampleid_number)
                ]
                if len(possible_samples) == 1:
                    sampleid = possible_samples[0]
                    print(f"Barcode {possible_sampleid} inferred as Sample ID {sampleid}")
                    break
        else:
            sampleid = possible_sampleid
            print("Sample ID found:", sampleid)
            break
    if not sampleid:
        print(f"Barcode: '{barcode_sampleid}', or Remark: '{remark_sampleid}' not recognised as a Sample ID")
    return sampleid


def get_neware_xlsx_data(file_path: Path) -> pd.DataFrame:
    """Convert Neware xlsx file to dictionary."""
    df = pd.read_excel(file_path, sheet_name="record", header=0, engine="calamine")
    output_df = pd.DataFrame()
    output_df["V (V)"] = df["Voltage(V)"]
    output_df["I (A)"] = df["Current(A)"]
    output_df["technique"] = df["Step Type"].apply(lambda x: state_dict_rev.get(x, -1)).astype(int)
    # Every time the Step Type changes from a string containing "DChg" or "Rest" increment the cycle number
    output_df["cycle_number"] = (
        df["Step Type"].str.contains(r" DChg| DCHg|Rest", regex=True).shift(1)
        & df["Step Type"].str.contains(r" Chg", regex=True)
    ).cumsum()
    # convert date string from df["Date"] in format YYYY-MM-DD HH:MM:SS to uts timestamp in seconds
    output_df["uts"] = df["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    # add 1e-6 to Timestamp where Time is 0 - negligible and avoids errors when sorting
    output_df["uts"] = output_df["uts"] + (df["Time"] == 0) * 1e-6
    return output_df


def get_neware_ndax_data(file_path: Path) -> pd.DataFrame:
    """Convert Neware ndax file to dictionary."""
    df = NewareNDA.read(file_path)
    # where Time is 0, add 1e-6 to Timestamp - negligible and avoids errors when sorting
    df["Timestamp"] = df["Timestamp"] + pd.to_timedelta((df["Time"] == 0) * 1e-6, unit="s")
    output_df = pd.DataFrame()
    output_df["V (V)"] = df["Voltage"]
    output_df["I (A)"] = df["Current(mA)"] / 1000
    output_df["technique"] = df["Status"].apply(lambda x: state_dict_rev_underscored.get(x, 0)).astype(int)
    output_df["cycle_number"] = (
        df["Status"].str.contains(r"_DChg|_DCHg|Rest", regex=True).shift(1)
        & df["Status"].str.contains(r"_Chg", regex=True)
    ).cumsum()
    # convert datetime timestamp to uts timestamp in seconds
    output_df["uts"] = df["Timestamp"].apply(lambda x: x.timestamp())
    return output_df

def update_database_job(
    filepath: Path,
) -> None:
    """Update the database with job information.

    Args:
        filepath (Path): Path to the file

    """
    # Check that filename is in the format text_*_*_*_* where * is a number e.g. tt4_120_5_3_24.ndax
    # Otherwise we cannot get the full job ID, as sub-device ID is not reported properly
    if not re.match(r"^\S+-\d+-\d+-\d+-\d+", filepath.stem):
        msg = (
            "Filename not in expected format. "
            "Expect files in the format: "
            "{serverlabel}-{devid}-{subdevid}-{channelid}-{testid} "
            "e.g. nw4-120-1-3-24.ndax"
        )
        raise ValueError(msg)
    if filepath.suffix == ".xlsx":
        metadata = get_neware_xlsx_metadata(filepath)
    elif filepath.suffix == ".ndax":
        metadata = get_neware_ndax_metadata(filepath)
    else:
        msg = f"File type {filepath.suffix} not supported"
        raise ValueError(msg)
    sampleid = get_sampleid_from_metadata(metadata)
    if not sampleid:
        msg = f"Sample ID not found in metadata for file {filepath}"
        raise ValueError(msg)
    full_job_id = filepath.stem
    job_id_on_server = "-".join(full_job_id.split("-")[-4:])  # Get job ID from filename
    server_label = "-".join(full_job_id.split("-")[:-4])  # Get server label from filename
    pipeline = "-".join(job_id_on_server.split("-")[:-1]) # because sub-device ID reported properly
    submitted = metadata.get("Start time")
    payload = json.dumps(metadata.get("Payload"))
    last_snapshot_uts = filepath.stat().st_birthtime
    last_snapshot = datetime.fromtimestamp(last_snapshot_uts).strftime("%Y-%m-%d %H:%M:%S")
    server_hostname = next(
        (server["hostname"] for server in CONFIG.get("Neware harvester", {}).get("Servers", []) if server["label"] == server_label),
        None,
    )
    if not server_hostname:
        msg = f"Server hostname not found for server label {server_label}"
        raise ValueError(msg)

    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO jobs (`Job ID`) VALUES (?)",
            (full_job_id,),
        )
        cursor.execute(
            "UPDATE jobs SET "
            "`Job ID on server` = ?, `Pipeline` = ?, `Sample ID` = ?, "
            "`Server Label` = ?, `Server Hostname` = ?, `Submitted` = ?, "
            "`Payload` = ?, `Last Snapshot` = ?, `Job ID on server` = ? "
            "WHERE `Job ID` = ?",
            (
                job_id_on_server, pipeline, sampleid,
                server_label, server_hostname, submitted,
                payload, last_snapshot, job_id_on_server,
                full_job_id,
            ),
        )

def convert_neware_data(
    file_path: Path | str,
    output_jsongz_file: bool = False,
    output_hdf5_file: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Convert a neware file to a dataframe and save as a gzipped json file.

    Args:
        file_path (Path): Path to the neware file
        output_jsongz_file (bool): Whether to save the file as a gzipped json
        output_hdf5_file (bool): Whether to save the file as a hdf5

    Returns:
        tuple[pd.DataFrame, dict]: DataFrame containing the cycling data and metadata

    """
    # Get test information and Sample ID
    file_path = Path(file_path)
    if file_path.suffix == ".xlsx":
        job_data = get_neware_xlsx_metadata(file_path)
        job_data["job_type"] = "neware_xlsx"
        data = get_neware_xlsx_data(file_path)
    elif file_path.suffix == ".ndax":
        job_data = get_neware_ndax_metadata(file_path)
        job_data["job_type"] = "neware_ndax"
        data = get_neware_ndax_data(file_path)
    else:
        msg = f"File type {file_path.suffix} not supported"
        raise ValueError(msg)
    sampleid = get_sampleid_from_metadata(job_data)

    # If there is a valid Sample ID, get sample metadata from database
    sample_data = None
    if sampleid:
        sample_data = get_sample_data(sampleid)

    # Metadata to add
    job_data["Technique codes"] = state_dict
    current_datetime = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "provenance": {
            "snapshot_file": str(file_path),
            "aurora_metadata": {
                "mpr_conversion": {
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "neware_harvester.convert_neware_data",
                    "datetime": current_datetime,
                },
            },
        },
        "job_data": job_data,
        "sample_data": sample_data,
    }

    if output_jsongz_file or output_hdf5_file:
        if not sampleid:
            print(f"Not saving {file_path}, no valid Sample ID found")
            return data, metadata
        run_id = run_from_sample(sampleid)
        folder = Path(CONFIG["Processed snapshots folder path"]) / run_id / sampleid
        if not folder.exists():
            folder.mkdir(parents=True)

        if output_jsongz_file:
            file_name = f"snapshot.{file_path.stem}.json.gz"
            with gzip.open(folder / file_name, "wt") as f:
                json.dump({"data": data.to_dict(orient="list"), "metadata": metadata}, f)

        if output_hdf5_file:  # Save as hdf5
            file_name = f"snapshot.{file_path.stem}.h5"
            # Ensure smallest data types are used
            data = data.astype({"V (V)": "float32", "I (A)": "float32"})
            data = data.astype({"technique": "int16", "cycle_number": "int32"})
            data.to_hdf(
                folder / file_name,
                key="data",
                mode="w",
                complib="blosc",
                complevel=9,
            )
            # create a dataset called metadata and json dump the metadata
            with h5py.File(folder / file_name, "a") as f:
                f.create_dataset("metadata", data=json.dumps(metadata))

        # Update the database
        creation_date = datetime.fromtimestamp(
            file_path.stat().st_mtime,
        ).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)",
                (sampleid,),
            )
            cursor.execute(
                "UPDATE results SET `Last snapshot` = ? WHERE `Sample ID` = ?",
                (creation_date, sampleid),
            )
            cursor.close()

    return data, metadata


def convert_all_neware_data() -> None:
    """Convert all neware files to gzipped json files.

    The config file needs a key "Neware harvester" with the keys "Snapshots folder path"
    """
    # Get all xlsx and ndax files in the raw folder recursively
    snapshots_folder = get_snapshot_folder()
    neware_files = [file for file in snapshots_folder.rglob("*") if file.suffix in [".xlsx", ".ndax"]]
    for file in neware_files:
        try:
            convert_neware_data(file, output_hdf5_file=True)
        except ValueError as e:  # noqa: PERF203
            print(f"Error converting {file}: {e}")
        try:
            update_database_job(file)
        except ValueError as e:  # noqa: PERF203
            print(f"Error updating database for {file}: {e}")


def main() -> None:
    """Harvest and convert files that have changed."""
    new_files = harvest_all_neware_files()
    for file in new_files:
        print(f"Processing {file}")
        try:
            convert_neware_data(file, output_hdf5_file=True)
        except ValueError as e:  # noqa: PERF203
            print(f"Error converting {file}: {e}")
        try:
            update_database_job(file)
        except ValueError as e:  # noqa: PERF203
            print(f"Error updating database for {file}: {e}")


if __name__ == "__main__":
    main()
