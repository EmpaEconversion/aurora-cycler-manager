"""Copyright © 2025, Empa.

Harvest Neware data files and convert to aurora-compatible hdf5 files.

Define the machines to grab files from in the config.json file.

get_neware_data will copy all files from specified folders on a remote machine,
if they have been modified since the last time the function was called.

get_all_neware_data does this for all machines defined in the config.

convert_neware_data converts the file to a pandas dataframe and metadata
dictionary, and optionally saves as a hdf5 file. This file contains all cycling
data as well as metadata and information about the sample from the database.

convert_all_neware_data does this for all files in the local snapshot folder,
and saves them to the processed snapshot folder.

Run the script to harvest and convert all neware files.
"""

import base64
import json
import logging
import os
import re
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import NewareNDA
import pandas as pd
import paramiko
import pytz
import xmltodict

from aurora_cycler_manager.analysis import analyse_sample
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import get_sample_data
from aurora_cycler_manager.setup_logging import setup_logging
from aurora_cycler_manager.utils import run_from_sample
from aurora_cycler_manager.version import __url__, __version__

# Load configuration
CONFIG = get_config()
tz = pytz.timezone(CONFIG.get("Time zone", "Europe/Zurich"))
logger = logging.getLogger(__name__)


def get_neware_snapshot_folder() -> Path:
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
        logger.info("Connecting to host %s user %s", server_hostname, server_username)
        ssh.connect(server_hostname, username=server_username, key_filename=CONFIG.get("SSH private key path"))

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
        logger.info("Found %d files modified since %s", len(modified_files), cutoff_date_str)

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
                local_path = local_path.with_name(
                    f"{server_label}-{local_path.name.replace('_', '-').replace(' ', '-')}"
                )
                logger.info("Copying '%s' to '%s'", file, local_path)
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


def snapshot_raw_data(job_id: str) -> Path | None:
    """Copy latest data from server into local .ndax file.

    Connects to server, searches for the raw .ndc files, copies into local .ndax file.

    Args:
        job_id (str): full job ID from database with server label e.g. nw4-22-6-4-26

    Returns:
        Path to the .ndax file created or modified, or None if no files updated.

    """
    # Job ID has form {server_label}-{device_id}-{subdevice_id}-{channel_id}-{test_id}
    server_label, dev_id, subdev_id, channel_id, test_id = job_id.split("-")
    # Neware has a different format for raw data. Folder is raw_data_folder/YYYYMMDD/,
    # file is YYYYMMDD_HHMMSS_27{len(4) 0 padded device_id}_0_{(subdevid-1)*8 + channel_id}_{test_id} + file type .ndc

    # Get the job server label and submit date from the database
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT `Submitted`, `Server label` FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        row = cursor.fetchone()
        cursor.close()
    if not row:
        msg = f"No job found with ID '{job_id}' in database."
        raise ValueError(msg)
    submitted = row[0].split(" ")[0].replace("-", "")  # Get YYYYMMDD format
    server_label = row[1]

    # Get the server from the config
    server = next((server for server in CONFIG["Neware harvester"]["Servers"] if server["label"] == server_label), None)
    if not server:
        msg = f"No server found with label '{server_label}' in config."
        raise ValueError(msg)
    server_hostname = server["hostname"]
    server_username = server["username"]
    server_shell_type = server["shell_type"]
    raw_data_folder = server.get("Neware raw folder location", "C:/Program Files (x86)/NEWARE/BTSServer80/NdcFile/")

    # Build the paths to check - assumes device type 27
    full_folder = raw_data_folder + submitted
    full_folder_alt = full_folder + "_NoTestInfoData"
    file_middle = (
        "27" + str(dev_id).zfill(4) + "_0_" + str((int(subdev_id) - 1) * 8 + int(channel_id)) + "_" + str(test_id)
    )

    # Powershell command to return paths of files matching the pattern
    powershell_command = f"""
        $ProgressPreference = 'SilentlyContinue'

        $searchFolders = @("{full_folder}", "{full_folder_alt}")
        $file_middle = "{file_middle}"
        $file_endings = @(".ndc", "_step.ndc", "_runInfo.ndc", "_log.ndc", "_es.ndc")

        $foundFiles = [ordered]@{{}}
        foreach ($ending in $file_endings) {{
            $matchedFile = $null
            foreach ($folder in $searchFolders) {{
                if (Test-Path $folder) {{
                    $pattern = "*$file_middle$ending"
                    $match = Get-ChildItem -Path $folder -Recurse -File -ErrorAction SilentlyContinue |
                        Where-Object {{ $_.Name -like $pattern }} | Select-Object -First 1

                    if ($match) {{
                        $matchedFile = $match.FullName
                        break
                    }}
                }}
            }}
            $foundFiles[$ending] = $matchedFile
        }}
        $foundFiles | ConvertTo-Json -Compress
    """

    # Connect to the server
    with paramiko.SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        logger.info("Connecting to host %s user %s", server_hostname, server_username)
        ssh.connect(server_hostname, username=server_username, key_filename=CONFIG.get("SSH private key path"))

        # Use powershell command to search for the files on the server
        if server_shell_type == "powershell":
            command = powershell_command
        elif server_shell_type == "cmd":
            # Base64 encode the command to avoid quote/semicolon issues
            encoded_script = base64.b64encode(powershell_command.encode("utf-16le")).decode("ascii")
            command = f"powershell.exe -EncodedCommand {encoded_script}"
        else:
            msg = f"Unsupported shell type '{server_shell_type}' for server {server_label}."
            raise ValueError(msg)
        _stdin, stdout, stderr = ssh.exec_command(command)

        # Parse the output
        output = stdout.read().decode("utf-8").strip()
        error = stderr.read().decode("utf-8").strip()
        if error:
            msg = f"Error finding raw files: {error}"
            raise RuntimeError(msg)
        found_files = json.loads(output)

        # Create or update the ndax file with new raw data
        ndax_path = None
        if any(file is not None for file in found_files.values()):
            ndax_path = get_neware_snapshot_folder() / f"{job_id}.ndax"
            logger.info("Updating ndax file at '%s'", ndax_path)

            # Create a temporary directory to store the files
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)

                # If ndax already exists, extract it to the temporary directory
                if ndax_path.exists():
                    with zipfile.ZipFile(ndax_path, "r") as zf:
                        zf.extractall(tmp_path)

                # Copy the new files over, replacing any existing ones
                with ssh.open_sftp() as sftp:
                    for ending, file in found_files.items():
                        sftp.get(file, tmp_path / ("data" + ending))

                # Write a new zip
                with zipfile.ZipFile(ndax_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file in tmp_path.rglob("*"):
                        if file.is_file():
                            zf.write(file, arcname=file.relative_to(tmp_path))
    return ndax_path


def harvest_all_neware_files(force_copy: bool = False) -> list[Path]:
    """Get neware files from all servers specified in the config."""
    all_new_files = []
    snapshots_folder = get_neware_snapshot_folder()
    for server in CONFIG.get("Neware harvester", {}).get("Servers", []):
        new_files = harvest_neware_files(
            server_label=server["label"],
            server_hostname=server["hostname"],
            server_username=server["username"],
            server_shell_type=server["shell_type"],
            server_copy_folder=server["Neware folder location"],
            local_folder=snapshots_folder,
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

    # Check if test is finished
    df = pd.read_excel(file_path, sheet_name="log", header=None, engine="calamine")
    if df["Event"].iloc[-1] == "Finished test":
        test_info["End time"] = df["Time"].iloc[-1]
        test_info["Finished"] = True
    else:
        test_info["Finished"] = False

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
    # Get step.xml and testinfo.xml from the .ndax file, if not present check database for metadata

    zf = zipfile.PyZipFile(str(file_path))

    if "Step.xml" in zf.namelist() and "TestInfo.xml" in zf.namelist():
        # Get the step info from step.xml
        step = zf.read("Step.xml")
        step_parsed = xmltodict.parse(step.decode(errors="replace"), attr_prefix="")
        metadata = _clean_ndax_step(step_parsed)

        # Add test info
        testinfo = xmltodict.parse(zf.read("TestInfo.xml").decode(errors="replace"), attr_prefix="")
        testinfo = testinfo.get("root", {}).get("config", {}).get("TestInfo", {})
        metadata["Barcode"] = testinfo.get("Barcode")
        metadata["Start time"] = testinfo.get("StartTime")
        endtime = testinfo.get("EndTime")
        metadata["End time"] = endtime if endtime else None
        metadata["Finished"] = bool(endtime)
        metadata["Step name"] = testinfo.get("StepName")
        metadata["Device type"] = testinfo.get("DevType")
        metadata["Device ID"] = testinfo.get("DevID")
        metadata["Subdevice ID"] = testinfo.get("UnitID")  # Seems like this doesn't work from Neware's side
        metadata["Channel ID"] = testinfo.get("ChlID")
        metadata["Test ID"] = testinfo.get("TestID")
        metadata["Voltage range (V)"] = float(testinfo.get("VoltRange", 0))
        metadata["Current range (mA)"] = float(testinfo.get("CurrRange", 0))

    else:
        metadata = get_neware_metadata_from_db(file_path.stem)

    return metadata


def get_neware_metadata_from_db(job_id: str) -> dict:
    """Get metadata from the database for a Neware file.

    Args:
        job_id (str): Name of Job ID, (should be filename without extension)

    Returns:
        dict: Metadata from the database

    """
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        row = cursor.fetchone()

        # Check if the job is still running with pipelines table
        cursor.execute(
            "SELECT 1 FROM pipelines WHERE `Job ID` = ? LIMIT 1",
            (job_id,),
        )
        finished = cursor.fetchone() is None

    if not row:
        msg = f"No metadata found for Job ID '{job_id}' in database."
        raise ValueError(msg)
    row = dict(row)
    # convert string to xml then to dict

    xml_payload = xmltodict.parse(row["Payload"], attr_prefix="")
    metadata = _clean_ndax_step(xml_payload)
    server_label, Device_ID, Subdevice_ID, Channel_ID, Test_ID = job_id.split("-")
    metadata["Device ID"] = Device_ID
    metadata["Subdevice ID"] = Subdevice_ID
    metadata["Channel ID"] = Channel_ID
    metadata["Test ID"] = Test_ID
    metadata["Barcode"] = row["Sample ID"]
    metadata["Finished"] = finished
    return metadata


def get_known_samples() -> list[str]:
    """Get a list of Sample IDs from the database."""
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Sample ID` FROM samples")
        rows = cursor.fetchall()
        cursor.close()
    return [row[0] for row in rows]


def get_sampleid_from_metadata(metadata: dict, known_samples: list[str] | None = None) -> str | None:
    """Get sample ID from Remarks or Barcode in the Neware metadata."""
    # Get sampleid from test_info
    barcode_sampleid = metadata.get("Barcode", "")
    remark_sampleid = metadata.get("Remarks", "")
    sampleid = None

    if not known_samples:
        known_samples = get_known_samples()
    for possible_sampleid in [barcode_sampleid, remark_sampleid]:
        if possible_sampleid in known_samples:
            sampleid = possible_sampleid
            break
    if sampleid is None:  # May be user error, try some common fixes
        logger.info(
            "Could not find Sample ID '%s' or '%s' in database, trying to infer it", barcode_sampleid, remark_sampleid
        )
        for possible_sampleid in [remark_sampleid, barcode_sampleid]:
            # Should be YYMMDD-otherstuff-XX, where XX is a number
            sampleid_parts = re.split("_|-", possible_sampleid)
            if len(sampleid_parts) > 1:
                if len(sampleid_parts[0]) == 8:  # YYYYMMDD -> YYMMDD
                    sampleid_parts[0] = sampleid_parts[0][2:]
                sampleid_parts[-1] = sampleid_parts[-1].zfill(2)  # pad with zeros
                # Check if this is consistent with any known samples
                possible_samples = [s for s in known_samples if all(parts in s for parts in sampleid_parts)]
                if len(possible_samples) > 1:
                    possible_samples = [s for s in possible_samples if "_".join(sampleid_parts) in s]
                if len(possible_samples) == 1:
                    sampleid = possible_samples[0]
                    logger.info("Barcode '%s' inferred as Sample ID '%s'", possible_sampleid, sampleid)
                    break
    if not sampleid:
        logger.warning(
            "Barcode: '%s', or Remark: '%s' not recognised as a Sample ID", barcode_sampleid, remark_sampleid
        )
    return sampleid


def get_neware_xlsx_data(file_path: Path) -> pd.DataFrame:
    """Convert Neware xlsx file to dictionary."""
    df = pd.read_excel(file_path, sheet_name="record", header=0, engine="calamine")
    required_columns = ["Voltage(V)", "Current(A)", "Step Type", "Date", "Time"]
    if not all(col in df.columns for col in required_columns):
        msg = f"Missing required columns in {file_path}: {required_columns}"
        raise ValueError(msg)
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
    # convert time to 64bit, add 1e-6 at new steps - negligible and avoids errors when sorting
    df["Time"] = df["Time"].astype("float64") + (df["Time"] == 0) * 1e-6

    output_df = pd.DataFrame()
    output_df["V (V)"] = df["Voltage"]
    output_df["I (A)"] = df["Current(mA)"] / 1000
    output_df["technique"] = df["Status"].apply(lambda x: state_dict_rev_underscored.get(x, 0)).astype(int)
    output_df["cycle_number"] = (
        df["Status"].str.contains(r"_DChg|_DCHg|Rest", regex=True).shift(1)
        & df["Status"].str.contains(r"_Chg", regex=True)
    ).cumsum()
    # "Date" from df is not reliable, instead calc from "Time" and add first date to get uts timestamp
    # Get last time for each step and add to next steps
    last_times = df.groupby("Step")["Time"].last()
    offsets = last_times.shift(fill_value=0).cumsum()
    total_time = df["Time"] + df["Step"].map(offsets)
    # Get first datetime, add the total time to get uts
    start_uts = float(df["Timestamp"].iloc[0].timestamp())
    output_df["uts"] = start_uts + total_time

    return output_df


def update_database_job(
    filepath: Path,
    sampleid: str | None = None,
    known_samples: list[str] | None = None,
) -> None:
    """Update the database with job information.

    Args:
        filepath (Path): Path to the file
        sampleid (str, optional): Sample ID to use, otherwise find from metadata
        known_samples (list[str], optional): List of known Sample IDs to check against

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
    if sampleid is None:
        sampleid = get_sampleid_from_metadata(metadata, known_samples)
    if not sampleid:
        msg = f"Sample ID not found in metadata for file {filepath}"
        raise ValueError(msg)
    full_job_id = filepath.stem
    job_id_on_server = "-".join(full_job_id.split("-")[-4:])  # Get job ID from filename
    server_label = "-".join(full_job_id.split("-")[:-4])  # Get server label from filename
    pipeline = "-".join(job_id_on_server.split("-")[:-1])  # because sub-device ID reported properly
    submitted = metadata.get("Start time")
    payload = json.dumps(metadata.get("Payload"))
    last_snapshot_uts = filepath.stat().st_birthtime
    last_snapshot = datetime.fromtimestamp(last_snapshot_uts).strftime("%Y-%m-%d %H:%M:%S")
    server_hostname = next(
        (
            server["hostname"]
            for server in CONFIG.get("Neware harvester", {}).get("Servers", [])
            if server["label"] == server_label
        ),
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
                job_id_on_server,
                pipeline,
                sampleid,
                server_label,
                server_hostname,
                submitted,
                payload,
                last_snapshot,
                job_id_on_server,
                full_job_id,
            ),
        )


def convert_neware_data(
    file_path: Path | str,
    sampleid: str | None = None,
    known_samples: list[str] | None = None,
    output_hdf5_file: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Convert a neware file to a dataframe and save as hdf5.

    Args:
        file_path (Path): Path to the neware file
        output_hdf5_file (bool): Whether to save the file as a hdf5
        known_samples (list[str], optional): List of known Sample IDs to check against

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
    if sampleid is None:
        sampleid = get_sampleid_from_metadata(job_data, known_samples)

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
        "glossary": {
            "uts": "Unix time stamp in seconds",
            "V (V)": "Cell voltage in volts",
            "I (A)": "Current across cell in amps",
            "cycle_number": "Number of cycles, increments when changing from discharge or rest to charge",
            "technique": "Code of technique using Neware convention, see technique codes",
            "technique codes": state_dict,
        },
    }

    if output_hdf5_file:
        if not sampleid:
            logger.warning("Not saving %s, no valid Sample ID found", file_path)
            return data, metadata
        run_id = run_from_sample(sampleid)
        folder = Path(CONFIG["Processed snapshots folder path"]) / run_id / sampleid
        if not folder.exists():
            folder.mkdir(parents=True)

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
    """Convert all neware files to hdf5 files.

    The config file needs a key "Neware harvester" with the keys "Snapshots folder path"
    """
    # Get all xlsx and ndax files in the raw folder recursively
    snapshots_folder = get_neware_snapshot_folder()
    neware_files = [file for file in snapshots_folder.rglob("*") if file.suffix in [".xlsx", ".ndax"]]
    new_samples = set()
    known_samples = get_known_samples()
    for file in neware_files:
        logger.info("Converting %s", file)
        try:
            _data, metadata = convert_neware_data(file, output_hdf5_file=True, known_samples=known_samples)
            if metadata is not None:
                sampleid = metadata.get("sample_data", {}).get("Sample ID") if metadata.get("sample_data") else None
                if sampleid:
                    update_database_job(file, sampleid=sampleid, known_samples=known_samples)
                    new_samples.add(sampleid)
                    logger.info("Converted %s", sampleid)
        except (ValueError, AttributeError):
            logger.exception("Error converting %s", file)
    for sample in new_samples:
        logger.info("Analysing %d samples", len(new_samples))
        try:
            analyse_sample(sample)
            logger.info("Analysed %s", sample)
        except (ValueError, PermissionError, RuntimeError, FileNotFoundError, KeyError):
            logger.exception("Error analysing %s", sample)


def main() -> None:
    """Harvest and convert files that have changed."""
    new_files = harvest_all_neware_files()
    new_samples = set()
    known_samples = get_known_samples()
    logger.info("Processing %d files", len(new_files))
    for file in new_files:
        logger.info("Processing %s", file)
        try:
            _data, metadata = convert_neware_data(file, output_hdf5_file=True, known_samples=known_samples)
            update_database_job(file, known_samples=known_samples)
            if metadata is not None:
                sampleid = metadata.get("sample_data", {}).get("Sample ID")
                if sampleid:
                    new_samples.add(sampleid)
                    logger.info("Converted %s", sampleid)
        except Exception:
            logger.exception("Error converting %s", file)
    logger.info("Analysing %d samples", len(new_samples))
    for sample in new_samples:
        try:
            analyse_sample(sample)
            logger.info("Analysed %s", sample)
        except Exception:  # noqa: PERF203
            logger.exception("Error analysing %s", sample)


if __name__ == "__main__":
    setup_logging()
    main()
