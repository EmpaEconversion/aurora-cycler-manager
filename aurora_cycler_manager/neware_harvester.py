""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Harvest Neware data files and convert to aurora-compatible gzipped json files. 

Define the machines to grab files from in the config.json file.

get_neware_data will copy all files from specified folders on a remote machine, if they have been
modified since the last time the function was called.

get_all_neware_data does this for all machines defined in the config.

convert_neware_data converts a neware file to a dataframe and optionally saves it as a gzipped json
file. This file contains all cycling data as well as metadata and information about the sample from
the database.

convert_all_neware_data does this for all files in the local snapshot folder, and saves them to the
processed snapshot folder.

Run the script to harvest and convert all neware files.
"""
import os
import sys
import json
import sqlite3
import warnings
import gzip
import re
import paramiko
import pandas as pd
from datetime import datetime
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.analysis import _run_from_sample
from aurora_cycler_manager.version import __version__, __url__

# Load configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config.json')
with open(config_path, encoding = 'utf-8') as f:
    config = json.load(f)
neware_config = config.get("Neware harvester", {})
db_path = config["Database path"]

def harvest_neware_files(
    server_label: str,
    server_hostname: str,
    server_username: str,
    server_shell_type: str,
    server_copy_folder: str,
    local_folder: str,
    local_private_key_path: str = None,
    force_copy: bool = False,
) -> None:
    """ Get Neware files from subfolders of specified folder.
    
    Args:
        server_label (str): Label of the server
        server_hostname (str): Hostname of the server
        server_username (str): Username to login with
        server_shell_type (str): Type of shell to use (powershell or cmd)
        server_copy_folder (str): Folder to search and copy TODO file types
        local_folder (str): Folder to copy files to
        local_private_key (str, optional): Local private key path for ssh
        force_copy (bool): Copy all files regardless of modification date
    """
    if force_copy:  # Set cutoff date to 1970
        cutoff_datetime = datetime.fromtimestamp(0)
    else:  # Set cutoff date to last snapshot from database
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT `Last snapshot` FROM harvester WHERE "
                f"`Server label`='{server_label}' "
                f"AND `Server hostname`='{server_hostname}' "
                f"AND `Folder`='{server_copy_folder}'"
            )
            result = cursor.fetchone()
            cursor.close()
        if result:
            cutoff_datetime = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
        else:
            cutoff_datetime = datetime.fromtimestamp(0)
    cutoff_date_str = cutoff_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # Connect to the server and copy the files
    with paramiko.SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"Connecting to host {server_hostname} user {server_username}")
        ssh.connect(server_hostname, username=server_username, key_filename=local_private_key_path)

        # Shell commands to find files modified since cutoff date
        # TODO need to grab all the filenames and modified dates, copy if they are newer than local files not just cutoff date
        if server_shell_type == "powershell":
            command = (
                f'Get-ChildItem -Path \'{server_copy_folder}\' -Recurse '
                f'| Where-Object {{ $_.LastWriteTime -gt \'{cutoff_date_str}\' -and ($_.Extension -eq \'.xlsx\' -or $_.Extension -eq \'.csv\')}} '
                f'| Select-Object -ExpandProperty FullName'
            )
        elif server_shell_type == "cmd":
            command = (
                f'powershell.exe -Command "Get-ChildItem -Path \'{server_copy_folder}\' -Recurse '
                f'| Where-Object {{ $_.LastWriteTime -gt \'{cutoff_date_str}\' -and ($_.Extension -eq \'.xlsx\' -or $_.Extension -eq \'.csv\')}} '
                f'| Select-Object -ExpandProperty FullName"'
            )
        stdin, stdout, stderr = ssh.exec_command(command)

        # Parse the output
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()
        assert not stderr.read(), f"Error finding modified files: {stderr.read()}"
        modified_files = output.splitlines()
        print(f"Found {len(modified_files)} files modified since {cutoff_date_str}")

        # Copy the files using SFTP
        current_datetime = datetime.now()  # Keep time of copying for database
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

    # Update the database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO harvester (`Server label`, `Server hostname`, `Folder`) "
            "VALUES (?, ?, ?)",
            (server_label, server_hostname, server_copy_folder)
        )
        cursor.execute(
            "UPDATE harvester "
            "SET `Last snapshot` = ? "
            "WHERE `Server label` = ? AND `Server hostname` = ? AND `Folder` = ?",
            (current_datetime.strftime('%Y-%m-%d %H:%M:%S'), server_label, server_hostname, server_copy_folder)
        )
        cursor.close()

def harvest_all_neware_files(force_copy = False) -> None:
    """ Get neware files from all servers specified in the config. """
    for server in neware_config["Servers"]:
        harvest_neware_files(
            server_label = server["label"],
            server_hostname = server["hostname"],
            server_username = server["username"],
            server_shell_type = server["shell_type"],
            server_copy_folder = server["Neware folder location"],
            local_folder = neware_config["Snapshots folder path"],
            local_private_key_path = config["SSH private key path"],
            force_copy = force_copy
        )

def get_neware_metadata(file_path: str) -> dict:
    # Get the test info, including barcode / remarks
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    df = pd.read_excel(file_path, sheet_name="test", header=None)

    # In first column, find index where value is "Test information" and "Step plan"
    test_idx = df[df.iloc[:, 0] == "Test information"].index[0]
    step_idx = df[df.iloc[:, 0] == "Step plan"].index[0]

    # Get test info, remove empty columns and rows
    test_settings = df.iloc[test_idx+1:step_idx,:]
    test_settings = test_settings.dropna(axis=1, how="all")
    test_settings = test_settings.dropna(axis=0, how="all")

    # Flatten and convert to dict
    flattened = test_settings.values.flatten().tolist()
    flattened = [str(x) for x in flattened if str(x) != 'nan']
    test_info = {flattened[i]: flattened[i+1] for i in range(0, len(flattened), 2) if flattened[i] and flattened[i] != '-'}
    test_info = {k: v for k, v in test_info.items() if (k and k != '-' and k != 'nan') or (v and v != '-' and v != 'nan')}

    # Payload
    payload = df.iloc[step_idx+2:, :]
    payload.columns = df.iloc[step_idx+1]
    payload_dict = payload.to_dict(orient="records")

    payload_dict = [{k: v for k, v in record.items() if str(v) != 'nan'} for record in payload_dict]

    # In Neware step information, 'Cycle' steps have different columns defined within the row
    # E.g. the "Voltage (V)" column has a value like "Cycle count:2"
    # We find these entires, and rename the key e.g. "Voltage (V)": "Cycle count:2" becomes "Cycle count": 2
    for record in payload_dict:
        if record.get("Step Name") == "Cycle":
            # find values with ":" in them, and split them into key value pairs, delete the original key
            bad_key_vals = {k: v for k, v in record.items() if ":" in str(v)}
            for k, v in bad_key_vals.items():
                del record[k]
                new_k, new_v = v.split(":")
                record[new_k] = new_v

    # Add to test_info
    test_info["Payload"] = payload_dict

    # Get sampleid from test_info
    sampleid = test_info.get("Barcode", None)
    if not sampleid:
        sampleid = test_info.get("Remarks", None)
    
    # Check against known samples
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT `Sample ID` FROM samples")
        rows = cursor.fetchall()
        cursor.close()
    known_samples = [row[0] for row in rows]

    # If sampleid is not known, it might be using the convention 'date-number-other'
    if sampleid not in known_samples:
        # Extract date and number
        sampleid_parts = re.split('_|-', sampleid)
        if len(sampleid_parts) > 1:
            sampleid_date = sampleid_parts[0]
            sampleid_number = sampleid_parts[1].zfill(2) # pad with zeros
        # Check if this is consistent with any known samples
        possible_samples = [s for s in known_samples if s.startswith(sampleid_date) and s.endswith(sampleid_number)]
        if len(possible_samples) == 1:
            print(f"Barcode {sampleid} inferred as Sample ID {possible_samples[0]}")
            sampleid = possible_samples[0]
        else:
            print(f"Sample ID {sampleid} not found in database")
            sampleid = None

    return test_info, sampleid

def get_neware_data(file_path: str) -> dict:
    df = pd.read_excel(file_path, sheet_name="record", header=0)
    output_df = pd.DataFrame()
    output_df["V (V)"] = df["Voltage(V)"]
    output_df["I (A)"] = df["Current(A)"]
    output_df["technique"] = df["Step Type"]
    output_df["loop_number"] = 0
    output_df["cycle_number"] = df["Cycle Index"]
    output_df["index"] = 0
    # convert date string from df["Date"] in format YYYY-MM-DD HH:MM:SS to uts timestamp in seconds
    output_df["uts"] = df["Date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    return output_df.to_dict(orient='list')

def convert_neware_data(
        file_path: str,
        output_jsongz_file: bool = True,
) -> tuple[dict, dict]:
    """ Convert a neware file to a dataframe and save as a gzipped json file.

    Args:
        file_path (str): Path to the neware file
        sampleid (str): Sample ID
        output_jsongz_file (bool): Whether to save the file as a gzipped json
    """

    # Get test information and Sample ID    
    test_info, sampleid = get_neware_metadata(file_path)

    # Get data
    data = get_neware_data(file_path)

    # If there is a valid Sample ID, get sample metadata from database
    sample_data = None
    if sampleid:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM samples WHERE `Sample ID`='{sampleid}'")
            row = cursor.fetchone()
            if row:
                columns = [column[0] for column in cursor.description]
                sample_data = dict(zip(columns, row))

    # Metadata to add
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metadata = {
        "provenance": {
            "snapshot_file": file_path,
            "aurora_metadata": {
                "mpr_conversion" : {
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "neware_harvester.convert_neware_data",
                    "datetime": current_datetime,
                },
            }
        },
        "neware_metadata": test_info,
        "sample_data": sample_data,
    }

    if output_jsongz_file:
        if not sampleid:
            print(f"Not saving {file_path}, no valid Sample ID found")
            return data, metadata
        folder = os.path.join(config["Processed snapshots folder path"], _run_from_sample(sampleid),sampleid)
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_jsongz_file = os.path.join(folder, "snapshot."+os.path.basename(file_path).replace(".xlsx", ".json.gz"))
        with gzip.open(output_jsongz_file, 'wt') as f:
            json.dump({'data': data, 'metadata': metadata}, f)

        # Update the database
        creation_date = datetime.fromtimestamp(
            os.path.getmtime(file_path)
        ).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)",
                (sampleid,)
            )
            cursor.execute(
                "UPDATE results "
                "SET `Last snapshot` = ? "
                "WHERE `Sample ID` = ?",
                (creation_date, sampleid)
            )
            cursor.close()

    return data, metadata

def convert_all_neware_data() -> None:
    """ Converts all neware files to gzipped json files. 
    
    The config file needs a key "Neware harvester" with the keys "Snapshots folder path"
    """
    raw_folder = neware_config["Snapshots folder path"]

    # Get all xlsx files in the raw folder recursively
    neware_files = []
    for root, _, files in os.walk(raw_folder):
        for file in files:
            if file.endswith(".xlsx"):
                neware_files.append(os.path.join(root, file))
    for file in neware_files:
        convert_neware_data(file)

if __name__ == "__main__":
    harvest_all_neware_files()
    convert_all_neware_data()
