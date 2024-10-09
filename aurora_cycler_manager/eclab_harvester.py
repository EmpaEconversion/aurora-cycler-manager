""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Harvest EC-lab .mpr files and convert to aurora-compatible gzipped json files. 

Define the machines to grab files from in the config.json file.

get_mpr will copy all files from specified folders on a remote machine, if they
have been modified since the last time the function was called.

get_all_mprs does this for all machines defined in the config.

convert_mpr converts an mpr to a dataframe and optionally saves it as a hdf5
file and/or a gzipped json file. This file contains all cycling data as well as
metadata from the mpr and information about the sample from the database.

convert_all_mprs does this for all mpr files in the local snapshot folder, and
saves them to the processed snapshot folder.

Run the script to harvest and convert all mpr files.
"""
import os
import sys
import re
import json
import gzip
import sqlite3
import warnings
from datetime import datetime
import pytz
import paramiko
from scp import SCPClient
import numpy as np
import pandas as pd
import h5py
import yadg
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.version import __version__, __url__

# Load configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config.json')
with open(config_path, encoding = 'utf-8') as f:
    config = json.load(f)
eclab_config = config["EC-lab harvester"]
db_path = config["Database path"]

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
    """ Get .mpr files from subfolders of specified folder.
    
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
        ssh.connect(server_hostname, username=server_username, pkey=local_private_key)

        # Shell commands to find files modified since cutoff date
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

def get_all_mprs(force_copy=False) -> None:
    """ Get all MPR files from the folders specified in the config.
    
    The config file needs a key "EC-lab harvester" with a key "Snapshots folder 
    path" with a location to save to, and a key "Servers" containing a list of 
    dictionaries with the keys "label" and "EC-lab folder location".
    The "label" must match a server in the "Servers" list in the main config.
    """
    for server in eclab_config["Servers"]:
        server_config = next((s for s in config["Servers"] if s["label"] == server["label"]), None)
        get_mprs(
            server["label"],
            server_config["hostname"],
            server_config["username"],
            server_config["shell_type"],
            server["EC-lab folder location"],
            paramiko.RSAKey.from_private_key_file(config["SSH private key path"]),
            eclab_config["Snapshots folder path"],
            force_copy,
        )

def convert_mpr(
        sampleid: str,
        mpr_file: str,
        output_hdf_file: str = None,
        output_jsongz_file: str = None,
        capacity_Ah: float = None,
        ) -> pd.DataFrame:
    """ Convert a tomato json to dataframe, optionally save as hdf5 or zipped json file.

    Args:
        sampleid (str): sample ID from robot output
        mpr_file (str): path to the raw mpr file
        output_hdf_file (str, optional): path to save the output hdf5 file
        output_jsongz_file (str, optional): path to save the output zipped json file
        capacity_Ah (float, optional): capacity of the cell in Ah

    Returns:
        pd.DataFrame: DataFrame containing the cycling data

    Columns in output DataFrame:
    - uts: unix timestamp in seconds
    - V (V): Cell voltage in volts
    - I (A): Current in amps
    - loop_number: number of loops over several techniques
    - cycle_number: number of cycles within one technique
    - index: index of the method in the payload, not used here
    - technique: code of technique using Biologic convention, not used here
    
    TODO: use capacity to define C rate

    """

    # Normalize paths to avoid escape character issues
    mpr_file = os.path.normpath(mpr_file)
    output_hdf_file = os.path.normpath(output_hdf_file) if output_hdf_file else None
    output_jsongz_file = os.path.normpath(output_jsongz_file) if output_jsongz_file else None

    creation_date = datetime.fromtimestamp(
        os.path.getmtime(mpr_file)
    ).strftime('%Y-%m-%d %H:%M:%S')

    # Extract data from mpr file
    data = yadg.extractors.extract('eclab.mpr', mpr_file)

    df = pd.DataFrame()
    df['uts'] = data.coords['uts'].values

    # Check if the time is incorrect and fix it
    if df['uts'].values[0] < 1000000000:  # The measurement started before 2001, assume wrong
        # Grab the start time from mpl file
        mpl_file = mpr_file.replace(".mpr",".mpl")
        try:
            with open(mpl_file, encoding='ANSI') as f:
                lines = f.readlines()
            for line in lines:
                # Find the start datetime from the mpl
                found_start_time = False
                if line.startswith("Acquisition started on : "):
                    datetime_str = line.split(":",1)[1].strip()
                    datetime_object = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S.%f')
                    timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
                    uts_timestamp = timezone.localize(datetime_object).timestamp()
                    df['uts'] = df['uts'] + uts_timestamp
                    found_start_time = True
                    break
            if not found_start_time:
                warnings.warn(f"Incorrect start time in {mpr_file} and no start time in found {mpl_file}")
        except FileNotFoundError:
            warnings.warn(f"Incorrect start time in {mpr_file} and no mpl file found.")

    # Only keep certain columns in dataframe
    df['V (V)'] = data.data_vars['Ewe'].values
    df['I (A)'] = (
        (3600 / 1000) * data.data_vars['dq'].values /
        np.diff(data.coords['uts'].values,prepend=[np.inf])
    )
    df['loop_number'] = data.data_vars['half cycle'].values//2
    df['cycle_number'] = 0
    df['index'] = 0
    df['technique'] = data.data_vars['mode'].values

    # If saving files, add metadata, save, update database
    if output_hdf_file or output_jsongz_file:
        # get sample data from database
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM samples WHERE `Sample ID`='{sampleid}'")
                row = cursor.fetchone()
                columns = [column[0] for column in cursor.description]
                sample_data = dict(zip(columns, row))
        except Exception as e:
            print(f"Error getting job and sample data from database: {e}")
            sample_data = None

        # Get job data from the snapshot file
        mpr_metadata = json.loads(data.attrs['original_metadata'])
        yadg_metadata = {k: v for k, v in data.attrs.items() if k.startswith('yadg')}

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
                        "datetime": datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %z'),
                    },
                }
            },
            "mpr_metadata": mpr_metadata,
            "sample_data": sample_data if sample_data is not None else {},
        }

        if output_hdf_file:  # Save as hdf5
            folder = os.path.dirname(output_hdf_file)
            if not folder:
                folder = '.'
            if not os.path.exists(folder):
                os.makedirs(folder)
            df.to_hdf(
                output_hdf_file,
                key="cycling",
                complib="blosc",
                complevel=2
            )
            # Open the HDF5 file with h5py and add metadata
            with h5py.File(output_hdf_file, 'a') as file:
                if 'cycling' in file:
                    file['cycling'].attrs['metadata'] = json.dumps(metadata)
                else:
                    print("Dataset 'cycling' not found.")

        if output_jsongz_file:  # Save as zipped json
            folder = os.path.dirname(output_jsongz_file)
            if not folder:
                folder = '.'
            if not os.path.exists(folder):
                os.makedirs(folder)
            with gzip.open(output_jsongz_file, 'wt') as f:
                json.dump({'data': df.to_dict(orient='list'), 'metadata': metadata}, f)

        # Update the database
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
    return df

def convert_all_mprs() -> None:
    """ Converts all raw .mpr files to gzipped json files. 
    
    The config file needs a key "EC lab harvester" with the keys "Snapshots folder path",
     and "Run ID lookup" containing a dictionary with
    """
    raw_folder = eclab_config["Snapshots folder path"]
    processed_folder = config["Processed snapshots folder path"]
    db_path = config["Database path"]

    # Lookup dict for folder_name: run_id
    # In case folders are named differently to run_id on the server
    run_id_lookup = eclab_config.get("Run ID lookup", {})

    for run_folder in os.listdir(raw_folder):
        print("Processing folder", run_folder)
        if not os.path.isdir(os.path.join(raw_folder, run_folder)):
            print(f"Skipping {run_folder} - not a folder")
            continue
        run_id = run_id_lookup.get(run_folder, None)

        # Try looking up in the database if not found in the lookup table
        if not run_id:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT `Run ID` FROM samples WHERE `Sample ID` LIKE '%{run_folder}%'")
                result = cursor.fetchone()
                cursor.close()
            if result:
                run_id = result[0]
                print(f"Found Run ID '{run_id}' for {run_folder}")
            else:
                print(f"Skipping {run_folder} - could not find a Run ID")
                continue
        print(f"Found Run ID '{run_id}'")

        # Check for loose .mpr in folder
        mprs = [f for f in os.listdir(os.path.join(raw_folder, run_folder)) if f.endswith('.mpr')]
        if mprs:
            for i,mpr in enumerate(mprs):
                mpr_path = os.path.join(raw_folder, run_folder, mpr)
                match = re.search(r'cell(\d+)[_(]?', mpr)
                if match:
                    sample_number = int(match.group(1))
                else:
                    print(f"Skipping {mpr} - could not recognise a sample number")
                    continue
                sample_id = f"{run_id}_{sample_number:02d}"
                print(f"Processing {sample_id}")

                # Convert the mpr to hdf
                mpr_path = os.path.join(raw_folder, run_folder, mpr)
                output_jsongz = os.path.join(processed_folder, run_id, sample_id, f"snapshot.mpr-{i}.json.gz")
                try:
                    convert_mpr(sample_id, mpr_path, None, output_jsongz, 0.00154)
                except Exception as e:
                    print(f"Error processing {mpr}: {e}")

        # Check for sample folders
        for sample_folder in os.listdir(os.path.join(raw_folder, run_folder)):
            root_folder = os.path.join(raw_folder, run_folder, sample_folder)
            if not os.path.isdir(root_folder):
                continue
            try:
                sample_number = int(sample_folder)
            except ValueError:
                match = re.search(r'cell(\d+)[_(]?', sample_folder)
                if match:
                    sample_number = int(match.group(1))
                else:
                    print(f"Skipping {sample_folder} - could not recognise a sample number")
                    continue
            sample_id = f"{run_id}_{sample_number:02d}"
            print(f"Processing {sample_id}")

            # Walk through the sample folder, find all .mpr files including in subfolders
            mprs = []
            for dirpath, dirnames, filenames in os.walk(root_folder):
                for filename in filenames:
                    if filename.endswith(".mpr"):
                        mprs.append(os.path.relpath(os.path.join(dirpath, filename), root_folder))
            if not mprs:
                print(f"No mpr files found for {sample_id}")
                continue

            # Convert the mpr to hdf
            for i,mpr in enumerate(mprs):
                mpr_path = os.path.join(root_folder, mpr)
                output_jsongz = os.path.join(processed_folder, run_id, sample_id, f"snapshot.mpr-{i}.json.gz")
                try:
                    convert_mpr(sample_id, mpr_path, None, output_jsongz, 0.00154)
                except Exception as e:
                    print(f"Error processing {mpr}: {e}")

if __name__ == "__main__":
    get_all_mprs()
    convert_all_mprs()
