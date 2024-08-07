""" Grab EC-lab files from remote PCs and save them as a cucumber-compatible hdf5 file. 

Define the machines to grab files from in the config dictionary.
The functions will grab all files from specified folders on a remote machine,
save them locally, then convert them to hdf5 files and save in a processed
data folder cucumber-style, i.e. with run-id/sample-id/snapshot-id.h5.

These files can be processed with the same tools in cucumber_analysis as used
for the data from tomato.
"""
import os
import re
import json
import sqlite3
from datetime import datetime
import pytz
import paramiko
from scp import SCPClient
import numpy as np
import pandas as pd
import h5py
import yadg
import yadg.extractors
from version import __version__

eclab_config = {
    "Servers": [
        {
            "label" : "tt1", # must match the label in the main config file
            "EC-lab folder location": "C:/Users/lab131/Desktop/eclab/svfe/",
            "EC-lab copy location": "C:/Users/lab131/eclabcopy/"
        },
        {
            "label" : "tt2", # must match the label in the main config file
            "EC-lab folder location": "C:/Users/lab131/Desktop/EC-lab/",
            "EC-lab copy location": "C:/Users/lab131/eclabcopy/"
        }
    ],
    "Snapshots folder path": "C:/", # will add 'eclabcopy' folder to this path
    "Processed snapshots folder path": "K:/Aurora/cucumber/ec-lab snapshots/"
}

def get_mprs(
    server_hostname: str,
    server_username: str,
    local_private_key: str,
    server_eclab_folder: str,
    server_copy_folder: str,
    local_folder: str,
) -> None:
    """ Get all MPR files from subfolders of specified folder.
    
    Args:
        server_hostname (str): Hostname of the server
        server_username (str): Username to login with
        server_private_key (str): Private key for ssh
        folder (str): Folder to search for MPR files
    """
    with paramiko.SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f"Connecting to host {server_hostname} user {server_username}")
        ssh.connect(server_hostname, username=server_username, pkey=local_private_key)
        _, _, stderr = ssh.exec_command(
            f"robocopy {server_eclab_folder} {server_copy_folder} "
            "/E /Z /NP /NC /NS /NFL /NDL /NJH /NJS"
            )
        assert not stderr.read(), f"Error copying folder: {stderr.read()}"
        with SCPClient(ssh.get_transport(), socket_timeout=120) as scp:
            scp.get(server_copy_folder, recursive=True, local_path=local_folder)

def get_mprs_from_folders() -> None:
    """ Get all mpr files from all servers using the config file """
    with open("./config.json", "r", encoding = 'utf-8') as f:
        config = json.load(f)
    for server in eclab_config["Servers"]:
        server_config = next((s for s in config["Servers"] if s["label"] == server["label"]), None)
        get_mprs(
            server_config["hostname"],
            server_config["username"],
            paramiko.RSAKey.from_private_key_file(config["SSH private key path"]),
            server["EC-lab folder location"],
            server["EC-lab copy location"],
            eclab_config["Snapshots folder path"],
        )

def convert_mpr_to_hdf(
        sampleid: str,
        mpr_file: str,
        output_hdf_file: str = None,
        capacity_Ah: float = None,
        ) -> pd.DataFrame:
    """ Convert a tomato json to dataframe, optionally save as hdf5 file.

    Args:
        sampleid (str): sample ID from robot output
        snapshot_file (str): path to the raw mpr file
        hdf_save_location (str, optional): path to save the output hdf5 file
        capacity_Ah (float, optional): capacity of the cell in Ah

    Returns:
        pd.DataFrame: DataFrame containing the cycling data

    Columns in output DataFrame:
    - uts: timestamp, starting from 0
    - V (V): Cell voltage in volts
    - I (A): Current in amps
    - loop_number: how many loops have been completed
    - cycle_number: not used here
    - index: index of the method in the payload, not used here
    - technique: code of technique using Biologic convention, not used here
    
    TODO: use capacity to define C rate

    """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    db_path = config["Database path"]

    creation_date = datetime.fromtimestamp(
        os.path.getmtime(mpr_file)
    ).strftime('%Y-%m-%d %H:%M:%S')

    data = yadg.extractors.extract('eclab.mpr', mpr_file)

    df = pd.DataFrame()
    df['uts'] = data.coords['uts'].values
    # Check if the time is incorrect and fix it
    if df['uts'].values[0] < 1000000000: # the measurement started before 2001
        # grab the start time from mpl file
        mpl_file = mpr_file.replace(".mpr",".mpl")
        try:
            with open(mpl_file, encoding='ANSI') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Incorrect start time in {mpr_file} and no mpl file found.")
        # find the date from line with Acquisition started on : 07/17/2024 11:36:40.528
        for line in lines:
            if line.startswith("Acquisition started on : "):
                datetime_str = line.split(":",1)[1].strip()
                datetime_object = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S.%f')
                timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
                uts_timestamp = timezone.localize(datetime_object).timestamp()
                df['uts'] = df['uts'] + uts_timestamp

    df['V (V)'] = data.data_vars['Ewe'].values
    df['I (A)'] = (
        (3600 / 1000) * data.data_vars['dq'].values /
        np.diff(data.coords['uts'].values,prepend=[np.inf])
    )
    df['loop_number'] = data.data_vars['half cycle'].values//2
    df['cycle_number'] = 0
    df['index'] = 0
    df['technique'] = data.data_vars['mode'].values

    if output_hdf_file:
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

        # Try get sample data from database
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
        mpr_metadata = data.attrs['original_metadata']
        yadg_metadata = json.dumps(
            {k: v for k, v in data.attrs.items() if k.startswith('yadg')}
        )

        # Metadata to add
        metadata = {
            "provenance": {
                "snapshot_file": mpr_file,
                "yadg_metadata": yadg_metadata,
                "cucumber_metadata": {
                    "cucumber_version": __version__,
                    "conversion_method": "cucumber_eclab_harvester.py convert_mpr_to_hdf",
                    "conversion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
            },
            "mpr_metadata": mpr_metadata,
            "sample_data": json.dumps(sample_data) if sample_data is not None else "",
        }
        # Open the HDF5 file with h5py and add metadata
        with h5py.File(output_hdf_file, 'a') as file:
            if 'cycling' in file:
                for key, value in metadata.items():
                    if value is None:
                        value=""
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    file['cycling'].attrs[key] = value
            else:
                print("Dataset 'cycling' not found.")
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
    """ Converts all raw eclab files to hdf5 files. """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    raw_folder = os.path.join(eclab_config["Snapshots folder path"], "eclabcopy")
    processed_folder = config["Processed snapshots folder path"]
    db_path = config["Database path"]

    # HACK - lookup table for run IDs from incorrectly named folders
    # it is not possible to change folder names while data is being collected
    run_id_lookup = {
        "270624_gen5_2_FEC_C3": "240620_svfe_gen5",
        "010724_gen6_FEC_1C": "240701_svfe_gen6",
        "090724_Gen7": "240709_svfe_gen7",
        "100724_Gen8": "240709_svfe_gen8",
    }
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
                # convert the mpr to hdf
                mpr_path = os.path.join(raw_folder, run_folder, mpr)
                output_path = os.path.join(processed_folder, run_id, sample_id, f"snapshot.mpr-{i}.h5")
                try:
                    convert_mpr_to_hdf(sample_id, mpr_path, output_path, 0.00154)
                except Exception as e:
                    print(f"Error processing {mpr}: {e}")

        # Check for subfolders with .mpr
        for sample_folder in os.listdir(os.path.join(raw_folder, run_folder)):
            if not os.path.isdir(os.path.join(raw_folder, run_folder, sample_folder)):
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

            # convert the mpr to hdf
            mprs = [f for f in os.listdir(os.path.join(raw_folder, run_folder, sample_folder)) if f.endswith('.mpr')]
            if not mprs:
                print(f"No mpr files found for {sample_id}")
                continue
            for i,mpr in enumerate(mprs):
                mpr_path = os.path.join(raw_folder, run_folder, sample_folder, mpr)
                output_path = os.path.join(processed_folder, run_id, sample_id, f"snapshot.mpr-{i}.h5")
                try:
                    convert_mpr_to_hdf(sample_id, mpr_path, output_path, 0.00154)
                except Exception as e:
                    print(f"Error processing {mpr}: {e}")

if __name__ == "__main__":
    get_mprs_from_folders()
    convert_all_mprs()
