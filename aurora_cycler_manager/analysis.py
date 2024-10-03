""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Functions used for parsing, analysing and plotting.

Parsing:
Contains functions for converting raw jsons from tomato to pandas dataframes,
which can be saved to compressed hdf5 files.

Also includes functions for analysing the cycling data, extracting the
charge, discharge and efficiency of each cycle, and links this to various
quantities extracted from the cycling, such as C-rate and max voltage, and
from the sample database such as cathode active material mass.
"""
import os
import re
import sqlite3
from typing import List, Tuple, Literal
from datetime import datetime
import traceback
import json
import gzip
import fractions
import pytz
import yaml
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from aurora_cycler_manager.version import __version__, __url__

def convert_tomato_json(
        snapshot_file: str,
        output_hdf_file: str = None,
        output_jsongz_file: str = None,
        ) -> pd.DataFrame:
    """ Convert a raw json file from tomato to a pandas dataframe.

    Args:
        snapshot_file (str): path to the raw json file
        hdf_save_location (str, optional): path to save the output hdf5 file

    Returns:
        pd.DataFrame: DataFrame containing the cycling data

    Columns in output DataFrame:
    - uts: Unix time stamp in seconds
    - V (V): Cell voltage in volts
    - I (A): Current in amps
    - loop_number: how many loops have been completed
    - cycle_number: used if there is a loop of loops
    - index: index of the method in the payload
    - technique: code of technique using Biologic convention
        100 = OCV, 101 = CA, 102 = CP, 103 = CV, 155 = CPLIMIT, 157 = CALIMIT, 
        -1 = Unknown

    The dataframe is saved to 'cycling' key in the hdf5 file.
    Metadata is added to the 'cycling' attributes in hdf5 file.
    The metadata crucially includes json dumps of the job data and sample data
    extracted from the database.
    """
    with open(snapshot_file, "r", encoding="utf-8") as f:
        input_dict = json.load(f)
    n_steps = len(input_dict["steps"])
    data = []
    technique_code = {"NONE":0,"OCV":100,"CA":101,"CP":102,"CV":103,"CPLIMIT":155,"CALIMIT":157}
    for i in range(n_steps):
        step_data = input_dict["steps"][i]["data"]
        step_dict = {
            "uts" : [row["uts"] for row in step_data],
            "V (V)" : [row["raw"]["Ewe"]["n"] for row in step_data],
            "I (A)": [row["raw"]["I"]["n"] if "I" in row["raw"] else 0 for row in step_data],
            "cycle_number": [row["raw"]["cycle number"] if "cycle number" in row["raw"] else 0 for row in step_data],
            "loop_number": [row["raw"]["loop number"] if "loop number" in row["raw"] else 0 for row in step_data],
            "index" : [row["raw"]["index"] if "index" in row["raw"] else -1 for row in step_data],
            "technique" : [technique_code.get(row["raw"]["technique"], -1) if "technique" in row["raw"] else -1 for row in step_data],
        }
        data.append(pd.DataFrame(step_dict))
    data = pd.concat(data, ignore_index=True)
    if output_hdf_file or output_jsongz_file:
        # Try to get the job number from the snapshot file and add to metadata
        try:
            json_filename = os.path.basename(snapshot_file)
            jobid = "".join(json_filename.split(".")[1:-1])
            # look up jobid in the database
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', 'config.json')
            with open(config_path, encoding = 'utf-8') as f:
                config = json.load(f)
            db_path = config["Database path"]
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Get all data about this job
                cursor.execute(f"SELECT * FROM jobs WHERE `Job ID`='{jobid}'")
                job_data = dict(cursor.fetchone())
                job_data['Payload'] = json.loads(job_data['Payload'])
                sampleid = job_data["Sample ID"]
                # Get all data about this sample
                cursor.execute(f"SELECT * FROM samples WHERE `Sample ID`='{sampleid}'")
                sample_data = dict(cursor.fetchone())
        except Exception as e:
            print(f"Error getting job and sample data from database: {e}")
            job_data = None
            sample_data = None

        # Create metadata
        metadata = {
            "provenance": {
                "snapshot_file": snapshot_file,
                "tomato_metadata": input_dict["metadata"],
                "aurora_metadata": {
                    "json_conversion": {
                        "repo_url": __url__,
                        "repo_version": __version__,
                        "method": "analysis.py convert_tomato_json",
                        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    
                },
            },
            "job_data": job_data if job_data is not None else {},
            "sample_data": sample_data if sample_data is not None else {},
            "glossary": {
                "uts": "Unix time stamp in seconds",
                "V (V)": "Cell voltage in volts",
                "I (A)": "Current across cell in amps",
                "loop_number": "Number of loops completed from EC-lab loop technique",
                "cycle_number": "Number of cycles within one technique from EC-lab",
                "index": "index of the method in the payload, i.e. 0 for the first method, 1 for the second etc.",
                "technique": "code of technique using definitions from MPG2 developer package",
            },
        }
        if output_hdf_file:
            folder = os.path.dirname(output_hdf_file)
            if not folder:
                folder = '.'
            if not os.path.exists(folder):
                os.makedirs(folder)
            data.to_hdf(
                output_hdf_file,
                key="cycling",
                complib="blosc",
                complevel=2
            )
            with h5py.File(output_hdf_file, 'a') as file:
                if 'cycling' in file:
                    file['cycling'].attrs['metadata'] = json.dumps(metadata)
                else:
                    print("Dataset 'cycling' not found.")
        if output_jsongz_file:
            folder = os.path.dirname(output_jsongz_file)
            if not folder:
                folder = '.'
            if not os.path.exists(folder):
                os.makedirs(folder)
            full_data = {'data': data.to_dict(orient='list'), 'metadata': metadata}
            with gzip.open(output_jsongz_file, 'wt', encoding='utf-8') as f:
                json.dump(full_data, f)

    return data

def convert_all_tomato_jsons(
    sampleid_contains: str = ""
    ) -> None:
    """ Goes through all the raw json files in the snapshots folder and converts them to hdf5. """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    raw_folder = config["Snapshots folder path"]
    processed_folder = config["Processed snapshots folder path"]
    for batch_folder in os.listdir(raw_folder):
        for sample_folder in os.listdir(os.path.join(raw_folder, batch_folder)):
            if sampleid_contains and sampleid_contains not in sample_folder:
                continue
            for snapshot_file in os.listdir(os.path.join(raw_folder, batch_folder, sample_folder)):
                if snapshot_file.startswith('snapshot') and snapshot_file.endswith('.json'):
                    output_folder = os.path.join(processed_folder,batch_folder,sample_folder)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    convert_tomato_json(
                        os.path.join(raw_folder,batch_folder,sample_folder,snapshot_file),
                        None,
                        os.path.join(output_folder,snapshot_file.replace('.json','.json.gz')),
                    )
                    print(f"Converted {snapshot_file}")

def combine_jobs(
        job_files: List[str],
) -> Tuple[pd.DataFrame, dict]:
    """ Read multiple job files and return a single dataframe.

    Merges the data, identifies cycle numbers and changes column names.
    Columns are now 'V (V)', 'I (A)', 'uts', 'dt (s)', 'Iavg (A)', 
    'dQ (mAh)', 'Step', 'Cycle'.

    Args:
        job_files (List[str]): list of paths to the job files

    Returns:
        pd.DataFrame: DataFrame containing the cycling data
        dict: metadata from the files
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    # Get the metadata from the files
    dfs = []
    metadatas = []
    sampleids = []
    for f in job_files:
        if f.endswith('.h5'):
            dfs.append(pd.read_hdf(f))
            with h5py.File(f, 'r') as file:
                try:
                    metadata=json.loads(file['cycling'].attrs['metadata'])
                    metadatas.append(metadata)
                    sampleids.append(
                        metadata.get('sample_data',{}).get('Sample ID','')
                    )
                except KeyError as exc:
                    print(f"Metadata not found in {f}")
                    raise KeyError from exc
        elif f.endswith('.json.gz'):
            with gzip.open(f, 'rt', encoding='utf-8') as file:
                data = json.load(file)
                dfs.append(pd.DataFrame(data['data']))
                metadatas.append(data['metadata'])
                sampleids.append(
                    data['metadata'].get('sample_data',{}).get('Sample ID','')
                )
    assert len(set(sampleids)) == 1, "All files must be from the same sample"
    dfs = [df for df in dfs if 'uts' in df.columns and not df['uts'].empty]
    assert dfs, "No 'uts' column found in any of the files"
    order = np.argsort([df['uts'].iloc[0] for df in dfs])
    dfs = [dfs[i] for i in order]
    job_files = [job_files[i] for i in order]
    metadatas = [metadatas[i] for i in order]

    for i,df in enumerate(dfs):
        df["job_number"] = i
    df = pd.concat(dfs)
    df = df.sort_values('uts')
    # rename columns
    df = df.rename(columns={
        'Ewe': 'V (V)',
        'I': 'I (A)',
        'uts': 'uts',
    })
    df["dt (s)"] = np.concatenate([[0],df["uts"].values[1:] - df["uts"].values[:-1]])
    df["Iavg (A)"] = np.concatenate([[0],(df["I (A)"].values[1:] + df["I (A)"].values[:-1]) / 2])
    df["dQ (mAh)"] = 1e3 * df["Iavg (A)"] * df["dt (s)"] / 3600
    df.loc[df["dt (s)"] > 600, "dQ (mAh)"] = 0

    df['group_id'] = (
        (df['loop_number'].shift(-1) < df['loop_number']) |
        (df['cycle_number'].shift(-1) < df['cycle_number']) |
        (df['job_number'].shift(-1) < df['job_number'])
    ).cumsum()
    df['Step'] = df.groupby(['job_number','group_id','cycle_number','loop_number']).ngroup()
    df.drop(columns=['job_number', 'group_id', 'cycle_number', 'loop_number','index'], inplace=True)
    df['Cycle']=0
    cycle=1
    for step, group_df in df.groupby('Step'):
        # To be considered a cycle (subject to change):
        # - more than 10 data points
        # - 99th percentile min and max voltage of charge and discharge within 2 V of each other
        # - total change in charge less than 50% of absolute total charge
        # - e.g. 1 mAh charge and 0.3 mAh discharge gives 0.7 mAh change and 1.3 mAh total = 54%
        #        this would not be considered a cycle
        # - e.g. 1 mAh charge and 0.5 mAh discharge gives 0.5 mAh change and 1.5 mAh total = 33%
        #        this would be considered a cycle
        if len(group_df) > 10:
            if abs(group_df["dQ (mAh)"].sum()) < 0.5 * group_df["dQ (mAh)"].abs().sum():
                df.loc[df['Step'] == step, 'Cycle'] = cycle
                cycle += 1
    
    # Add provenance to the metadatas
    timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
    metadata = {
        'provenance': {
            'aurora_metadata': {
                'data_merging': {
                    'job_files': job_files,
                    'repo_url': __url__,
                    'repo_version': __version__,
                    'method': 'analysis.combine_jobs',
                    'datetime': datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %z'),
                }
            },
            'original_file_provenance': {f: m['provenance'] for f, m in zip(job_files, metadatas)},
        },
        'sample_data': metadatas[-1].get('sample_data',{}),
        'job_data': [m.get('job_data',{}) for m in metadatas],
        'mpr_metadata': [m.get('mpr_metadata',{}) for m in metadatas],
        'glossary': {
            'uts': 'Unix time stamp in seconds',
            'V (V)': 'Cell voltage in volts',
            'I (A)': 'Current across cell in amps',
            'dt (s)': 'Time between data points in seconds',
            'Iavg (A)': 'Average current between adjacent datapoints in amps',
            'dQ (mAh)': 'Change in charge between adjacent datapoints in mAh',
            'Step': 'A step is unique combination of job file, cycle number and loop number, this can be a full charge/discharge cycle, but can also be e.g. a protection step',
            'Cycle': 'A cycle is a step which is considered a full, valid charge/discharge cycle, see the function specified in provenance for the criteria',
        },
    }

    return df, metadata

def analyse_cycles(
        job_files: List[str],
        voltage_lower_cutoff: float = 0,
        voltage_upper_cutoff: float = 5,
        save_cycle_dict: bool = False,
        save_merged_hdf: bool = False,
        save_merged_jsongz: bool = False,
    ) -> Tuple[pd.DataFrame, dict, dict]:
    """ Take multiple dataframes, merge and analyse the cycling data.
    
    Args:
        job_files (List[str]): list of paths to the json.gz job files
        voltage_lower_cutoff (float, optional): lower cutoff for voltage data
        voltage_upper_cutoff (float, optional): upper cutoff for voltage data
        save_cycle_dict (bool, optional): save the cycle_dict as a json file
        save_merged_hdf (bool, optional): save the merged dataframe as an hdf5 file

    Returns:
        pd.DataFrame: DataFrame containing the cycling data
        dict: dictionary containing the cycling analysis
        dict: metadata from the files

    TODO: Add save location as an argument.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    db_path = config["Database path"]

    df, metadata = combine_jobs(job_files)

    # update metadata
    timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
    metadata.setdefault('provenance', {}).setdefault('aurora_metadata', {})
    metadata['provenance']['aurora_metadata'].update({
        'analysis': {
            'repo_url': __url__,
            'repo_version': __version__,
            'method': 'analysis.analyse_cycles',
            'datetime': datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %z'),
        },
    })

    sample_data = metadata.get('sample_data',{})
    sampleid = sample_data.get('Sample ID',None)
    job_data = metadata.get('job_data',[])
    mpr_metadata = metadata.get('mpr_metadata',[])
    snapshot_status = job_data[-1].get('Snapshot status',None)
    snapshot_pipeline = job_data[-1].get('Pipeline',None)
    last_snapshot = job_data[-1].get('Last snapshot',None)

    # Extract useful information from the metadata
    mass_mg = sample_data.get('Cathode active material mass (mg)',np.nan)
    # Extract information from the tomato or mpr job data
    assert not (any(job_data) and any(mpr_metadata)), "Both tomato job and mpr data found, cannot be processed"

    max_V = 0
    formation_C = 0
    cycle_C = 0
    
    # TOMATO DATA
    # TODO separate max voltage in formation and cycling
    pipeline = None
    status = None
    if any(job_data):
        payloads = [j.get('Payload',[]) for j in job_data]
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Pipeline`, `Job ID` FROM pipelines WHERE `Sample ID` = ?", (sampleid,))
            row = cursor.fetchone()
            if row:
                pipeline = row[0]
                job_id = row[1]
                if job_id:
                    cursor.execute("SELECT `Status` FROM jobs WHERE `Job ID` = ?", (f"{job_id}",))
                    status = cursor.fetchone()[0]
    
        for payload in payloads:
            for method in payload.get('method',[]):
                voltage = method.get('limit_voltage_max',0)
                if voltage > max_V:
                    max_V = voltage

        for payload in payloads:
            for method in payload.get('method',[]):
                if method.get('technique',None) == 'loop':
                    if method['n_gotos'] < 4: # it is probably formation
                        for m in payload.get('method',[]):
                            if 'current' in m and 'C' in m['current']:
                                try:
                                    formation_C = _c_to_float(m['current'])
                                except ValueError:
                                    print(f"Not a valid C-rate: {m['current']}")
                                    formation_C = 0
                                break
                    if method.get('n_gotos',0) > 10: # it is probably cycling
                        for m in payload.get('method',[]):
                            if 'current' in m and 'C' in m['current']:
                                try:
                                    cycle_C = _c_to_float(m['current'])
                                except ValueError:
                                    print(f"Not a valid C-rate: {m['current']}")
                                    cycle_C = 0
                                break

    # MPR DATA
    # TODO get formation and cycle C, separate max voltage in formation and cycling
    if any(mpr_metadata):
        for m in mpr_metadata:
            for params in m.get('params',[]):
                V = round(params.get("EM",0),3)
                if V > max_V:
                    max_V = V

    # Fill some missing values
    if not formation_C:
        if not cycle_C:
            print(f"No formation C or cycle C found for {sampleid}, using 0")
        else:
            print(f"No formation C found for {sampleid}, using cycle_C")
            formation_C = cycle_C
    
    # Analyse each cycle in the cycling data
    charge_capacity_mAh = []
    discharge_capacity_mAh = []
    charge_avg_V = []
    discharge_avg_V = []
    charge_energy_mWh = []
    discharge_energy_mWh = []
    charge_avg_I = []
    discharge_avg_I = []
    started_charge = False
    started_discharge = False
    for cycle, group_df in df.groupby('Cycle'):
        if cycle <= 0:
            continue
        charge_data = group_df[
            (group_df['Iavg (A)'] > 0) &
            (group_df['V (V)'] > voltage_lower_cutoff) &
            (group_df['V (V)'] < voltage_upper_cutoff) &
            (group_df['dt (s)'] < 600)
        ]
        discharge_data = group_df[
            (group_df['Iavg (A)'] < 0) &
            (group_df['V (V)'] > voltage_lower_cutoff) &
            (group_df['V (V)'] < voltage_upper_cutoff) &
            (group_df['dt (s)'] < 600)
        ]
        # Only consider cycles with more than 10 data points
        started_charge=len(charge_data)>10
        started_discharge=len(discharge_data)>10

        if started_charge and started_discharge:
            charge_capacity_mAh.append(charge_data['dQ (mAh)'].sum())
            charge_avg_V.append((charge_data['V (V)']*charge_data['dQ (mAh)']).sum()/charge_data['dQ (mAh)'].sum())
            charge_energy_mWh.append((charge_data['V (V)']*charge_data['dQ (mAh)']).sum())
            charge_avg_I.append((charge_data['Iavg (A)']*charge_data['dQ (mAh)']).sum()/charge_data['dQ (mAh)'].sum())
            discharge_capacity_mAh.append(-discharge_data['dQ (mAh)'].sum())
            discharge_avg_V.append((discharge_data['V (V)']*discharge_data['dQ (mAh)']).sum()/discharge_data['dQ (mAh)'].sum())
            discharge_energy_mWh.append((-discharge_data['V (V)']*discharge_data['dQ (mAh)']).sum())
            discharge_avg_I.append((-discharge_data['Iavg (A)']*discharge_data['dQ (mAh)']).sum()/discharge_data['dQ (mAh)'].sum())

    formation_cycle_count = 3
    initial_cycle = formation_cycle_count + 2

    formed = len(charge_capacity_mAh) >= initial_cycle
    # A row is added if charge data is complete and discharge started
    # Last dict may have incomplete discharge data
    # TODO remove incomplete cycles based on voltage limits
    if snapshot_status != 'c':
        if started_charge and started_discharge:
            # Probably recorded an incomplete discharge for last recorded cycle
            discharge_capacity_mAh[-1] = np.nan
            complete = 0
        else:
            # Last recorded cycle is complete
            complete = 1
    else:
        complete = 1

    # Create a dictionary with the cycling data
    # TODO add datetime of every cycle
    cycle_dict = {
        'Sample ID': sampleid,
        'Cycle': list(range(1,len(charge_capacity_mAh)+1)),
        'Charge capacity (mAh)': charge_capacity_mAh,
        'Discharge capacity (mAh)': discharge_capacity_mAh,
        'Efficiency (%)': [100*d/c for d,c in zip(discharge_capacity_mAh,charge_capacity_mAh)],
        'Specific charge capacity (mAh/g)': [c/(mass_mg*1e-3) for c in charge_capacity_mAh],
        'Specific discharge capacity (mAh/g)': [d/(mass_mg*1e-3) for d in discharge_capacity_mAh],
        'Normalised discharge capacity (%)': [100*d/discharge_capacity_mAh[initial_cycle-1] for d in discharge_capacity_mAh] if formed else None,
        'Normalised discharge energy (%)': [100*d/discharge_energy_mWh[initial_cycle-1] for d in discharge_energy_mWh] if formed else None,
        'Charge average voltage (V)': charge_avg_V,
        'Discharge average voltage (V)': discharge_avg_V,
        'Delta V (V)': [c-d for c,d in zip(charge_avg_V,discharge_avg_V)],
        'Charge average current (A)': charge_avg_I,
        'Discharge average current (A)': discharge_avg_I,
        'Charge energy (mWh)': charge_energy_mWh,
        'Discharge energy (mWh)': discharge_energy_mWh,
        'Max voltage (V)': max_V,
        'Formation C': formation_C,
        'Cycle C': cycle_C,
    }

    # Add other columns from sample table to cycle_dict
    sample_cols_to_add = [
        "Actual N:P ratio",
        "Anode type",
        "Cathode type",
        "Anode active material mass (mg)",
        "Cathode active material mass (mg)",
        "Electrolyte name",
        "Electrolyte description",
        "Electrolyte amount (uL)",
        "Rack position",
    ]
    for col in sample_cols_to_add:
        cycle_dict[col] = sample_data.get(col, None)

    # Calculate additional quantities from cycling data and add to cycle_dict
    if not cycle_dict['Cycle']:
        print(f"No cycles found for {sampleid}")
        return df, cycle_dict, metadata
    if len(cycle_dict['Cycle']) == 1 and not complete:
        print(f"No complete cycles found for {sampleid}")
        return df, cycle_dict, metadata
    last_idx = -1 if complete else -2

    cycle_dict['First formation efficiency (%)'] = cycle_dict['Efficiency (%)'][0]
    cycle_dict['First formation specific discharge capacity (mAh/g)'] = cycle_dict['Specific discharge capacity (mAh/g)'][0]
    cycle_dict['Initial specific discharge capacity (mAh/g)'] = cycle_dict['Specific discharge capacity (mAh/g)'][initial_cycle-1] if formed else None
    cycle_dict['Initial efficiency (%)'] = cycle_dict['Efficiency (%)'][initial_cycle-1] if formed else None
    cycle_dict['Capacity loss (%)'] = 100 - cycle_dict['Normalised discharge capacity (%)'][last_idx] if formed else None
    cycle_dict['Last specific discharge capacity (mAh/g)'] = cycle_dict['Specific discharge capacity (mAh/g)'][last_idx]
    cycle_dict['Last efficiency (%)'] = cycle_dict['Efficiency (%)'][last_idx]
    cycle_dict['Formation average voltage (V)'] = np.mean(cycle_dict['Charge average voltage (V)'][:initial_cycle-1]) if formed else None
    cycle_dict['Formation average current (A)'] = np.mean(cycle_dict['Charge average current (A)'][:initial_cycle-1]) if formed else None
    cycle_dict['Initial delta V (V)'] = cycle_dict['Delta V (V)'][initial_cycle-1] if formed else None

    # Calculate cycles to x% of initial discharge capacity
    pcents = [95,90,85,80,75,70,60,50]
    norm = cycle_dict['Normalised discharge capacity (%)']
    for pcent in pcents:
        cycle_dict[f'Cycles to {pcent}% capacity'] = next(
            (i + 1 - initial_cycle for i in range(initial_cycle-1, len(norm) - 1)
            if norm[i] < pcent and norm[i+1] < pcent), None) if formed else None
    norm = cycle_dict['Normalised discharge energy (%)']
    for pcent in pcents:
        cycle_dict[f'Cycles to {pcent}% energy'] = next(
            (i + 1 - initial_cycle for i in range(initial_cycle-1, len(norm) - 1)
            if norm[i] < pcent and norm[i+1] < pcent), None) if formed else None

    cycle_dict['Run ID'] = _run_from_sample(sampleid)

    # Add times to cycle_dict
    uts_steps = {}
    for step in [3,5,6,10]:
        datetime_str = sample_data.get(f"Timestamp step {step}", None)
        if not datetime_str:
            uts_steps[step] = np.nan
            continue
        datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        datetime_object = timezone.localize(datetime_object)
        uts_steps[step] = datetime_object.timestamp()
    job_start = df['uts'].iloc[0]
    cycle_dict['Electrolyte to press (s)'] = round(uts_steps[10] - np.nanmin([uts_steps[3],uts_steps[5]]))
    cycle_dict['Electrolyte to electrode (s)'] = round(uts_steps[6] - np.nanmin([uts_steps[3],uts_steps[5]]))
    cycle_dict['Electrode to protection (s)'] = round(job_start - uts_steps[6])
    cycle_dict['Press to protection (s)'] = round(job_start - uts_steps[10])

    # Update the database with some of the results
    flag = None
    job_complete = status and status.endswith('c')
    if pipeline:
        if not job_complete:
            if formed and cycle_dict['Capacity loss (%)'] > 20:
                flag = 'Cap loss'
            if cycle_dict['First formation efficiency (%)'] < 60:
                flag = 'Form eff'
            if formed and cycle_dict['Initial efficiency (%)'] < 50:
                flag = 'Init eff'
            if formed and cycle_dict['Initial specific discharge capacity (mAh/g)'] < 100:
                flag = 'Init cap'
        else:
            flag = 'Complete'

    update_row = {
        'Pipeline': pipeline,
        'Status': status,
        'Flag': flag,
        'Number of cycles': int(max(cycle_dict['Cycle'])),
        'Capacity loss (%)': cycle_dict['Capacity loss (%)'],
        'Max voltage (V)': cycle_dict['Max voltage (V)'],
        'Formation C': cycle_dict['Formation C'],
        'Cycling C': cycle_dict['Cycle C'],
        'First formation efficiency (%)': cycle_dict['First formation efficiency (%)'],
        'Initial specific discharge capacity (mAh/g)': cycle_dict['Initial specific discharge capacity (mAh/g)'],
        'Initial efficiency (%)': cycle_dict['Initial efficiency (%)'],
        'Last specific discharge capacity (mAh/g)': cycle_dict['Last specific discharge capacity (mAh/g)'],
        'Last efficiency (%)': cycle_dict['Last efficiency (%)'],
        'Last snapshot': last_snapshot,
        'Last analysis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #'Last plotted' # do not update this column here
        'Snapshot status': snapshot_status,
        'Snapshot pipeline': snapshot_pipeline,
    }
    # round any floats to 3 decimal places
    for k,v in update_row.items():
        if isinstance(v, float):
            update_row[k] = round(v,3)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # insert a row with sampleid if it doesn't exist
        cursor.execute("INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)", (sampleid,))
        # update the row
        columns = ", ".join([f"`{k}` = ?" for k in update_row.keys()])
        cursor.execute(
            f"UPDATE results SET {columns} WHERE `Sample ID` = ?",
            (*update_row.values(), sampleid)
        )

    if save_cycle_dict or save_merged_hdf or save_merged_jsongz:
        save_folder = os.path.dirname(job_files[0])
        if not save_folder:
            save_folder = '.'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if save_cycle_dict:
            with open(f'{save_folder}/cycles.{sampleid}.json','w',encoding='utf-8') as f:
                json.dump({"data": cycle_dict, "metadata": metadata},f)
        if save_merged_hdf or save_merged_jsongz:
            df.drop(columns=['dt (s)', 'Iavg (A)'], inplace=True)
        if save_merged_hdf:
            df.to_hdf(
                f'{save_folder}/full.{sampleid}.h5',
                key="cycling",
                complib="blosc",
                complevel=4
            )
            with h5py.File(f'{save_folder}/full.{sampleid}.h5', 'a') as file:
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    file['cycling'].attrs[key] = value
        if save_merged_jsongz:
            with gzip.open(f'{save_folder}/full.{sampleid}.json.gz', 'wt', encoding='utf-8') as f:
                json.dump({'data': df.to_dict(orient='list'), 'metadata': metadata}, f)
    return df, cycle_dict, metadata

def analyse_sample(sample: str) -> Tuple[pd.DataFrame, dict, dict]:
    """ Analyse a single sample.
    
    Will search for the sample in the processed snapshots folder and analyse the cycling data.
    """
    run_id = _run_from_sample(sample)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed snapshots folder path"]
    file_location = os.path.join(data_folder, run_id, sample)
    job_files = [
        os.path.join(file_location,f) for f in os.listdir(file_location)
        if (f.startswith('snapshot') and f.endswith('.json.gz'))
    ]
    if not job_files:  # check if there are .h5 files
        job_files = [
            os.path.join(file_location,f) for f in os.listdir(file_location)
            if (f.startswith('snapshot') and f.endswith('.h5'))
        ]
    df, cycle_dict, metadata = analyse_cycles(job_files, save_cycle_dict=True, save_merged_hdf=False, save_merged_jsongz=True)
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE results SET `Last analysis` = ? WHERE `Sample ID` = ?",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sample)
        )
    return df, cycle_dict, metadata

def analyse_all_samples(
        sampleid_contains: str = "",
        mode: Literal['always','new_data','if_not_exists'] = 'new_data',
    ) -> None:
    """ Analyse all samples in the processed snapshots folder.

    Args: sampleid_contains (str, optional): only analyse samples with this
        string in the sampleid
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    snapshot_folder = config["Processed snapshots folder path"]

    if mode == 'new_data':
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID`, `Last snapshot`, `Last analysis` FROM results")
            results = cursor.fetchall()
        dtformat = '%Y-%m-%d %H:%M:%S'
        print([r for r in results])
        samples_to_analyse = [
            r[0] for r in results
            if r[0] and (
                not r[1] or
                not r[2] or
                datetime.strptime(r[1], dtformat) > datetime.strptime(r[2], dtformat)
            )
        ]
        print(f"Analyzing: {samples_to_analyse}")
    if mode == 'if_not_exists':
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID` FROM results WHERE `Last analysis` IS NULL")
            results = cursor.fetchall()
        samples_to_analyse = [r[0] for r in results]
        print(f"Analyzing: {samples_to_analyse}")

    for batch_folder in os.listdir(snapshot_folder):
        for sample in os.listdir(os.path.join(snapshot_folder, batch_folder)):
            if sampleid_contains and sampleid_contains not in sample:
                continue
            if mode != "always" and sample not in samples_to_analyse:
                continue
            try:
                analyse_sample(sample)
            except KeyError as e:
                print(f"No metadata found for {sample}: {e}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Failed to analyse {sample} with error {e}\n{tb}")

def plot_sample(sample: str) -> None:
    """ Plot the data for a single sample.
    
    Will search for the sample in the processed snapshots folder and plot V(t) 
    and capacity(cycle).
    """
    run_id = _run_from_sample(sample)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed snapshots folder path"]
    file_location = f"{data_folder}/{run_id}/{sample}"
    save_location = f"{config['Graphs folder path']}/{run_id}"

    # plot V(t)
    files = os.listdir(file_location)
    cycling_file = next((f for f in files if (f.startswith('full') and f.endswith('.json.gz'))), None)
    if not cycling_file:
        print(f"No full cycling file found in {file_location}")
        return
    with gzip.open(f'{file_location}/{cycling_file}', 'rt', encoding='utf-8') as f:
        data = json.load(f)['data']
    df = pd.DataFrame(data)
    df.sort_values('uts', inplace=True)
    fig, ax = plt.subplots(figsize=(6,4),dpi=72)
    plt.plot(pd.to_datetime(df["uts"], unit="s"),df["V (V)"])
    plt.ylabel('Voltage (V)')
    plt.xticks(rotation=45)
    plt.title(sample)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    fig.savefig(f'{save_location}/{sample}_V(t).png',bbox_inches='tight')

    # plot capacity
    analysed_file = next((f for f in files if (f.startswith('cycles') and f.endswith('.json'))), None)
    if not analysed_file:
        print(f"No files starting with 'cycles' found in {file_location}.")
        return
    with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
        cycle_dict = json.load(f)
        cycle_df = pd.DataFrame(cycle_dict["data"])
    assert not cycle_df.empty, f"Empty dataframe for {sample}"
    assert 'Cycle' in cycle_df.columns, f"No 'Cycle' column in {sample}"
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,4),dpi=72)
    ax[0].plot(cycle_df['Cycle'],cycle_df['Discharge capacity (mAh)'],'.-')
    ax[1].plot(cycle_df['Cycle'],cycle_df['Efficiency (%)'],'.-')
    ax[0].set_ylabel('Discharge capacity (mAh)')
    ax[1].set_ylabel('Efficiency (%)')
    ax[1].set_xlabel('Cycle')
    ax[0].set_title(sample)
    fig.savefig(f'{save_location}/{sample}_Capacity.png',bbox_inches='tight')
    with sqlite3.connect(config["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE results SET `Last plotted` = ? WHERE `Sample ID` = ?",
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sample)
        )
        conn.commit()
        cursor.close()

def plot_all_samples(
        snapshot_folder: str = None,
        sampleid_contains: str = None,
        mode: Literal['always','new_data','if_not_exists'] = 'new_data',
    ) -> None:
    """ Plots all samples in the processed snapshots folder.

    Args: snapshot_folder (str): path to the folder containing the processed 
        snapshots. Defaults to the path in the config file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    if not snapshot_folder:
        snapshot_folder = config["Processed snapshots folder path"]
    if mode == 'new_data':
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID`, `Last analysis`, `Last plotted` FROM results")
            results = cursor.fetchall()
        dtformat = '%Y-%m-%d %H:%M:%S'
        samples_to_plot = [
            r[0] for r in results
            if r[0] and (
                not r[1] or
                not r[2] or
                datetime.strptime(r[1], dtformat) > datetime.strptime(r[2], dtformat)
            )
        ]
    if mode == 'if_not_exists':
        with sqlite3.connect(config["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID` FROM results WHERE `Last plotted` IS NULL")
            results = cursor.fetchall()
        samples_to_plot = [r[0] for r in results]
    for run_folder in os.listdir(snapshot_folder):
        for sample in os.listdir(f'{snapshot_folder}/{run_folder}'):
            if sampleid_contains and sampleid_contains not in sample:
                continue
            if mode != "always" and sample not in samples_to_plot:
                continue
            try:
                print("Plotting", sample)
                plot_sample(sample)
                plt.close('all')
            except Exception as e:
                print(f"Failed to plot {sample} with error {e}")

def parse_sample_plotting_file(
        file_path: str = "K:/Aurora/cucumber/graph_config.yml"
    ) -> dict:
    """ Reads the graph config file and returns a dictionary of the batches to plot.
    
    Args: file_path (str): path to the yaml file containing the plotting configuration
        Defaults to "K:/Aurora/cucumber/graph_config.yml"
    
    Returns: dict: dictionary of the batches to plot
        Dictionary contains the plot name as the key and a dictionary of the batch details as the
        value. Batch dict contains the samples to plot and any other plotting options.

    TODO: Put the graph config location in the config file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed snapshots folder path"]

    with open(file_path, 'r', encoding = 'utf-8') as file:
        batches = yaml.safe_load(file)

    for plot_name, batch in batches.items():
        samples = batch['samples']
        transformed_samples = []
        for sample in samples:
            split_name = sample.split(' ',1)
            if len(split_name) == 1:  # the batch is a single sample
                sample_id = sample
                run_id = _run_from_sample(sample_id)
                transformed_samples.append(sample_id)
            else:
                run_id, sample_range = split_name
                if sample_range.strip().startswith('[') and sample_range.strip().endswith(']'):
                    sample_numbers = json.loads(sample_range)
                    transformed_samples.extend([f"{run_id}_{i:02d}" for i in sample_numbers])
                elif sample_range == 'all':
                    # Check the folders
                    if os.path.exists(f"{data_folder}/{run_id}"):
                        transformed_samples.extend(os.listdir(f"{data_folder}/{run_id}"))
                    else:
                        print(f"Folder {data_folder}/{run_id} does not exist")
                else:
                    numbers = re.findall(r'\d+', sample_range)
                    start, end = map(int, numbers) if len(numbers) == 2 else (int(numbers[0]), int(numbers[0]))
                    transformed_samples.extend([f"{run_id}_{i:02d}" for i in range(start, end+1)])

        # Check if individual sample folders exist
        for sample in transformed_samples:
            run_id = _run_from_sample(sample)
            if not os.path.exists(f"{data_folder}/{run_id}/{sample}"):
                print(f"Folder {data_folder}/{run_id}/{sample} does not exist")
                # remove this element from the list
                transformed_samples.remove(sample)

        # overwrite the samples with the transformed samples
        batches[plot_name]['samples'] = transformed_samples

    return batches

def analyse_batch(plot_name: str, batch: dict) -> None:
    """ Combines data for a batch of samples. """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed snapshots folder path"]
    save_location = os.path.join(config['Batches folder path'],plot_name)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    samples = batch.get('samples')
    cycle_dicts = []
    metadata = {"sample_metadata":{}}
    for sample in samples:
        # get the anaylsed data
        run_id = _run_from_sample(sample)
        sample_folder = os.path.join(data_folder,run_id,sample)
        try:
            analysed_file = next(
                f for f in os.listdir(sample_folder)
                if (f.startswith('cycles') and f.endswith('.json'))
            )
            with open(f'{sample_folder}/{analysed_file}', 'r', encoding='utf-8') as f:
                data = json.load(f)
                cycle_dict = data.get('data',{})
                metadata['sample_metadata'][sample] = data.get('metadata',{})
            if cycle_dict.get('Cycle') and cycle_dict['Cycle']:
                cycle_dicts.append(cycle_dict)
            else:
                print(f"No cycling data for {sample}")
                continue
        except StopIteration:
            # Handle the case where no file starts with 'cycles'
            print(f"No files starting with 'cycles' found in {sample_folder}.")
            continue
    cycle_dicts = [d for d in cycle_dicts if d.get('Cycle') and d['Cycle']]
    assert len(cycle_dicts) > 0, "No cycling data found for any sample"

    # update the metadata
    timezone = pytz.timezone(config.get("Time zone", "Europe/Zurich"))
    metadata['provenance'] = {
        'aurora_metadata': {
            'batch_analysis': {
                'repo_url': __url__,
                'repo_version': __version__,
                'method': 'analysis.analyse_batch',
                'datetime': datetime.now(timezone).strftime('%Y-%m-%d %H:%M:%S %z'),
            },
        },
    }

    # make another df where we only keep the lists from the dictionaries in the list
    only_lists = pd.concat([pd.DataFrame({k:v for k,v in cycle_dict.items() if isinstance(v, list) or k=="Sample ID"}) for cycle_dict in cycle_dicts])
    only_vals = pd.DataFrame([{k:v for k,v in cycle_dict.items() if not isinstance(v, list)} for cycle_dict in cycle_dicts])

    with pd.ExcelWriter(f'{save_location}/batch.{plot_name}.xlsx') as writer:
        only_lists.to_excel(writer, sheet_name='Data by cycle', index=False)
        only_vals.to_excel(writer, sheet_name='Results by sample', index=False)
    with open(f'{save_location}/batch.{plot_name}.json','w',encoding='utf-8') as f:
        json.dump({"data":cycle_dicts, "metadata": metadata},f)

def plot_batch(plot_name: str, batch: dict) -> None:
    """ Plots the data for a batch of samples.
    
    Args:
        plot_name (str): name of the plot
        batch (dict): dict with 'samples' key containing list of samples to plot
            and any other plotting options e.g. group_by, palette, etc.
    """
    # Load the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path, encoding = 'utf-8') as f:
        config = json.load(f)
    save_location = os.path.join(config['Batches folder path'],plot_name)
    filename = next((f for f in os.listdir(save_location) if f.startswith('batch') and f.endswith('.json')), None)
    if not filename:
        raise FileNotFoundError(f"No batch data found for {plot_name}")
    with open(os.path.join(save_location,filename),'r',encoding='utf-8') as f:
        data = json.load(f)
    cycle_dicts = data.get('data',[])
    data = [pd.DataFrame(d) for d in cycle_dicts if not pd.DataFrame(d).dropna(how='all').empty]
    cycle_df = pd.concat(pd.DataFrame(d) for d in data).reset_index(drop=True)

    palette = batch.get('palette', 'deep')
    group_by = batch.get('group_by', None)

    n_cycles = max(cycle_df["Cycle"])
    if n_cycles > 10:
        cycles_to_plot = [1] + list(range(0, n_cycles, n_cycles // 10))[1:]
    else:
        cycles_to_plot = list(range(1, n_cycles + 1))
    plot_data = cycle_df[cycle_df["Cycle"].isin(cycles_to_plot)]

    # Set limits
    discharge_ylim = batch.get('discharge_ylim', None)
    if discharge_ylim:
        d_ymin, d_ymax = sorted(discharge_ylim)
    else:
        d_ymin = max(0, 0.95*cycle_df['Specific discharge capacity (mAh/g)'].min())
        d_ymax = cycle_df['Specific discharge capacity (mAh/g)'].max()*1.05
    efficiency_ylim = batch.get('efficiency_ylim', None)
    if efficiency_ylim:
        e_ymin, e_ymax = sorted(efficiency_ylim)
    else:
        e_ymin = max(70, 0.9*cycle_df['Efficiency (%)'].min())
        e_ymax = min(101, cycle_df['Efficiency (%)'].max()*1.1)

    ### STRIP PLOT ###
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\ncapacity (mAh/g)")
    sns.stripplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific discharge capacity (mAh/g)",
        size=3,
        edgecolor='k',
        palette=palette,
        hue = group_by,
    )
    sns.stripplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        size=3,
        edgecolor='k',
        palette=palette,
        hue = group_by,
    )
    ax[0].set_ylim(d_ymin, d_ymax)
    ax[1].set_ylim(e_ymin, e_ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_strip.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_strip.pdf")
    plt.close('all')

    ### Swarm plot ###
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\ncapacity (mAh/g)")
    sns.swarmplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific discharge capacity (mAh/g)",
        size=3,
        dodge=True,
        edgecolor='k',
        palette=palette,
        hue = group_by,
    )
    sns.swarmplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        size=3,
        dodge=True,
        edgecolor='k',
        palette=palette,
        hue = group_by,
    )
    ax[0].set_ylim(d_ymin, d_ymax)
    ax[1].set_ylim(e_ymin, e_ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_swarm.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_swarm.pdf")
    plt.close('all')

    ### Box plot ###
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\ncapacity (mAh/g)")
    sns.boxplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific discharge capacity (mAh/g)",
        fill=False,
        palette=palette,
        hue = group_by,
    )
    sns.boxplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        fill=False,
        palette=palette,
        hue = group_by,
    )
    ax[0].set_ylim(d_ymin, d_ymax)
    ax[1].set_ylim(e_ymin, e_ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_box.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_box.pdf")
    plt.close('all')

    ### Interative plot ###
    if group_by:  # Group points by 'group_by' column and sample id
        sorted_df = cycle_df.sort_values(by=[group_by, 'Sample ID'])
        sorted_df['Group_Number'] = sorted_df.groupby([group_by, 'Sample ID']).ngroup()
    else:  # Just group by sample id
        sorted_df = cycle_df.sort_values(by='Sample ID')
        sorted_df['Group_Number'] = sorted_df.groupby('Sample ID').ngroup()
    # Apply an offset to the 'Cycle' column based on group number
    num_combinations = sorted_df['Group_Number'].nunique()
    offsets = np.linspace(-0.25, 0.25, num_combinations)
    group_to_offset = dict(zip(sorted_df['Group_Number'].unique(), offsets))
    sorted_df['Offset'] = sorted_df['Group_Number'].map(group_to_offset)
    sorted_df['Jittered cycle'] = sorted_df['Cycle'] + sorted_df['Offset']
    cycle_df = sorted_df.drop(columns=['Group_Number'])  # drop the temporary column

    # We usually want voltage as a categorical
    cycle_df["Max voltage (V)"] = cycle_df["Max voltage (V)"].astype(str)
    # C-rate should be a fraction
    cycle_df["Formation C/"] = cycle_df["Formation C"].apply(
        lambda x: str(fractions.Fraction(x).limit_denominator())
        )
    cycle_df["Formation C"] = 1/cycle_df["Formation C"]
    cycle_df["Cycle C/"] = cycle_df["Cycle C"].apply(
        lambda x: str(fractions.Fraction(x).limit_denominator())
        )
    cycle_df["Cycle C"] = 1/cycle_df["Cycle C"]
    cycle_df["Formation C"] = pd.to_numeric(cycle_df["Formation C"], errors='coerce')
    hover_columns = [
        'Sample ID',
        'Cycle',
        'Specific discharge capacity (mAh/g)',
        'Efficiency (%)',
        'Max voltage (V)',
        'Formation C/',
        'Cycle C/',
        'Cathode active material mass (mg)',
        'Electrolyte name',
        'Actual N:P ratio',
    ]
    hover_template = (
        'Sample ID: %{customdata[0]}<br>'
        'Cycle: %{customdata[1]}<br>'
        'Specific discharge capacity (mAh/g): %{customdata[2]:.2f}<br>'
        'Efficiency (%): %{customdata[3]:.3f}<br>'
        'Max voltage (V): %{customdata[4]}<br>'
        'Formation C-rate: %{customdata[5]}<br>'
        'Cycle C-rate: %{customdata[6]}<br>'
        'Cathode active material mass (mg): %{customdata[7]:.4f}<br>'
        'Electrolyte: %{customdata[8]}<br>'
        'N:P ratio: %{customdata[9]:.4f}<br><extra></extra>'
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.1)
    fig.update_layout(
        template = 'ggplot2',
    )
    hex_colours = sns.color_palette(palette, num_combinations).as_hex()
    scatter1 = px.scatter(
        cycle_df,
        x='Jittered cycle',
        y='Specific discharge capacity (mAh/g)',
        color=group_by,
        color_discrete_sequence=hex_colours,
        custom_data=cycle_df[hover_columns],
    )
    for trace in scatter1.data:
        trace.hovertemplate = hover_template
        fig.add_trace(trace, row=1, col=1)

    scatter2 = px.scatter(
        cycle_df,
        x='Jittered cycle',
        y='Efficiency (%)',
        color=group_by,
        color_discrete_sequence=hex_colours,
        custom_data=cycle_df[hover_columns],
    )
    for trace in scatter2.data:
        trace.showlegend = False
        trace.hovertemplate = hover_template
        fig.add_trace(trace, row=2, col=1)

    fig.update_xaxes(title_text="Cycle", row=2, col=1)
    if discharge_ylim:
        ymin, ymax = sorted(discharge_ylim)
    else:
        ymin = max(0, 0.95*cycle_df['Specific discharge capacity (mAh/g)'].min())
        ymax = cycle_df['Specific discharge capacity (mAh/g)'].max()*1.05
    fig.update_yaxes(title_text="Specific discharge<br>capacity (mAh/g)", row=1, col=1, range=[ymin, ymax])
    ymin = max(70, cycle_df['Efficiency (%)'].min())
    ymax = min(101, 1.05*cycle_df['Efficiency (%)'].max())
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1, range=[ymin, ymax])
    if group_by:
        fig.update_layout(
            legend_title_text=group_by,
            )
    fig.update_layout(coloraxis = dict(colorscale=palette))

    # save the plot
    try:
        fig.write_html(
            os.path.join(save_location,f'{plot_name}_interactive.html'),
            config={'scrollZoom':True, 'displaylogo': False}
        )
    except PermissionError:
        print(
            "Permission error saving "
            f"{os.path.join(save_location,f'{plot_name}_interactive.html')}"
        )

def analyse_all_batches(
        graph_config_path: str= "K:/Aurora/cucumber/graph_config.yml"
    ) -> None:
    """ Analyses all the batches according to the configuration file.

    Args:
        file_path (str): path to the yaml file containing the plotting config
            Defaults to "K:/Aurora/cucumber/graph_config.yml"
    
    Will search for analysed data in the processed snapshots folder and plot and
    save the capacity and efficiency vs cycle for each batch of samples.
    """
    batches = parse_sample_plotting_file(graph_config_path)
    for plot_name, batch in batches.items():
        try:
            analyse_batch(plot_name,batch)
        except Exception as e:
            print(f"Failed to analyse {plot_name} with error {e}")

def plot_all_batches(
        graph_config_path: str= "K:/Aurora/cucumber/graph_config.yml"
    ) -> None:
    """ Plots all the batches according to the configuration file.

    Args:
        file_path (str): path to the yaml file containing the plotting config
            Defaults to "K:/Aurora/cucumber/graph_config.yml"
    
    Will search for analysed data in the processed snapshots folder and plot and
    save the capacity and efficiency vs cycle for each batch of samples.
    """
    batches = parse_sample_plotting_file(graph_config_path)
    for plot_name, batch in batches.items():
        try:
            plot_batch(plot_name,batch)
        except AssertionError as e:
            print(f"Failed to plot {plot_name} with error {e}")
        plt.close('all')

def _c_to_float(c_rate: str) -> float:
    """ Convert a C-rate string to a float.

    Args:
        c_rate (str): C-rate string, e.g. 'C/2', '0.5C', '3D/5', '1/2 D'
    Returns:
        float: C-rate as a float
    """
    if 'C' in c_rate:
        sign = 1
    elif 'D' in c_rate:
        c_rate = c_rate.replace('D', 'C')
        sign = -1
    else:
        raise ValueError(f"Invalid C-rate: {c_rate}")

    num, _, denom = c_rate.partition('C')
    number = "".join([num, denom]).strip()

    if '/' in number:
        num, denom = number.split('/')
        if not num:
            num = 1
        if not denom:
            denom = 1
        return sign * float(num) / float(denom)
    else:
        return sign * float(number)

def _run_from_sample(sampleid: str) -> str:
    """ Get the run_id from a sample_id. """
    if len(sampleid.split("_")) < 2 or not sampleid.split("_")[-1].isdigit():
        return "misc"
    else:
        return sampleid.rsplit('_', 1)[0]
