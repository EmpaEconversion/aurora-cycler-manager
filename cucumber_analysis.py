""" Functions used by cucumber for parsing, analysing and plotting.

Parsing:
Contains functions for converting raw jsons from tomato to pandas dataframes,
which can be saved to compressed hdf5 files.

Also includes functions for analysing the cycling data, extracting the
charge, discharge and efficiency of each cycle, and links this to various
quantities extracted from the cycling, such as C-rate and max voltage, and
from the sample database such as cathode mass.
"""
import os
import re
import sqlite3
from typing import List, Tuple
from datetime import datetime
import traceback
import json
import fractions
import yaml
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from version import __version__

def convert_tomato_json(
        snapshot_file: str,
        output_hdf_file: str = None,
        ) -> pd.DataFrame:
    """ Convert a raw json file from tomato to a pandas dataframe.

    Args:
        snapshot_file (str): path to the raw json file
        hdf_save_location (str, optional): path to save the output hdf5 file

    Returns:
        pd.DataFrame: DataFrame containing the cycling data

    Columns in output DataFrame:
    - uts: UTS Timestamp in seconds
    - Ewe: Voltage in V
    - I: Current in A
    - loop_number: how many loops have been completed
    - cycle_number: used if there is a loop of loops
    - index: index of the method in the payload
    - technique: code of technique using Biologic convention
        100 = OCV, 101 = CA, 102 = CP, 103 = CV, 155 = CPLIMIT, 157 = CALIMIT, 
        -1 = Unknown to Cucumber
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
            "Ewe" : [row["raw"]["Ewe"]["n"] for row in step_data],
            "I": [row["raw"]["I"]["n"] if "I" in row["raw"] else 0 for row in step_data],
            "cycle_number": [row["raw"]["cycle number"] if "cycle number" in row["raw"] else -1 for row in step_data],
            "loop_number": [row["raw"]["loop number"] if "cycle number" in row["raw"] else -1 for row in step_data],
            "index" : [row["raw"]["index"] if "index" in row["raw"] else -1 for row in step_data],
            "technique" : [technique_code.get(row["raw"]["technique"], -1) if "technique" in row["raw"] else -1 for row in step_data],
        }
        data.append(pd.DataFrame(step_dict))
    data = pd.concat(data, ignore_index=True)
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
        # Try to get the job number from the snapshot file and add to metadata
        try:
            json_filename = os.path.basename(snapshot_file)
            jobid = "".join(json_filename.split(".")[1:-1])
            # look up jobid in the database
            with open('./config.json', encoding = 'utf-8') as f:
                config = json.load(f)
            db_path = config["Database Path"]
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Get all data about this job
                cursor.execute(f"SELECT * FROM jobs WHERE `Job ID`='{jobid}'")
                job_data = dict(cursor.fetchone())
                sampleid = job_data["Sample ID"]
                # Get all data about this sample
                cursor.execute(f"SELECT * FROM samples WHERE `Sample ID`='{sampleid}'")
                sample_data = dict(cursor.fetchone())
        except Exception as e:
            print(f"Error getting job and sample data from database: {e}")
            job_data = None
            sample_data = None

        # add metadata to the hdf5 file
        # Metadata to add
        metadata = {
            "snapshot_file": snapshot_file,
            "n_steps": n_steps,
            "tomato_metadata": json.dumps(input_dict["metadata"]),
            "conversion_method": f"cucumber_analysis.py convert_tomato_json v{__version__}",
            "job_data": json.dumps(job_data) if job_data is not None else None,
            "sample_data": json.dumps(sample_data) if sample_data is not None else None,
        }
        # Open the HDF5 file with h5py and add metadata
        with h5py.File(output_hdf_file, 'a') as file:
            if 'cycling' in file:
                for key, value in metadata.items():
                    file['cycling'].attrs[key] = value
            else:
                print("Dataset 'cycling' not found.")
    return data

def convert_all_tomato_jsons() -> None:
    """ Goes through all the raw json files in the snapshots folder and converts them to hdf5.
    
    TODO: Add option to only convert files with new data.
    """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    raw_folder = config["Snapshots Folder Path"]
    processed_folder = config["Processed Snapshots Folder Path"]
    for batch_folder in os.listdir(raw_folder):
        for sample_folder in os.listdir(os.path.join(raw_folder, batch_folder)):
            for snapshot_file in os.listdir(os.path.join(raw_folder, batch_folder, sample_folder)):
                if snapshot_file.startswith('snapshot') and snapshot_file.endswith('.json'):
                    output_folder = os.path.join(processed_folder,batch_folder,sample_folder)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    convert_tomato_json(
                        os.path.join(raw_folder,batch_folder,sample_folder,snapshot_file),
                        os.path.join(output_folder,snapshot_file.replace('.json','.h5'))
                    )
                    print(f"Converted {snapshot_file}")

def analyse_cycles(
        h5_files: List[str],
        voltage_lower_cutoff: float = 0,
        voltage_upper_cutoff: float = 5,
        save_files: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
    """ Take multiple dataframes, merge and analyse the cycling data.
    
    Args:
        h5_files (List[str]): list of paths to the hdf5 files
        voltage_lower_cutoff (float, optional): lower cutoff for voltage data
        voltage_upper_cutoff (float, optional): upper cutoff for voltage data
        save_files (bool, optional): whether to save the output files
            files will be saved in the same folder as the first input file
    
    TODO: Add save location as an argument.
    """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    db_path = config["Database Path"]

    # Get the metadata from the files
    dfs = []
    metadatas = []
    payloads = []
    sampleids = []
    for f in h5_files:
        dfs.append(pd.read_hdf(f))
        with h5py.File(f, 'r') as file:
            try:
                metadata=dict(file['cycling'].attrs)
                job_data = json.loads(metadata.get('job_data','{}'))
                metadatas.append(metadata)
                sampleids.append(
                    json.loads(metadata['sample_data'])['Sample ID']
                )
                payloads.append(json.loads(job_data.get('Payload','{}')))
            except KeyError as exc:
                print(f"Metadata not found in {f}")
                raise KeyError from exc
    assert len(set(sampleids)) == 1, "All files must be from the same sample"
    sampleid = sampleids[0]
    order = np.argsort([df['uts'].iloc[0] for df in dfs])
    dfs = [dfs[i] for i in order]
    h5_files = [h5_files[i] for i in order]
    metadatas = [metadatas[i] for i in order]
    payloads = [payloads[i] for i in order]

    metadata = metadatas[-1]
    sample_data = json.loads(metadata['sample_data'])
    job_data = json.loads(metadata.get('job_data','{}'))
    snapshot_status = job_data.get('Snapshot Status',None)
    snapshot_pipeline = job_data.get('Pipeline',None)
    last_snapshot = job_data.get('Last Snapshot',None)

    pipeline = None
    status = None
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Pipeline`, `Job ID`, `Server Label` FROM pipelines WHERE `Sample ID` = ?", (sampleid,))
        row = cursor.fetchone()
        if row:
            pipeline = row[0]
            job_id = row[1]
            server_label = row[2]
            if job_id:
                cursor.execute("SELECT `Status` FROM jobs WHERE `Job ID` = ?", (f"{server_label}-{job_id}",))
                status = cursor.fetchone()[0]

    for i,df in enumerate(dfs):
        df["job_number"] = i
    df = pd.concat(dfs)
    df["dt (s)"] = np.concatenate([[0],df["uts"].values[1:] - df["uts"].values[:-1]])
    df["Iavg (A)"] = np.concatenate([[0],(df["I"].values[1:] + df["I"].values[:-1]) / 2])
    df["dQ (mAh)"] = 1e3 * df["Iavg (A)"] * df["dt (s)"] / 3600

    # Extract useful information from the metadata
    mass_mg = sample_data['Cathode Active Material Weight (mg)']
    max_V = 0
    formation_C = 0
    cycle_C = 0
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
    if not formation_C:
        if not cycle_C:
            print(f"No formation C or cycle C found for {sampleid}, using 0")
        else:
            print(f"No formation C found for {sampleid}, using cycle_C")
            formation_C = cycle_C

    # If converted from an mpr file, get the max voltage from the metadata
    if 'mpr_metadata' in metadata:
        mpr_metadata = json.loads(metadata['mpr_metadata'])
        for params in mpr_metadata.get('params',[]):
            V = round(params.get("EM",0),3)
            if V > max_V:
                max_V = V

    # Detect whenever job, cycle or loop changes
    # Necessary because the cycle number is not always recorded correctly so
    # the combination of job, cycle, loop is not necessarily unique
    df = df.sort_values('uts')
    df['group_id'] = (
        (df['loop_number'] < df['loop_number'].shift(1)) |
        (df['cycle_number'] < df['cycle_number'].shift(1)) |
        (df['job_number'] < df['job_number'].shift(1))
    ).cumsum()
    df['global_idx'] = df.groupby(['job_number', 'group_id', 'cycle_number', 'loop_number']).ngroup()
    charge_capacity_mAh = []
    discharge_capacity_mAh = []
    for _, group_df in df.groupby('global_idx'):
        charge_data = group_df[
            (group_df['Iavg (A)'] > 0) &
            (group_df['Ewe'] > voltage_lower_cutoff) &
            (group_df['Ewe'] < voltage_upper_cutoff) &
            (group_df['dt (s)'] < 600)
        ]
        discharge_data = group_df[
            (group_df['Iavg (A)'] < 0) &
            (group_df['Ewe'] > voltage_lower_cutoff) &
            (group_df['Ewe'] < voltage_upper_cutoff) &
            (group_df['dt (s)'] < 600)
        ]
        # Only consider cycles with more than 10 data points
        started_charge=len(charge_data)>10
        started_discharge=len(discharge_data)>10

        if started_charge and started_discharge:
            charge_capacity_mAh.append(charge_data['dQ (mAh)'].sum())
            discharge_capacity_mAh.append(-discharge_data['dQ (mAh)'].sum())
    cycle_dict = {
        'Sample ID': sampleid,
        'Cycle': list(range(1,len(charge_capacity_mAh)+1)),
        'Charge Capacity (mAh)': charge_capacity_mAh,
        'Discharge Capacity (mAh)': discharge_capacity_mAh,
        'Efficiency (%)': [100*d/c for d,c in zip(discharge_capacity_mAh,charge_capacity_mAh)],
        'Specific Charge Capacity (mAh/g)': [c/(mass_mg*1e-3) for c in charge_capacity_mAh],
        'Specific Discharge Capacity (mAh/g)': [d/(mass_mg*1e-3) for d in discharge_capacity_mAh],
        'Cathode Mass (mg)': mass_mg,
        'Max Voltage (V)': max_V,
        'Formation C': formation_C,
        'Cycle C': cycle_C,
    }
    # Add other columns from sample table to cycle_dict
    sample_cols_to_add = [
        "Actual N:P Ratio",
        "Electrolyte Name",
    ]
    for col in sample_cols_to_add:
        cycle_dict[col] = sample_data.get(col, None)

    # A dict is made if charge data is complete and discharge started
    # Last dict may have incomplete discharge data
    if snapshot_status != 'c':
        if started_charge and started_discharge:
            # Probably recorded an incomplete discharge for last recorded cycle
            cycle_dict['Discharge Capacity (mAh)'][-1] = np.nan
            cycle_dict['Efficiency (%)'][-1] = np.nan
            cycle_dict['Specific Discharge Capacity (mAh/g)'][-1] = np.nan
            complete = 0
        else:
            # Last recorded cycle is complete
            complete = 1
    else:
        complete = 1

    if not cycle_dict['Cycle']:
        print(f"No cycles found for {sampleid}")
        return df, cycle_dict
    if len(cycle_dict['Cycle']) == 1 and not complete:
        print(f"No complete cycles found for {sampleid}")
        return df, cycle_dict
    last_idx = -1 if complete else -2
    form_eff = round(cycle_dict['Efficiency (%)'][last_idx],3)
    init_dis_cap = (
        round(cycle_dict['Specific Discharge Capacity (mAh/g)'][4],3)
        if len(cycle_dict['Cycle']) > 5
        else None
    )
    init_eff = (
        round(cycle_dict['Efficiency (%)'][4],3)
        if len(cycle_dict['Cycle']) > 5
        else None
    )
    last_dis_cap = round(cycle_dict['Specific Discharge Capacity (mAh/g)'][last_idx],3)
    last_eff = round(cycle_dict['Efficiency (%)'][last_idx],3)
    cap_loss = (
        round((init_dis_cap - last_dis_cap) / init_dis_cap * 100, 3)
        if init_dis_cap
        else None
    )
    flag = None
    job_complete = status and status.endswith('c')
    if pipeline:
        if not job_complete:
            if cap_loss and cap_loss > 20:
                flag = 'Cap loss'
            if form_eff < 50:
                flag = 'Form eff'
            if init_eff and init_eff < 50:
                flag = 'Init eff'
            if init_dis_cap and init_dis_cap< 50:
                flag = 'Init cap'
        else:
            flag = 'Complete'

    update_row = {
        'Pipeline': pipeline,
        'Status': status,
        'Flag': flag,
        'Number of cycles': int(max(cycle_dict['Cycle'])),
        'Capacity loss (%)': cap_loss,
        'Max Voltage (V)': cycle_dict['Max Voltage (V)'],
        'Formation C': cycle_dict['Formation C'],
        'Cycling C': cycle_dict['Cycle C'],
        'First formation efficiency (%)': form_eff,
        'Initial discharge specific capacity (mAh/g)': init_dis_cap,
        'Initial efficiency (%)': init_eff,
        'Last discharge specific capacity (mAh/g)': last_dis_cap,
        'Last efficiency (%)': last_eff,
        'Last Snapshot': last_snapshot,
        'Last analysis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #'Last plotted' # do not update this column here
        'Snapshot status': snapshot_status,
        'Snapshot pipeline': snapshot_pipeline,
    }
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

    if save_files:
        save_folder = os.path.dirname(h5_files[0])
        if not save_folder:
            save_folder = '.'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        with open(f'{save_folder}/cycles.{sampleids[0]}.json','w',encoding='utf-8') as f:
            json.dump(cycle_dict,f)
    return df, cycle_dict

def analyse_sample(sample: str) -> Tuple[pd.DataFrame, dict]:
    """ Analyse a single sample.
    
    Will search for the sample in the processed snapshots folder and analyse the cycling data.
    """
    batch = sample.rsplit('_',1)[0]
    with open('./config.json', encoding='utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]
    file_location = os.path.join(data_folder, batch, sample)
    h5_files = [
        os.path.join(file_location,f) for f in os.listdir(file_location)
        if (f.startswith('snapshot') and f.endswith('.h5'))
    ]
    df, cycle_dict = analyse_cycles(h5_files, save_files=True)
    return df, cycle_dict

def analyse_all_samples() -> None:
    """ Analyse all samples in the processed snapshots folder.
    
    TODO: Only analyse files with new data since last analysis
    """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    snapshot_folder = config["Processed Snapshots Folder Path"]
    for batch_folder in os.listdir(snapshot_folder):
        for sample in os.listdir(os.path.join(snapshot_folder, batch_folder)):
            try:
                analyse_sample(sample)
            except KeyError:
                print(f"No metadata found for {sample}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Failed to analyse {sample} with error {e}\n{tb}")

def plot_sample(sample: str) -> None:
    """ Plot the data for a single sample.
    
    Will search for the sample in the processed snapshots folder and plot V(t) 
    and capacity(cycle).
    """
    batch = sample.rsplit('_',1)[0]
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]
    file_location = f"{data_folder}/{batch}/{sample}"
    save_location = f"{config['Graphs Folder Path']}/{batch}"

    # plot V(t)
    files = os.listdir(file_location)
    cycling_files = [f for f in files if (f.startswith('snapshot') and f.endswith('.h5'))]
    if not cycling_files:
        print(f"No cycling files found in {file_location}")
        return
    dfs = [pd.read_hdf(f'{file_location}/{f}') for f in cycling_files]
    df = pd.concat(dfs)
    df.sort_values('uts', inplace=True)
    fig, ax = plt.subplots(figsize=(6,4),dpi=72)
    plt.plot(pd.to_datetime(df["uts"], unit="s"),df["Ewe"])
    plt.ylabel('Voltage (V)')
    plt.xticks(rotation=45)
    plt.title(sample)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    fig.savefig(f'{save_location}/{sample}_V(t).png',bbox_inches='tight')

    # plot capacity
    try:
        analysed_file = next(f for f in files if (f.startswith('cycles') and f.endswith('.json')))
    except StopIteration:
        print(f"No files starting with 'cycles' found in {file_location}.")
        return
    with open(f'{file_location}/{analysed_file}', 'r', encoding='utf-8') as f:
        cycle_df = pd.DataFrame(json.load(f))
    assert not cycle_df.empty, f"Empty dataframe for {sample}"
    assert 'Cycle' in cycle_df.columns, f"No 'Cycle' column in {sample}"
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,4),dpi=72)
    ax[0].plot(cycle_df['Cycle'],cycle_df['Discharge Capacity (mAh)'],'.-')
    ax[1].plot(cycle_df['Cycle'],cycle_df['Efficiency (%)'],'.-')
    ax[0].set_ylabel('Discharge Capacity (mAh)')
    ax[1].set_ylabel('Efficiency (%)')
    ax[1].set_xlabel('Cycle')
    ax[0].set_title(sample)
    fig.savefig(f'{save_location}/{sample}_Capacity.png',bbox_inches='tight')

def plot_all_samples(snapshot_folder: str = None) -> None:
    """ Plots all samples in the processed snapshots folder.

    Args: snapshot_folder (str): path to the folder containing the processed 
        snapshots. Defaults to the path in the config file.

    TODO: Only plot samples with new data since last plot
    """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    if not snapshot_folder:
        snapshot_folder = config["Processed Snapshots Folder Path"]
    for batch_folder in os.listdir(snapshot_folder):
        for sample in os.listdir(f'{snapshot_folder}/{batch_folder}'):
            try:
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
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]

    with open(file_path, 'r', encoding = 'utf-8') as file:
        batches = yaml.safe_load(file)

    for plot_name, batch in batches.items():
        samples = batch['samples']
        transformed_samples = []
        for sample in samples:
            split_name = sample.split(' ',1)
            if len(split_name) == 1:  # the batch is a single sample
                sample_name = sample
                batch_name = sample.rsplit('_',1)[0]
                transformed_samples.append(sample_name)
            else:
                batch_name, sample_range = split_name
                if sample_range.strip().startswith('[') and sample_range.strip().endswith(']'):
                    sample_numbers = json.loads(sample_range)
                    transformed_samples.extend([f"{batch_name}_{i:02d}" for i in sample_numbers])
                elif sample_range == 'all':
                    # Check the folders
                    if os.path.exists(f"{data_folder}/{batch_name}"):
                        transformed_samples.extend(os.listdir(f"{data_folder}/{batch_name}"))
                    else:
                        print(f"Folder {data_folder}/{batch_name} does not exist")
                else:
                    numbers = re.findall(r'\d+', sample_range)
                    start, end = map(int, numbers) if len(numbers) == 2 else (int(numbers[0]), int(numbers[0]))
                    transformed_samples.extend([f"{batch_name}_{i:02d}" for i in range(start, end+1)])

        # Check if individual sample folders exist
        for sample in transformed_samples:
            batch_name = sample.rsplit('_',1)[0]
            if not os.path.exists(f"{data_folder}/{batch_name}/{sample}"):
                print(f"Folder {data_folder}/{batch_name}/{sample} does not exist")
                # remove this element from the list
                transformed_samples.remove(sample)

        # overwrite the samples with the transformed samples
        batches[plot_name]['samples'] = transformed_samples

    return batches

def plot_batch(plot_name: str, batch: dict) -> None:
    """ Plots the data for a batch of samples.
    
    Args:
        plot_name (str): name of the plot
        batch (dict): dict with 'samples' key containing list of samples to plot
            and any other plotting options e.g. group_by, palette, etc.

    TODO: make robust to missing data, raise warning instead of errors
    TODO: split into two functions, one to save the data and one to plot
    """
    with open('./config.json', encoding = 'utf-8') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]
    save_location = os.path.join(config['Batches Folder Path'],plot_name)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    samples = batch.get('samples')
    group_by = batch.get('group_by', None)
    palette = batch.get('palette', 'deep')
    cycle_dicts = []
    for sample in samples:
        # get the anaylsed data
        batch_name = sample.rsplit('_',1)[0]
        sample_folder = os.path.join(data_folder,batch_name,sample)
        try:
            analysed_file = next(
                f for f in os.listdir(sample_folder)
                if (f.startswith('cycles') and f.endswith('.json'))
            )
            with open(f'{sample_folder}/{analysed_file}', 'r', encoding='utf-8') as f:
                cycle_dict = json.load(f)
            if cycle_dict.get('Cycle') and cycle_dict['Cycle']:
                cycle_dicts.append(cycle_dict)
            else:
                print(f"No cycling data for {sample}")
                continue
        except StopIteration:
            # Handle the case where no file starts with 'cycles'
            print(f"No files starting with 'cycles' found in {sample_folder}.")
            continue
    assert len(cycle_dicts) > 0, "No cycling data found for any sample"
    cycle_df = pd.concat(
        [pd.DataFrame(d) for d in cycle_dicts],
    ).reset_index(drop=True)

    # Save the data
    cycle_df.to_excel(f'{save_location}/{plot_name}_data.xlsx',index=False)
    with open(f'{save_location}/{plot_name}_data.json','w',encoding='utf-8') as f:
        json.dump(cycle_dicts,f)

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
        d_ymin = max(0, 0.95*cycle_df['Specific Discharge Capacity (mAh/g)'].min())
        d_ymax = cycle_df['Specific Discharge Capacity (mAh/g)'].max()*1.05
    efficiency_ylim = batch.get('efficiency_ylim', None)
    if efficiency_ylim:
        e_ymin, e_ymax = sorted(efficiency_ylim)
    else:
        e_ymin = max(70, 0.95*cycle_df['Efficiency (%)'].min())
        e_ymax = min(101, cycle_df['Efficiency (%)'].max()*1.05)

    ### STRIP PLOT ###
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.stripplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
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
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.swarmplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
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
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.boxplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
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
    sorted_df['Jittered Cycle'] = sorted_df['Cycle'] + sorted_df['Offset']
    cycle_df = sorted_df.drop(columns=['Group_Number'])  # drop the temporary column

    # We usually want voltage as a categorical
    cycle_df["Max Voltage (V)"] = cycle_df["Max Voltage (V)"].astype(str)
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
        'Max Voltage (V)',
        'Cathode Mass (mg)',
        'Formation C/',
        'Cycle C/',
        'Electrolyte Name',
        'Actual N:P Ratio',
    ]
    hover_data = {col: True for col in hover_columns}
    hover_data['Cycle'] = False  # Exclude jittered 'Cycle' from hover data
    hover_template = (
        'Sample ID: %{customdata[0]}<br>'
        'Cycle: %{customdata[1]}<br><extra></extra>'
        'Max Voltage (V): %{customdata[2]}<br>'
        'Cathode Mass (mg): %{customdata[3]}<br>'
        'Formation C-rate: %{customdata[4]}<br>'
        'Cycle C-rate: %{customdata[5]}<br>'
        'Electrolyte: %{customdata[6]}<br>'
        'N:P Ratio: %{customdata[7]}'
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,vertical_spacing=0.1)
    fig.update_layout(
        template = 'ggplot2',
    )
    hex_colours = sns.color_palette(palette, num_combinations).as_hex()
    scatter1 = px.scatter(
        cycle_df,
        x='Jittered Cycle',
        y='Specific Discharge Capacity (mAh/g)',
        color=group_by,
        color_discrete_sequence=hex_colours,
        hover_data=hover_data,
    )
    for trace in scatter1.data:
        trace.hovertemplate = hover_template
        fig.add_trace(trace, row=1, col=1)

    scatter2 = px.scatter(
        cycle_df,
        x='Jittered Cycle',
        y='Efficiency (%)',
        color=group_by,
        color_discrete_sequence=hex_colours,
        hover_data=hover_data,
    )
    for trace in scatter2.data:
        trace.showlegend = False
        trace.hovertemplate = hover_template
        fig.add_trace(trace, row=2, col=1)

    fig.update_xaxes(title_text="Cycle", row=2, col=1)
    if discharge_ylim:
        ymin, ymax = sorted(discharge_ylim)
    else:
        ymin = max(0, 0.95*cycle_df['Specific Discharge Capacity (mAh/g)'].min())
        ymax = cycle_df['Specific Discharge Capacity (mAh/g)'].max()*1.05
    fig.update_yaxes(title_text="Specific Discharge<br>Capacity (mAh/g)", row=1, col=1, range=[ymin, ymax])
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
        fig.write_html(os.path.join(save_location,f'{plot_name}_interactive.html'))
    except PermissionError:
        print(
            "Permission error saving "
            f"{os.path.join(save_location,f'{plot_name}_interactive.html')}"
        )

def plot_all_batches(
        file_path: str= "K:/Aurora/cucumber/graph_config.yml"
    ) -> None:
    """ Plots all the batches according to the configuration file.

    Args:
        file_path (str): path to the yaml file containing the plotting config
            Defaults to "K:/Aurora/cucumber/graph_config.yml"
    
    Will search for analysed data in the processed snapshots folder and plot and
    save the capacity and efficiency vs cycle for each batch of samples.
    """
    batches = parse_sample_plotting_file(file_path)
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
