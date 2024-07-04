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
from typing import List
from datetime import datetime
import json
import yaml
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
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
            with open('config.json', encoding = 'utf-8') as f:
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
    with open('config.json') as f:
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
    ) -> pd.DataFrame:
    """ Take multiple dataframes, merge and analyse the cycling data.
    
    Args:
        h5_files (List[str]): list of paths to the hdf5 files
        voltage_lower_cutoff (float, optional): lower cutoff for voltage data
        voltage_upper_cutoff (float, optional): upper cutoff for voltage data
        save_files (bool, optional): whether to save the output files
            files will be saved in the same folder as the first input file
    
    TODO: Add save location as an argument.
    """

    # Get the metadata from the files
    dfs = []
    metadatas = []
    payloads = []
    sampleids = []
    job_starts = []
    for f in h5_files:
        dfs.append(pd.read_hdf(f))
        with h5py.File(f, 'r') as file:
            try:
                metadata=dict(file['cycling'].attrs)
                job_data = json.loads(metadata['job_data'])
                sample_data = json.loads(metadata['sample_data'])
                metadatas.append(metadata)
                sampleids.append(
                    sample_data['Sample ID']
                )
                job_starts.append(
                    datetime.strptime(job_data['Submitted'], '%Y-%m-%d %H:%M:%S')
                )
                payloads.append(json.loads(job_data['Payload']))
            except KeyError as exc:
                print(f"Metadata not found in {f}")
                raise KeyError from exc
    assert len(set(sampleids)) == 1, "All files must be from the same sample"
    order = np.argsort(job_starts)
    dfs = [dfs[i] for i in order]
    h5_files = [h5_files[i] for i in order]
    metadatas = [metadatas[i] for i in order]
    payloads = [payloads[i] for i in order]

    for i,df in enumerate(dfs):
        df["job_number"] = i
    df = pd.concat(dfs)
    df["dt (s)"] = np.concatenate([df["uts"].values[1:] - df["uts"].values[:-1],[0]])
    df["Iavg (A)"] = np.concatenate([(df["I"].values[1:] + df["I"].values[:-1]) / 2,[0]])
    df["dQ (mAh)"] = 1e3 * df["Iavg (A)"] * df["dt (s)"] / 3600

    # Extract useful information from the metadata
    sample_data = json.loads(metadatas[0]['sample_data'])
    mass_g = sample_data['Cathode Active Material Weight (mg)'] * 1e-3
    max_V = 0
    formation_C = 0 
    for payload in payloads:
        for method in payload['method']:
            voltage = method.get('limit_voltage_max',0)
            if voltage > max_V:
                max_V = voltage

            for payload in payloads:
                if len(payload['method']) <= 4:
                    for method in payload['method']:
                        if method['technique'] == 'loop' and method['n_gotos'] < 4: # it is probably formation
                            for m in payload['method']:
                                if 'current' in m and 'C' in m['current']:
                                    try:
                                        formation_C = float(m['current'][2:])
                                    except ValueError:
                                        print(f"Not a valid C-rate: {m['current']}")
                                        formation_C = 0
                                    break

    df = df.sort_values('uts')
    # Detect whenever job, cycle or loop changes
    # Necessary because the cycle number is not always recorded correctly so
    # the combination of job, cycle, loop is not necessarily unique
    df['group_id'] = (
        (df['loop_number'] < df['loop_number'].shift(1)) | 
        (df['cycle_number'] < df['cycle_number'].shift(1)) | 
        (df['job_number'] < df['job_number'].shift(1))
    ).cumsum()
    df['global_idx'] = df.groupby(['job_number', 'group_id', 'cycle_number', 'loop_number']).ngroup()
    cycle_dicts = []
    cycle = 1
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
        if len(charge_data)>10 and len(discharge_data)>10:
            charge_capacity_mAh = charge_data['dQ (mAh)'].sum()
            discharge_capacity_mAh = -discharge_data['dQ (mAh)'].sum()       
            cycle_dicts.append({
                'Cycle': cycle,
                'Charge Capacity (mAh)': charge_capacity_mAh,
                'Discharge Capacity (mAh)': discharge_capacity_mAh,
                'Efficiency (%)': discharge_capacity_mAh/charge_capacity_mAh*100,
                'Specific Charge Capacity (mAh/g)': charge_capacity_mAh/mass_g,
                'Specific Discharge Capacity (mAh/g)': discharge_capacity_mAh/mass_g,
                'Cathode Mass (g)': mass_g,
                'Max Voltage (V)': max_V,
                'Formation C': formation_C,
                # 'Cycle C': cycle_C, # TODO extract cycle C-rate
            })
            group_df['Cycle'] = cycle
            cycle += 1
    cycle_df = pd.DataFrame(cycle_dicts)

    if save_files:
        save_folder = os.path.dirname(h5_files[0])
        if not save_folder:
            save_folder = '.'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Saving the merged df is probably a waste of space
        cycle_df.to_hdf(
            f'{save_folder}/cycles.{sampleids[0]}.h5',
            key="cycles",
            complib="blosc",
            complevel=2
        )
    return df, cycle_df

def analyse_all_cycles() -> None:
    """ Analyse all cycles in the processed snapshots folder.
    
    TODO: Only analyse files with new data since last analysis
    """
    with open('config.json') as f:
        config = json.load(f)
    snapshot_folder = config["Processed Snapshots Folder Path"]
    for batch_folder in os.listdir(snapshot_folder):
        for sample_folder in os.listdir(f'{snapshot_folder}/{batch_folder}'):
            source_folder = os.path.join(snapshot_folder, batch_folder, sample_folder)
            h5_files = [os.path.join(source_folder,f) for f in os.listdir(source_folder) if (f.startswith('snapshot') and f.endswith('.h5'))]
            try:
                analyse_cycles(h5_files, save_files=True)
                print(f"Analysed {sample_folder}")
            except:
                print(f"Failed to analyse {sample_folder}")

def plot_sample(sample: str) -> None:
    """ Plot the data for a single sample.
    
    Will search for the sample in the processed snapshots folder and plot V(t) 
    and capacity(cycle).
    """
    batch = sample.rsplit('_',1)[0]
    with open('config.json') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]
    file_location = f"{data_folder}/{batch}/{sample}"
    save_location = f"{config['Graphs Folder Path']}/{batch}"
    
    # plot V(t)
    files = os.listdir(file_location)
    cycling_files = [f for f in files if (f.startswith('snapshot') and f.endswith('.h5'))]
    dfs = [pd.read_hdf(f'{file_location}/{f}') for f in cycling_files]
    df = pd.concat(dfs)
    df.sort_values('uts', inplace=True)
    fig, ax = plt.subplots(figsize=(6,4),dpi=72)
    plt.plot(pd.to_datetime(df["uts"], unit="s"),df["Ewe"])
    plt.ylabel('Voltage (V)')
    plt.xticks(rotation=45)
    plt.title(sample)
    fig.savefig(f'{save_location}/{sample}_V(t).png',bbox_inches='tight')

    # plot capacity
    analysed_file = next(f for f in files if f.startswith('cycles'))
    cycle_df = pd.read_hdf(f'{file_location}/{analysed_file}')
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,4),dpi=72)
    ax[0].plot(cycle_df['Cycle'][:-1],cycle_df['Discharge Capacity (mAh)'][:-1],'.-')
    ax[1].plot(cycle_df['Cycle'][:-1],cycle_df['Efficiency (%)'][:-1],'.-')
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
    with open('config.json') as f:
        config = json.load(f)
    if not snapshot_folder:
        snapshot_folder = config["Processed Snapshots Folder Path"]
    for batch_folder in os.listdir(snapshot_folder):
        for sample in os.listdir(f'{snapshot_folder}/{batch_folder}'):
            plot_sample(sample)

def parse_sample_plotting_file(file_path: str = "K:/Aurora/cucumber/graph_config.yml") -> dict:
    """ Reads the graph config file and returns a dictionary of the batches to plot.
    
    Args: file_path (str): path to the yaml file containing the plotting configuration
        Defaults to "K:/Aurora/cucumber/graph_config.yml"
    
    Returns: dict: dictionary of the batches to plot
        Dictionary contains the plot name as the key and a dictionary of the batch details as the
        value. Batch dict contains the samples to plot and any other plotting options.
    """
    with open('config.json') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]

    with open(file_path, 'r') as file:
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
    """
    with open('config.json') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]
    save_location = os.path.join(config['Graphs Folder Path'],plot_name)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    samples = batch.pop('samples')
    group_by = batch.pop('group_by', None)
    kwargs = batch
    cycle_dfs = []
    for i,sample in enumerate(samples):
        # get the anaylsed data
        batch = sample.rsplit('_',1)[0]
        sample_folder = os.path.join(data_folder,batch,sample)
        try:
            analysed_file = next(f for f in os.listdir(sample_folder) if f.startswith('cycles'))
            cycle_df = pd.read_hdf(f'{sample_folder}/{analysed_file}')
            if cycle_df.empty:
                print(f"Empty dataframe for {sample}")
                continue
            ncycles = cycle_df['Cycle'].max()
            print(f"{sample} has {ncycles} cycles")
            cycle_dfs.append(cycle_df)
        except StopIteration:
            # Handle the case where no file starts with 'cycles', e.g., by logging, raising a custom error, or setting analysed_file to None
            print(f"No files starting with 'cycles' found in {sample_folder}.")
        
    cycle_df = pd.concat(cycle_dfs)

    # Get the 90th percentile cycle number
    n_cycles = max(cycle_df["Cycle"])
    if n_cycles > 15:
        cycles_to_plot = [1] + list(range(0, n_cycles, n_cycles // 15))[1:]
    else:
        cycles_to_plot = list(range(1, n_cycles + 1))
    plot_data = cycle_df[cycle_df["Cycle"].isin(cycles_to_plot)]

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
        hue = group_by,
        **kwargs,
    )
    sns.stripplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        size=3,
        edgecolor='k',
        hue = group_by,
        **kwargs,
    )
    ymin, ymax = ax[1].get_ylim()
    ymin = max(70,ymin)
    ymax = min(105,ymax)
    ax[1].set_ylim(ymin, ymax)
    current_bottom, current_top = ax[1].get_ylim()
    new_bottom = 0 if current_bottom < 0 else current_bottom
    new_top = 105 if current_top > 105 else current_top
    ax[1].set_ylim(new_bottom, new_top)

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
        hue = group_by,
        **kwargs,
    )
    sns.swarmplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        size=3,
        dodge=True,
        edgecolor='k',
        hue = group_by,
        **kwargs,
    )
    ymin, ymax = ax[1].get_ylim()
    ymin = max(70,ymin)
    ymax = min(105,ymax)
    ax[1].set_ylim(ymin, ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_swarm.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_swarm.pdf")
    plt.close('all')

    ### Violin plot ###
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.violinplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
        hue = group_by,
        **kwargs,
    )
    sns.violinplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        hue = group_by,
        **kwargs,
    )
    ymin, ymax = ax[1].get_ylim()
    ymin = max(70,ymin)
    ymax = min(105,ymax)
    ax[1].set_ylim(ymin, ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_violin.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_violin.pdf")
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
        hue = group_by,
        **kwargs,
    )
    sns.boxplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        fill=False,
        hue = group_by,
        **kwargs,
    )
    ymin, ymax = ax[1].get_ylim()
    ymin = max(70,ymin)
    ymax = min(105,ymax)
    ax[1].set_ylim(ymin, ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_box.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_box.pdf")
    plt.close('all')

    # Point plot capacity
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.pointplot(
        ax=ax[0],
        data=cycle_df,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
        linewidth=0.7,
        capsize=0.7,
        estimator='median',
        markersize=3,
        hue = group_by,
        **kwargs,
    )
    sns.pointplot(
        ax=ax[1],
        data=cycle_df,
        x="Cycle",
        y="Efficiency (%)",
        hue = group_by,
        linewidth=0.7,
        capsize=0.7,
        estimator='median',
        markersize=3,
        **kwargs,
    )
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10)) 
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10)) 
    ymin, ymax = ax[1].get_ylim()
    ymin = max(70,ymin)
    ymax = min(105,ymax)
    ax[1].set_ylim(ymin, ymax)
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_point.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_point.pdf")
    plt.close('all')

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
        plot_batch(plot_name,batch)

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
