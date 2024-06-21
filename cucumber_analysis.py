import os
import re
import sqlite3
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_sample_plotting_file(file_path = "K:/Aurora/cucumber/graph_config.txt"):
    with open('config.json') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    with open('K:/Aurora/cucumber/graph_config.yml', 'r') as file:
        batches = yaml.safe_load(file)

    for plot_name, batch in batches.items():
        print(f"proecssing {plot_name}")
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
                if sample_range == 'all':
                    print(f"Processing all samples in {batch_name}")
                    # Check the folders
                    if os.path.exists(f"{data_folder}/{batch_name}"):
                        transformed_samples.extend(os.listdir(f"{data_folder}/{batch_name}"))   
                    else:
                        print(f"Folder {data_folder}/{batch_name} does not exist")
                else:
                    print(f"Processing range in {batch_name}")
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

def plot_batch(plot_name,batch):
    with open('config.json') as f:
        config = json.load(f)
    data_folder = config["Processed Snapshots Folder Path"]
    save_location = f"{config['Graphs Folder Path']}/{plot_name}"
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    samples = batch.pop('samples')
    group_by = batch.pop('group_by', None)
    kwargs = batch
    
    fig, ax = plt.subplots(len(samples),1,sharex=True,figsize=(6,1*len(samples)),dpi=300)
    all_cap_data = []
    for i,sample in enumerate(samples):
        # Check database
        with sqlite3.connect(config["Database Path"]) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT `Cathode Active Material Weight (mg)` FROM samples WHERE `Sample ID`='{sample}'")
            mass = cursor.fetchone()[0]

            cursor.execute(f"SELECT Payload FROM jobs WHERE `Sample ID`='{sample}'")
            payload = cursor.fetchall()
            payload = [json.loads(p[0]) for p in payload]

        max_V = 0
        formation_C = 0
        info = ""
        if group_by == "max_V":
            for item in payload:
                for method in item['method']:
                    voltage = method.get('limit_voltage_max',0)
                    if voltage > max_V:
                        max_V = voltage
            info = f", max V = {max_V}"
            kwargs["hue"] = "Max Voltage (V)"
        if group_by == "formation_C":
            for item in payload:
                if len(item['method']) == 4:
                    for method in item['method']:
                        if method['technique'] == 'loop' and method['n_gotos'] < 4: # it is probably formation
                            for m in item['method']:
                                if 'current' in m and 'C' in m['current']:
                                    formation_C = float(m['current'][2:])
                                    break
            info = f", formation C/{formation_C}"
            kwargs["hue"] = "Formation C"

        # V(T)
        batch = sample.rsplit('_',1)[0]
        sample_folder = f"{data_folder}/{batch}/{sample}"
        files = os.listdir(sample_folder)
        hdf5_files = [f for f in files if f.endswith('.h5')]
        hdf5_files.sort(key=lambda x: int(x[:-3].rsplit('-',1)[1]))  # order by number
        dfs=[]
        for f in hdf5_files:
            df = pd.read_hdf(f'{sample_folder}/{f}')
            dfs.append(df)
        big_df=pd.concat(dfs)
        ax[i].plot(pd.to_datetime(big_df["uts"], unit="s"),big_df["Ewe"])
        ax[i].set_ylabel('Voltage (V)')
        ax[i].set_title(sample+info, fontsize=8)
        ax[i].tick_params(axis='x', rotation=45)

        # CAPACITY
        capacity = get_capacities(dfs)
        capacity = capacity[capacity["Cycle"] < capacity["Cycle"].max()] # Remove last cycle
        cycle_df = pd.DataFrame({
            "Cycle": capacity["Cycle"].astype(int),
            "Charge Capacity (mAh)": capacity["Charge Capacity (mAh)"],
            "Specific Charge Capacity (mAh/g)": 1000*capacity["Charge Capacity (mAh)"]/mass,
            "Discharge Capacity (mAh)": capacity["Discharge Capacity (mAh)"],
            "Specific Discharge Capacity (mAh/g)": 1000*capacity["Discharge Capacity (mAh)"]/mass,
            "Efficiency (%)": capacity["Efficiency (%)"],
            "Max Voltage (V)": max_V,
            "Cathode Mass (mg)": mass,
            "Formation C": formation_C
        })
        # Append to all_data list
        all_cap_data.append(cycle_df)
    all_cap_data = pd.concat(all_cap_data)

    # Save individual plots
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_V(t).pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_V(t).pdf")
    plt.close('all')

    # Get number of cycles, only plot around 10 cycles
    n_cycles = all_cap_data["Cycle"].max()
    if n_cycles > 10:
        cycles_to_plot = [1] + list(range(0, n_cycles, n_cycles // 10))[1:]
    else:
        cycles_to_plot = list(range(1, n_cycles + 1))
    
    plot_data = all_cap_data[all_cap_data["Cycle"].isin(cycles_to_plot)]

    # Strip plot capacity
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.stripplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
        size=3,
        edgecolor='k',
        **kwargs,
    )
    sns.stripplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        size=3,
        edgecolor='k',
        **kwargs,
    )
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_strip.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_strip.pdf")
    plt.close('all')

    # Swarm plot capacity
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
        **kwargs,
    )
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_swarm.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_swarm.pdf")
    plt.close('all')

    # Violin plot capacity
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.violinplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
        **kwargs,
    )
    sns.violinplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        **kwargs,
    )
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_violin.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_violin.pdf")
    plt.close('all')

    # Box plot capacity
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(8,5),dpi=300)
    ax[0].set_ylabel("Discharge\nCapacity (mAh/g)")
    sns.boxplot(
        ax=ax[0],
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
        fill=False,
        **kwargs,
    )
    sns.boxplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        fill=False,
        **kwargs,
    )
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
        data=plot_data,
        x="Cycle",
        y="Specific Discharge Capacity (mAh/g)",
        **kwargs,
    )
    sns.pointplot(
        ax=ax[1],
        data=plot_data,
        x="Cycle",
        y="Efficiency (%)",
        **kwargs,
    )
    fig.tight_layout()
    try:
        fig.savefig(f'{save_location}/{plot_name}_Capacity_point.pdf',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{plot_name}_Capacity_point.pdf")
    plt.close('all')

def plot_batches_from_file(file_path = "K:/Aurora/cucumber/graph_config.yml"):
    batches = parse_sample_plotting_file(file_path)
    with open('config.json') as f:
        config = json.load(f)
    for plot_name, batch in batches.items():
        plot_batch(plot_name,batch)

def plot_cycles(folder,save_location,sampleid):
    files = os.listdir(folder)
    hdf5_files = [f for f in files if f.endswith('.h5')]
    # order by number
    hdf5_files.sort(key=lambda x: int(x[:-3].rsplit('-',1)[1]))
    dfs=[]
    for f in hdf5_files:
        df = pd.read_hdf(f'{folder}\\{f}')
        dfs.append(df)
    big_df=pd.concat(dfs)
    fig, ax = plt.subplots(figsize=(6,4),dpi=72)
    plt.plot(pd.to_datetime(big_df["uts"], unit="s"),big_df["Ewe"])
    plt.ylabel('Voltage (V)')
    plt.xticks(rotation=45)
    plt.title(sampleid)
    fig.savefig(f'{save_location}/{sampleid}_V(t).png',bbox_inches='tight')
    plt.close()

    capacity = get_capacities(dfs)
    fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,4),dpi=72)
    ax[0].plot(capacity['Cycle'][:-1],capacity['Discharge Capacity (mAh)'][:-1],'.-')
    ax[1].plot(capacity['Cycle'][:-1],capacity['Efficiency (%)'][:-1],'.-')
    ax[0].set_ylabel('Discharge Capacity (mAh)')
    ax[1].set_ylabel('Efficiency (%)')
    ax[1].set_xlabel('Cycle')
    ax[0].set_title(sampleid)
    try:
        fig.savefig(f'{save_location}/{sampleid}_Capacity.png',bbox_inches='tight')
    except PermissionError:
        print(f"Permission error saving {save_location}/{sampleid}_Capacity.png")
    plt.close()

def plot_all_samples(snapshot_folder=None):
    with open('config.json') as f:
        config = json.load(f)
    if not snapshot_folder:
        snapshot_folder = config["Processed Snapshots Folder Path"]
    graph_folder = config["Graphs Folder Path"]
    for batch_folder in os.listdir(snapshot_folder):
        for sample_folder in os.listdir(f'{snapshot_folder}/{batch_folder}'):
            source_folder = f'{snapshot_folder}/{batch_folder}/{sample_folder}'
            save_folder = f"{graph_folder}/{batch_folder}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plot_cycles(source_folder,save_folder,sample_folder)

def get_capacities(dfs,voltage_lower_cutoff=0,plot_cycle=0):
    for i,df in enumerate(dfs):
        df["job_number"] = i
    df = pd.concat(dfs)
    df["dt (s)"] = np.concatenate([df["uts"].values[1:] - df["uts"].values[:-1],[0]])
    df["Iavg (A)"] = np.concatenate([(df["I"].values[1:] + df["I"].values[:-1]) / 2,[0]])
    df["dQ (mAh)"] = 1e3 * df["Iavg (A)"] * df["dt (s)"] / 3600

    job_numbers = df['job_number'].unique()
    cycle_numbers = df['cycle_number'].unique()
    loop_numbers = df['loop_number'].unique()

    charge_capacity_mAh = []
    discharge_capacity_mAh = []
    cycle = []
    i=1

    for job_number in job_numbers:
        for cycle_number in cycle_numbers:
            for loop_number in loop_numbers:
                # get the data for the current cycle and loop
                charge_data = df[
                    (df['job_number'] == job_number) &
                    (df['cycle_number'] == cycle_number) & 
                    (df['loop_number'] == loop_number) & 
                    (df['Iavg (A)'] > 0) &
                    (df['Ewe'] > voltage_lower_cutoff)
                ]
                discharge_data = df[
                    (df['job_number'] == job_number) &
                    (df['cycle_number'] == cycle_number) &
                    (df['loop_number'] == loop_number) &
                    (df['Iavg (A)'] < 0) &
                    (df['Ewe'] > voltage_lower_cutoff)
                ]
                # Check that there is enough data for both charge and discharge
                if len(charge_data)>10 and len(discharge_data)>10:
                    charge_capacity_mAh.append(charge_data['dQ (mAh)'].sum())
                    discharge_capacity_mAh.append(-discharge_data['dQ (mAh)'].sum())
                    cycle.append(i)
                    if i == plot_cycle:
                        plt.plot(charge_data["uts"],charge_data['Iavg (A)'],'.-')
                        plt.plot(discharge_data["uts"],discharge_data['Iavg (A)'],'.-')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Current (A)')
                        plt.show()
                    i+=1
    output_df = pd.DataFrame(
        {
        'Cycle': cycle,
        'Charge Capacity (mAh)': charge_capacity_mAh,
        'Discharge Capacity (mAh)': discharge_capacity_mAh,
        }
    )
    output_df['Efficiency (%)'] = output_df['Discharge Capacity (mAh)']/output_df['Charge Capacity (mAh)']*100
    return output_df

def analyse_cycles(sampleid,dfs,voltage_lower_cutoff=0,plot_cycle=0):
    """ Given a sampleid, and list of dataframes, merge and calculate the capacity of each cycle"""
    with open('config.json') as f:
        config = json.load(f)
    # Get mass and payload from database
    with sqlite3.connect(config["Database Path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT `Cathode Active Material Weight (mg)` FROM samples WHERE `Sample ID`='{sampleid}'")
        mass_g = cursor.fetchone()[0]/1000
        cursor.execute(f"SELECT Payload FROM jobs WHERE `Sample ID`='{sampleid}'")
        payload = cursor.fetchall()
        payload = [json.loads(p[0]) for p in payload]
    for item in payload:
        for method in item['method']:
            if method['technique'] == 'loop' and method['n_gotos'] <= 5: # it is probably formation
                for m in item['method']:
                    if 'current' in m and 'C' in m['current']:
                        formation_C = m['current']
                        continue
            if method['technique'] == 'loop' and method['n_gotos'] > 5: # it is probably cycling
                for m in item['method']:
                    if 'current' in m and 'C' in m['current']:
                        cycle_C = m['current']
                        continue
    
    for i,df in enumerate(dfs):
        df["job_number"] = i
    df = pd.concat(dfs)
    df["dt (s)"] = np.concatenate([df["uts"].values[1:] - df["uts"].values[:-1],[0]])
    df["Iavg (A)"] = np.concatenate([(df["I"].values[1:] + df["I"].values[:-1]) / 2,[0]])
    df["dQ (mAh)"] = 1e3 * df["Iavg (A)"] * df["dt (s)"] / 3600

    job_numbers = df['job_number'].unique()
    cycle_numbers = df['cycle_number'].unique()
    loop_numbers = df['loop_number'].unique()

    i=1

    for job_number in job_numbers:
        for cycle_number in cycle_numbers:
            for loop_number in loop_numbers:
                # get the data for the current cycle and loop
                charge_data = df[
                    (df['job_number'] == job_number) &
                    (df['cycle_number'] == cycle_number) & 
                    (df['loop_number'] == loop_number) & 
                    (df['Iavg (A)'] > 0) &
                    (df['Ewe'] > voltage_lower_cutoff)
                ]
                discharge_data = df[
                    (df['job_number'] == job_number) &
                    (df['cycle_number'] == cycle_number) &
                    (df['loop_number'] == loop_number) &
                    (df['Iavg (A)'] < 0) &
                    (df['Ewe'] > voltage_lower_cutoff)
                ]
                # Check that there is enough data for both charge and discharge
                if len(charge_data)>10 and len(discharge_data)>10:
                    charge_capacity_mAh = charge_data['dQ (mAh)'].sum()
                    discharge_capacity_mAh = -discharge_data['dQ (mAh)'].sum()
                    if i in plot_cycle:
                        plt.plot(charge_data["uts"],charge_data['Iavg (A)'],'.-')
                        plt.plot(discharge_data["uts"],discharge_data['Iavg (A)'],'.-')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Current (A)')
                        plt.show()
                    cycle_df = pd.DataFrame({
                        'Cycle': i,
                        'Charge Capacity (mAh)': charge_capacity_mAh,
                        'Discharge Capacity (mAh)': discharge_capacity_mAh,
                        'Efficiency (%)': discharge_capacity_mAh/charge_capacity_mAh*100,
                        'Specific Charge Capacity (mAh/g)': charge_capacity_mAh/mass_g,
                        'Specific Discharge Capacity (mAh/g)': discharge_capacity_mAh/mass_g,
                        'Cathode Mass (g)': mass_g,
                        'Formation C': formation_C,
                        'Cycle C': cycle_C,
                    })
                    dfs.append(cycle_df)
                    i+=1
                    
    cycle_df = pd.concat(dfs)
    
    return df, cycle_df