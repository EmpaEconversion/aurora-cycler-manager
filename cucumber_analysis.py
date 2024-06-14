import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3


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
