"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Functions used for parsing, analysing and plotting.

Parsing:
Contains functions for converting raw jsons from tomato to pandas dataframes,
which can be saved to compressed hdf5 files.

Also includes functions for analysing the cycling data, extracting the
charge, discharge and efficiency of each cycle, and links this to various
quantities extracted from the cycling, such as C-rate and max voltage, and
from the sample database such as cathode active material mass.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import pytz
import yaml
from tsdownsample import MinMaxLTTBDownsampler

from aurora_cycler_manager.config import CONFIG
from aurora_cycler_manager.database_funcs import get_sample_data
from aurora_cycler_manager.utils import c_to_float, run_from_sample
from aurora_cycler_manager.version import __url__, __version__

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN axis encountered")

# Metadata that gets copied in the json data file for more convenient access
SAMPLE_METADATA_TO_DATA = [
    "N:P ratio",
    "Anode type",
    "Cathode type",
    "Anode active material mass (mg)",
    "Cathode active material mass (mg)",
    "Electrolyte name",
    "Electrolyte description",
    "Electrolyte amount (uL)",
    "Rack position",
    "Label",
]


def _sort_times(start_times: list | np.ndarray, end_times: list | np.ndarray) -> np.ndarray:
    """Sort by start time, if equal only keep the longest."""
    start_times = np.array(start_times)
    end_times = np.array(end_times)

    # reverse sort by end time, then sort by start time
    sorted_indices = np.lexsort((np.array(end_times) * -1, np.array(start_times)))
    start_times = start_times[sorted_indices]
    end_times = end_times[sorted_indices]

    # remove duplicate start times, leaving only the first element = the latest end time
    unique_mask = np.concatenate(([True], start_times[1:] != start_times[:-1]))
    return sorted_indices[unique_mask]


def combine_jobs(
    job_files: list[Path],
) -> tuple[pd.DataFrame, dict]:
    """Read multiple job files and return a single dataframe.

    Merges the data, identifies cycle numbers and changes column names.
    Columns are now 'V (V)', 'I (A)', 'uts', 'dt (s)', 'Iavg (A)',
    'dQ (mAh)', 'Step', 'Cycle'.

    Args:
        job_files (List[str]): list of paths to the job files

    Returns:
        pd.DataFrame: DataFrame containing the cycling data
        dict: metadata from the files

    """
    # Get the metadata from the files
    dfs = []
    metadatas = []
    sampleids = []
    for f in job_files:
        if f.name.endswith(".h5"):
            dfs.append(pd.read_hdf(f))
            with h5py.File(f, "r") as h5f:
                metadata = json.loads(h5f["metadata"][()])
                metadatas.append(metadata)
                sampleids.append(
                    metadata.get("sample_data", {}).get("Sample ID", ""),
                )
        elif f.name.endswith(".json.gz"):
            with gzip.open(f, "rt", encoding="utf-8") as file:
                data = json.load(file)
                dfs.append(pd.DataFrame(data["data"]))
                metadatas.append(data["metadata"])
                sampleids.append(
                    data["metadata"].get("sample_data", {}).get("Sample ID", ""),
                )
    if len(set(sampleids)) != 1:
        msg = "All files must be from the same sample"
        raise ValueError(msg)
    dfs = [df for df in dfs if "uts" in df.columns and not df["uts"].empty]
    if not dfs:
        msg = "No 'uts' column found in any of the files"
        raise ValueError(msg)
    start_times = [df["uts"].iloc[0] for df in dfs]
    end_tmes = [df["uts"].iloc[-1] for df in dfs]
    order = _sort_times(start_times, end_tmes)
    dfs = [dfs[i] for i in order]
    job_files = [job_files[i] for i in order]
    metadatas = [metadatas[i] for i in order]

    for i, df in enumerate(dfs):
        df["job_number"] = i
    df = pd.concat(dfs)
    df = df.sort_values("uts")
    # rename columns
    df = df.rename(
        columns={
            "Ewe": "V (V)",
            "I": "I (A)",
            "uts": "uts",
        },
    )
    df["dt (s)"] = np.concatenate([[0], df["uts"].to_numpy()[1:] - df["uts"].to_numpy()[:-1]])
    df["Iavg (A)"] = np.concatenate([[0], (df["I (A)"].to_numpy()[1:] + df["I (A)"].to_numpy()[:-1]) / 2])
    df["dQ (mAh)"] = 1e3 * df["Iavg (A)"] * df["dt (s)"] / 3600
    df.loc[df["dt (s)"] > 600, "dQ (mAh)"] = 0
    if "loop_number" not in df.columns:
        df["loop_number"] = 0

    df["group_id"] = (
        (df["loop_number"].shift(-1) < df["loop_number"])
        | (df["cycle_number"].shift(-1) < df["cycle_number"])
        | (df["job_number"].shift(-1) < df["job_number"])
    ).cumsum()
    df["Step"] = df.groupby(["job_number", "group_id", "cycle_number", "loop_number"]).ngroup()
    df = df.drop(columns=["job_number", "group_id", "cycle_number", "loop_number", "index"], errors="ignore")
    df["Cycle"] = 0
    cycle = 1
    for step, group_df in df.groupby("Step"):
        # To be considered a cycle (subject to change):
        # - more than 10 data points
        # - more than 5 charging points
        # - more than 5 discharging points
        if len(group_df) > 10 and sum(group_df["I (A)"] > 0) > 5 and sum(group_df["I (A)"] < 0) > 5:
            df.loc[df["Step"] == step, "Cycle"] = cycle
            cycle += 1

    # Add provenance to the metadatas
    timezone = pytz.timezone(CONFIG.get("Time zone", "Europe/Zurich"))
    # Replace sample data with latest from database
    sample_data = get_sample_data(sampleids[0])
    # Merge glossary dicts
    glossary = {}
    for g in [m.get("glossary", {}) for m in metadatas]:
        glossary.update(g)
    metadata = {
        "provenance": {
            "aurora_metadata": {
                "data_merging": {
                    "job_files": [str(f) for f in job_files],
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "analysis.combine_jobs",
                    "datetime": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %z"),
                },
            },
            "original_file_provenance": {str(f): m["provenance"] for f, m in zip(job_files, metadatas)},
        },
        "sample_data": sample_data,
        "job_data": [m.get("job_data", {}) for m in metadatas],
        "glossary": glossary,
    }

    return df, metadata


def analyse_cycles(
    job_files: list[Path],
    voltage_lower_cutoff: float = 0,
    voltage_upper_cutoff: float = 5,
    save_cycle_dict: bool = False,
    save_merged_hdf: bool = False,
    save_merged_jsongz: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    """Take multiple dataframes, merge and analyse the cycling data.

    Args:
        job_files (List[Path]): list of paths to the json.gz job files
        voltage_lower_cutoff (float, optional): lower cutoff for voltage data
        voltage_upper_cutoff (float, optional): upper cutoff for voltage data
        save_cycle_dict (bool, optional): save the cycle_dict as a json file
        save_merged_hdf (bool, optional): save the merged dataframe as an hdf5 file
        save_merged_jsongz (bool, optional): save the merged dataframe as a json.gz file

    Returns:
        pd.DataFrame: DataFrame containing the cycling data
        dict: dictionary containing the cycling analysis
        dict: metadata from the files

    TODO: Add save location as an argument.

    """
    df, metadata = combine_jobs(job_files)

    # update metadata
    timezone = pytz.timezone(CONFIG.get("Time zone", "Europe/Zurich"))
    metadata.setdefault("provenance", {}).setdefault("aurora_metadata", {})
    metadata["provenance"]["aurora_metadata"].update(
        {
            "analysis": {
                "repo_url": __url__,
                "repo_version": __version__,
                "method": "analysis.analyse_cycles",
                "datetime": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %z"),
            },
        },
    )

    sample_data = metadata.get("sample_data", {})
    sampleid = sample_data.get("Sample ID", None)
    job_data = metadata.get("job_data", None)
    snapshot_status = job_data[-1].get("Snapshot status", None) if job_data else None
    snapshot_pipeline = job_data[-1].get("Pipeline", None) if job_data else None
    last_snapshot = job_data[-1].get("Last snapshot", None) if job_data else None

    # Extract useful information from the metadata
    mass_mg = sample_data.get("Cathode active material mass (mg)", np.nan)

    max_V = 0.0
    formation_C = 0.0
    cycle_C = 0.0

    # TODO: separate formation and cycling C-rates and voltages, get C-rates for mpr and neware

    # TOMATO DATA
    pipeline = None
    status = None
    if job_data:
        job_types = [j.get("job_type", None) for j in job_data]
        if all(jt == job_types[0] for jt in job_types):
            job_type = job_types[0]
        else:
            msg = "Different job types found in job data"
            raise ValueError(msg)

        # tomato 0.2.3 using biologic driver
        if job_type == "tomato_0_2_biologic":
            payloads = [j.get("Payload", []) for j in job_data]
            with sqlite3.connect(CONFIG["Database path"]) as conn:
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
                for method in payload.get("method", []):
                    voltage = method.get("limit_voltage_max", 0)
                    max_V = max(voltage, max_V)

            for payload in payloads:
                for method in payload.get("method", []):
                    if method.get("technique", None) == "loop":
                        if method["n_gotos"] < 4:  # it is probably formation
                            for m in payload.get("method", []):
                                if "current" in m and "C" in m["current"]:
                                    try:
                                        formation_C = c_to_float(m["current"])
                                    except ValueError:
                                        print(f"Not a valid C-rate: {m['current']}")
                                        formation_C = 0
                                    break
                        if method.get("n_gotos", 0) > 10:  # it is probably cycling
                            for m in payload.get("method", []):
                                if "current" in m and "C" in m["current"]:
                                    try:
                                        cycle_C = c_to_float(m["current"])
                                    except ValueError:
                                        print(f"Not a valid C-rate: {m['current']}")
                                        cycle_C = 0
                                    break

        # ec-lab mpr
        elif job_type == "eclab_mpr":
            for m in job_data:
                params = m.get("params", [])
                if isinstance(params, list):
                    for param in params:
                        if isinstance(param, dict):
                            V = round(param.get("EM", 0), 3)
                            max_V = max(V, max_V)

        # Neware xlsx
        elif job_type in ("neware_xlsx", "neware_ndax"):
            for m in job_data:
                V = max(float(step.get("Voltage (V)", 0)) for step in m["Payload"])
                max_V = max(V, max_V)

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
    for _, group_df in df.groupby("Step"):
        cycle = group_df["Cycle"].iloc[0]
        if cycle <= 0:
            if len(group_df) > 10:
                started_charge = False
                started_discharge = False
            continue
        charge_data = group_df[
            (group_df["Iavg (A)"] > 0)
            & (group_df["V (V)"] > voltage_lower_cutoff)
            & (group_df["V (V)"] < voltage_upper_cutoff)
            & (group_df["dt (s)"] < 600)
        ]
        discharge_data = group_df[
            (group_df["Iavg (A)"] < 0)
            & (group_df["V (V)"] > voltage_lower_cutoff)
            & (group_df["V (V)"] < voltage_upper_cutoff)
            & (group_df["dt (s)"] < 600)
        ]
        # Only consider cycles with more than 10 data points
        started_charge = len(charge_data) > 10
        started_discharge = len(discharge_data) > 10

        if started_charge and started_discharge:
            charge_capacity_mAh.append(charge_data["dQ (mAh)"].sum())
            charge_avg_V.append((charge_data["V (V)"] * charge_data["dQ (mAh)"]).sum() / charge_data["dQ (mAh)"].sum())
            charge_energy_mWh.append((charge_data["V (V)"] * charge_data["dQ (mAh)"]).sum())
            charge_avg_I.append(
                (charge_data["Iavg (A)"] * charge_data["dQ (mAh)"]).sum() / charge_data["dQ (mAh)"].sum(),
            )
            discharge_capacity_mAh.append(-discharge_data["dQ (mAh)"].sum())
            discharge_avg_V.append(
                (discharge_data["V (V)"] * discharge_data["dQ (mAh)"]).sum() / discharge_data["dQ (mAh)"].sum(),
            )
            discharge_energy_mWh.append((-discharge_data["V (V)"] * discharge_data["dQ (mAh)"]).sum())
            discharge_avg_I.append(
                (-discharge_data["Iavg (A)"] * discharge_data["dQ (mAh)"]).sum() / discharge_data["dQ (mAh)"].sum(),
            )

    formation_cycle_count = 3
    initial_cycle = formation_cycle_count + 2

    formed = len(charge_capacity_mAh) >= initial_cycle
    # A row is added if charge data is complete and discharge started
    # Last dict may have incomplete discharge data
    if snapshot_status in ["r", "cd", "ce"]:  # This only works for tomato, harvesters will assume last cycle complete
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
    # TODO: add datetime of every cycle
    cycle_dict = {
        "Sample ID": sampleid,
        "Cycle": list(range(1, len(charge_capacity_mAh) + 1)),
        "Charge capacity (mAh)": charge_capacity_mAh,
        "Discharge capacity (mAh)": discharge_capacity_mAh,
        "Efficiency (%)": [100 * d / c for d, c in zip(discharge_capacity_mAh, charge_capacity_mAh)],
        "Specific charge capacity (mAh/g)": [c / (mass_mg * 1e-3) for c in charge_capacity_mAh],
        "Specific discharge capacity (mAh/g)": [d / (mass_mg * 1e-3) for d in discharge_capacity_mAh],
        "Normalised discharge capacity (%)": [
            100 * d / discharge_capacity_mAh[initial_cycle - 1] for d in discharge_capacity_mAh
        ]
        if formed
        else None,
        "Normalised discharge energy (%)": [
            100 * d / discharge_energy_mWh[initial_cycle - 1] for d in discharge_energy_mWh
        ]
        if formed
        else None,
        "Charge average voltage (V)": charge_avg_V,
        "Discharge average voltage (V)": discharge_avg_V,
        "Delta V (V)": [c - d for c, d in zip(charge_avg_V, discharge_avg_V)],
        "Charge average current (A)": charge_avg_I,
        "Discharge average current (A)": discharge_avg_I,
        "Charge energy (mWh)": charge_energy_mWh,
        "Discharge energy (mWh)": discharge_energy_mWh,
        "Max voltage (V)": max_V,
        "Formation C": formation_C,
        "Cycle C": cycle_C,
    }

    # Add other columns from sample table to cycle_dict
    for col in SAMPLE_METADATA_TO_DATA:
        cycle_dict[col] = sample_data.get(col, None)

    # Calculate additional quantities from cycling data and add to cycle_dict
    if not cycle_dict["Cycle"]:
        print(f"No cycles found for {sampleid}")
    elif len(cycle_dict["Cycle"]) == 1 and not complete:
        print(f"No complete cycles found for {sampleid}")
    else:  # Analyse the cycling data
        last_idx = -1 if complete else -2

        cycle_dict["First formation efficiency (%)"] = cycle_dict["Efficiency (%)"][0]
        cycle_dict["First formation specific discharge capacity (mAh/g)"] = cycle_dict[
            "Specific discharge capacity (mAh/g)"
        ][0]
        cycle_dict["Initial specific discharge capacity (mAh/g)"] = (
            cycle_dict["Specific discharge capacity (mAh/g)"][initial_cycle - 1] if formed else None
        )
        cycle_dict["Initial efficiency (%)"] = cycle_dict["Efficiency (%)"][initial_cycle - 1] if formed else None
        cycle_dict["Capacity loss (%)"] = (
            100 - cycle_dict["Normalised discharge capacity (%)"][last_idx] if formed else None
        )
        cycle_dict["Last specific discharge capacity (mAh/g)"] = cycle_dict["Specific discharge capacity (mAh/g)"][
            last_idx
        ]
        cycle_dict["Last efficiency (%)"] = cycle_dict["Efficiency (%)"][last_idx]
        cycle_dict["Formation average voltage (V)"] = (
            np.mean(cycle_dict["Charge average voltage (V)"][: initial_cycle - 1]) if formed else None
        )
        cycle_dict["Formation average current (A)"] = (
            np.mean(cycle_dict["Charge average current (A)"][: initial_cycle - 1]) if formed else None
        )
        cycle_dict["Initial delta V (V)"] = cycle_dict["Delta V (V)"][initial_cycle - 1] if formed else None

        # Calculate cycles to x% of initial discharge capacity
        def _find_first_element(arr: np.ndarray, start_idx: int) -> int | None:
            """Find first element in array that is 1 where at least 1 of the next 2 elements are also 1.

            Since cycles are 1-indexed and arrays are 0-indexed, this gives the first cycle BEFORE a condition is met.
            """
            if len(arr) - start_idx < 3:
                return None
            for i in range(start_idx, len(arr) - 2):
                if arr[i] == 0:
                    continue
                if arr[i + 1] == 1 or arr[i + 2] == 1:
                    return i
            return None

        pcents = [95, 90, 85, 80, 75, 70, 60, 50]
        norm = np.array(cycle_dict["Normalised discharge capacity (%)"])
        for pcent in pcents:
            cycle_dict[f"Cycles to {pcent}% capacity"] = (
                _find_first_element(
                    norm < pcent,
                    initial_cycle - 1,
                )
                if formed
                else None
            )
        norm = np.array(cycle_dict["Normalised discharge energy (%)"])
        for pcent in pcents:
            cycle_dict[f"Cycles to {pcent}% energy"] = (
                _find_first_element(
                    norm < pcent,
                    initial_cycle - 1,
                )
                if formed
                else None
            )

        cycle_dict["Run ID"] = run_from_sample(sampleid)

        # If assembly history is available, calculate times between steps
        assembly_history = sample_data.get("Assembly history", [])
        if isinstance(assembly_history, str):
            assembly_history = json.loads(assembly_history)
        if assembly_history and isinstance(assembly_history, list):
            job_start = df["uts"].iloc[0]
            press = next((step.get("uts", None) for step in assembly_history if step["Step"] == "Press"), None)
            electrolyte_ind = [i for i, step in enumerate(assembly_history) if step["Step"] == "Electrolyte"]
            if electrolyte_ind:
                first_electrolyte = next(
                    (step.get("uts", None) for step in assembly_history if step["Step"] == "Electrolyte"),
                    None,
                )
                history_after_electrolyte = assembly_history[max(electrolyte_ind) :]
                cover_electrolyte = next(
                    (
                        step.get("uts", None)
                        for step in history_after_electrolyte
                        if step["Step"] in ["Anode", "Cathode"]
                    ),
                    None,
                )
                cycle_dict["Electrolyte to press (s)"] = (
                    press - first_electrolyte if first_electrolyte and press else None
                )
                cycle_dict["Electrolyte to electrode (s)"] = (
                    cover_electrolyte - first_electrolyte if first_electrolyte and cover_electrolyte else None
                )
                cycle_dict["Electrode to protection (s)"] = job_start - cover_electrolyte if cover_electrolyte else None
            cycle_dict["Press to protection (s)"] = job_start - press if press else None

        # Update the database with some of the results
        flag = None
        job_complete = status and status.endswith("c")
        if pipeline:
            if not job_complete:
                if formed and cycle_dict["Capacity loss (%)"] > 20:
                    flag = "Cap loss"
                if cycle_dict["First formation efficiency (%)"] < 60:
                    flag = "Form eff"
                if formed and cycle_dict["Initial efficiency (%)"] < 50:
                    flag = "Init eff"
                if formed and cycle_dict["Initial specific discharge capacity (mAh/g)"] < 100:
                    flag = "Init cap"
            else:
                flag = "Complete"
        update_row = {
            "Pipeline": pipeline,
            "Status": status,
            "Flag": flag,
            "Number of cycles": int(max(cycle_dict["Cycle"])),
            "Capacity loss (%)": cycle_dict["Capacity loss (%)"],
            "Max voltage (V)": cycle_dict["Max voltage (V)"],
            "Formation C": cycle_dict["Formation C"],
            "Cycling C": cycle_dict["Cycle C"],
            "First formation efficiency (%)": cycle_dict["First formation efficiency (%)"],
            "Initial specific discharge capacity (mAh/g)": cycle_dict["Initial specific discharge capacity (mAh/g)"],
            "Initial efficiency (%)": cycle_dict["Initial efficiency (%)"],
            "Last specific discharge capacity (mAh/g)": cycle_dict["Last specific discharge capacity (mAh/g)"],
            "Last efficiency (%)": cycle_dict["Last efficiency (%)"],
            "Last analysis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Only add the following keys if they are not None, otherwise they set to NULL in database
            **({"Last snapshot": last_snapshot} if last_snapshot else {}),
            **({"Snapshot status": snapshot_status} if snapshot_status else {}),
            **({"Snapshot pipeline": snapshot_pipeline} if snapshot_pipeline else {}),
        }

        # round any floats to 3 decimal places
        for k, v in update_row.items():
            if isinstance(v, float):
                update_row[k] = round(v, 3)

        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            # insert a row with sampleid if it doesn't exist
            cursor.execute("INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)", (sampleid,))
            # update the row
            columns = ", ".join([f"`{k}` = ?" for k in update_row])
            cursor.execute(
                f"UPDATE results SET {columns} WHERE `Sample ID` = ?",
                (*update_row.values(), sampleid),
            )

    if save_cycle_dict or save_merged_hdf or save_merged_jsongz:
        save_folder = job_files[0].parent
        if save_cycle_dict:
            with (save_folder / f"cycles.{sampleid}.json").open("w", encoding="utf-8") as f:
                json.dump({"data": cycle_dict, "metadata": metadata}, f)
        if save_merged_hdf or save_merged_jsongz:
            df = df.drop(columns=["dt (s)", "Iavg (A)"])
        if save_merged_hdf:
            output_hdf5_file = f"{save_folder}/full.{sampleid}.h5"
            # change to 32 bit floats
            # for some reason the file becomes much larger with uts in 32 bit, so keep it as 64 bit
            for col in ["V (V)", "I (A)", "dQ (mAh)"]:
                if col in df.columns:
                    df[col] = df[col].astype(np.float32)
            df.to_hdf(
                output_hdf5_file,
                key="data",
                mode="w",
                complib="blosc",
                complevel=9,
            )
            with h5py.File(output_hdf5_file, "a") as f:
                f.create_dataset("metadata", data=json.dumps(metadata))
        if save_merged_jsongz:
            with gzip.open(save_folder / f"full.{sampleid}.json.gz", "wt", encoding="utf-8") as f:
                json.dump({"data": df.to_dict(orient="list"), "metadata": metadata}, f)
    return df, cycle_dict, metadata


def analyse_sample(sample: str) -> tuple[pd.DataFrame, dict, dict]:
    """Analyse a single sample.

    Will search for the sample in the processed snapshots folder and analyse the cycling data.

    """
    run_id = run_from_sample(sample)
    file_location = Path(CONFIG["Processed snapshots folder path"]) / run_id / sample
    # Prioritise .h5 files
    job_files = list(file_location.glob("snapshot.*.h5"))
    if not job_files:  # check if there are .json.gz files
        job_files = list(file_location.glob("snapshot.*.json.gz"))
    df, cycle_dict, metadata = analyse_cycles(
        job_files,
        save_cycle_dict=True,
        save_merged_hdf=True,
        save_merged_jsongz=False,
    )
    # also save a shrunk version of the file
    shrink_sample(sample)
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE results SET `Last analysis` = ? WHERE `Sample ID` = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sample),
        )
    return df, cycle_dict, metadata


def update_sample_metadata(sample_ids: str | list[str]) -> None:
    """Update "sample_data" in metadata of full.x.hdf5 and cycles.x.json files.

    Args:
        sample_ids: sample id or list of sample ids to update

    """
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    for sample_id in sample_ids:
        run_id = run_from_sample(sample_id)
        sample_folder = Path(CONFIG["Processed snapshots folder path"]) / run_id / sample_id
        # HDF5 full file
        hdf5_file = sample_folder / f"full.{sample_id}.h5"
        if not hdf5_file.exists():
            print(f"File {hdf5_file} not found")
            continue
        with h5py.File(hdf5_file, "a") as f:
            # check the keys data and metadata exist
            if "data" not in f or "metadata" not in f:
                print(f"File {hdf5_file} has incorrect format")
                continue
            metadata = json.loads(f["metadata"][()])
            sample_data = get_sample_data(sample_id)
            metadata["sample_data"] = sample_data
            f["metadata"][()] = json.dumps(metadata)
        # JSON cycles file
        json_file = sample_folder / f"cycles.{sample_id}.json"
        if not json_file.exists():
            print(f"File {json_file} not found")
            continue
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # check it has keys data and metadata
            if "data" not in data or "metadata" not in data:
                print(f"File {json_file} has incorrect format")
                continue
            data["metadata"]["sample_data"] = sample_data
            for col in SAMPLE_METADATA_TO_DATA:
                data["data"][col] = sample_data.get(col, None)
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(data, f)


def shrink_sample(sample_id: str) -> None:
    """Find the full.x.h5 file for the sample and save a lossy, compressed version."""
    run_id = run_from_sample(sample_id)
    file_location = Path(CONFIG["Processed snapshots folder path"]) / run_id / sample_id / f"full.{sample_id}.h5"
    if not file_location.exists():
        msg = f"File {file_location} not found"
        raise FileNotFoundError(msg)
    df = pd.read_hdf(file_location)
    # Only keep a few columns
    df = df[["V (V)", "I (A)", "uts", "dQ (mAh)", "Cycle"]]
    # Reduce precision of some columns
    for col in ["V (V)", "I (A)", "dQ (mAh)"]:
        df[col] = df[col].astype(np.float16)
    df["Cycle"] = df["Cycle"].astype(np.int16)

    # Use the LTTB downsampler to reduce the number of data points
    original_length = len(df)
    new_length = min(original_length, original_length // 20 + 1000, 50000)
    if new_length < 3:
        msg = f"Too few data points ({original_length}) to shrink {sample_id}"
        raise ValueError(msg)
    s_ds_V = MinMaxLTTBDownsampler().downsample(df["uts"], df["V (V)"], n_out=new_length)
    s_ds_I = MinMaxLTTBDownsampler().downsample(df["uts"], df["I (A)"], n_out=new_length)
    ind = np.sort(np.concatenate([s_ds_V, s_ds_I]))

    df["Q (mAh)"] = df["dQ (mAh)"].cumsum()

    df = df.iloc[ind]

    df["dQ (mAh)"] = df["Q (mAh)"].diff().fillna(0)
    df = df.drop(columns=["Q (mAh)"])

    # Save the new file
    new_file_location = file_location.with_name(f"shrunk.{sample_id}.h5")
    df.to_hdf(new_file_location, key="data", mode="w", complib="blosc", complevel=9)


def shrink_all_samples(sampleid_contains: str = "") -> None:
    """Shrink all samples in the processed snapshots folder.

    Args:
        sampleid_contains (str, optional): only shrink samples with this string in the sampleid

    """
    for batch_folder in Path(CONFIG["Processed snapshots folder path"]).iterdir():
        if batch_folder.is_dir():
            for sample in batch_folder.iterdir():
                if sampleid_contains and sampleid_contains not in sample.name:
                    continue
                try:
                    shrink_sample(sample.name)
                    print(f"Shrunk {sample.name}")
                except KeyError as e:
                    print(f"No metadata found for {sample.name}: {e}")
                except (ValueError, PermissionError, RuntimeError, FileNotFoundError) as e:
                    tb = traceback.format_exc()
                    print(f"Failed to analyse {sample.name} with error {e}\n{tb}")


def analyse_all_samples(
    sampleid_contains: str = "",
    mode: Literal["always", "new_data", "if_not_exists"] = "new_data",
) -> None:
    """Analyse all samples in the processed snapshots folder.

    Args: sampleid_contains (str, optional): only analyse samples with this
        string in the sampleid

    """
    if mode == "new_data":
        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID`, `Last snapshot`, `Last analysis` FROM results")
            results = cursor.fetchall()
        dtformat = "%Y-%m-%d %H:%M:%S"
        samples_to_analyse = [
            r[0]
            for r in results
            if r[0] and (not r[1] or not r[2] or datetime.strptime(r[1], dtformat) > datetime.strptime(r[2], dtformat))
        ]
    elif mode == "if_not_exists":
        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID` FROM results WHERE `Last analysis` IS NULL")
            results = cursor.fetchall()
        samples_to_analyse = [r[0] for r in results]

    for batch_folder in Path(CONFIG["Processed snapshots folder path"]).iterdir():
        if batch_folder.is_dir():
            for sample in batch_folder.iterdir():
                if sampleid_contains and sampleid_contains not in sample.name:
                    continue
                if mode != "always" and sample.name not in samples_to_analyse:
                    continue
                try:
                    analyse_sample(sample.name)
                except KeyError as e:
                    print(f"No metadata found for {sample.name}: {e}")
                except (ValueError, PermissionError, RuntimeError, FileNotFoundError) as e:
                    tb = traceback.format_exc()
                    print(f"Failed to analyse {sample.name} with error {e}\n{tb}")


def parse_sample_plotting_file(
    file_path: Path,
) -> dict:
    """Read the graph config file and returns a dictionary of the batches to plot.

    Args: file_path (str): path to the yaml file containing the plotting configuration

    Returns: dict: dictionary of the batches to plot
        Dictionary contains the plot name as the key and a dictionary of the batch details as the
        value. Batch dict contains the samples to plot and any other plotting options.

    TODO: Put the graph config location in the config file.

    """
    data_folder = Path(CONFIG["Processed snapshots folder path"])

    with file_path.open(encoding="utf-8") as file:
        batches = yaml.safe_load(file)

    for plot_name, batch in batches.items():
        samples = batch["samples"]
        transformed_samples = []
        for sample in samples:
            split_name = sample.split(" ", 1)
            if len(split_name) == 1:  # the batch is a single sample
                sample_id = sample
                run_id = run_from_sample(sample_id)
                transformed_samples.append(sample_id)
            else:
                run_id, sample_range = split_name
                if sample_range.strip().startswith("[") and sample_range.strip().endswith("]"):
                    sample_numbers = json.loads(sample_range)
                    transformed_samples.extend([f"{run_id}_{i:02d}" for i in sample_numbers])
                elif sample_range == "all":
                    # Check the folders
                    run_path = data_folder / run_id
                    if run_path.exists():
                        transformed_samples.extend([f.name for f in run_path.iterdir() if f.is_dir()])
                    else:
                        print(f"Folder {run_path!s} does not exist")
                else:
                    numbers = re.findall(r"\d+", sample_range)
                    start, end = map(int, numbers) if len(numbers) == 2 else (int(numbers[0]), int(numbers[0]))
                    transformed_samples.extend([f"{run_id}_{i:02d}" for i in range(start, end + 1)])

        # Check if individual sample folders exist
        for sample in transformed_samples:
            run_id = run_from_sample(sample)
            sample_folder = Path(data_folder) / run_id / sample
            if not sample_folder.exists():
                print(f"{sample} has no data folder, removing from list")
                # remove this element from the list
                transformed_samples.remove(sample)

        # overwrite the samples with the transformed samples
        batches[plot_name]["samples"] = transformed_samples

    return batches


def analyse_batch(plot_name: str, batch: dict) -> None:
    """Combine data for a batch of samples."""
    save_location = Path(CONFIG["Batches folder path"]) / plot_name
    if not save_location.exists():
        save_location.mkdir(parents=True, exist_ok=True)
    samples = batch.get("samples", [])
    cycle_dicts = []
    metadata: dict[str, dict] = {"sample_metadata": {}}
    for sample in samples:
        # get the anaylsed data
        run_id = run_from_sample(sample)
        sample_folder = Path(CONFIG["Processed snapshots folder path"]) / run_id / sample
        try:
            analysed_file = next(
                f for f in sample_folder.iterdir() if (f.name.startswith("cycles.") and f.name.endswith(".json"))
            )
            with analysed_file.open(encoding="utf-8") as f:
                data = json.load(f)
                cycle_dict = data.get("data", {})
                metadata["sample_metadata"][sample] = data.get("metadata", {})
            if cycle_dict.get("Cycle") and cycle_dict["Cycle"]:
                cycle_dicts.append(cycle_dict)
            else:
                print(f"No cycling data for {sample}")
                continue
        except StopIteration:
            # Handle the case where no file starts with 'cycles'
            print(f"No files starting with 'cycles' found in {sample_folder}.")
            continue
    cycle_dicts = [d for d in cycle_dicts if d.get("Cycle") and d["Cycle"]]
    if len(cycle_dicts) == 0:
        msg = "No cycling data found for any sample"
        raise ValueError(msg)

    # update the metadata
    timezone = pytz.timezone(CONFIG.get("Time zone", "Europe/Zurich"))
    metadata["provenance"] = {
        "aurora_metadata": {
            "batch_analysis": {
                "repo_url": __url__,
                "repo_version": __version__,
                "method": "analysis.analyse_batch",
                "datetime": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %z"),
            },
        },
    }

    # make another df where we only keep the lists from the dictionaries in the list
    only_lists = pd.concat(
        [
            pd.DataFrame({k: v for k, v in cycle_dict.items() if isinstance(v, list) or k == "Sample ID"})
            for cycle_dict in cycle_dicts
        ],
    )
    only_vals = pd.DataFrame(
        [{k: v for k, v in cycle_dict.items() if not isinstance(v, list)} for cycle_dict in cycle_dicts],
    )

    with pd.ExcelWriter(f"{save_location}/batch.{plot_name}.xlsx") as writer:
        only_lists.to_excel(writer, sheet_name="Data by cycle", index=False)
        only_vals.to_excel(writer, sheet_name="Results by sample", index=False)
    with (save_location / f"batch.{plot_name}.json").open("w", encoding="utf-8") as f:
        json.dump({"data": cycle_dicts, "metadata": metadata}, f)


def analyse_all_batches() -> None:
    """Analyses all the batches according to the configuration file.

    Args:
        graph_config_path (str): path to the yaml file containing the plotting config
            Defaults to "K:/Aurora/cucumber/graph_config.yml"

    Will search for analysed data in the processed snapshots folder and plot and
    save the capacity and efficiency vs cycle for each batch of samples.

    """
    batches = parse_sample_plotting_file(Path(CONFIG["Graph config path"]))
    for plot_name, batch in batches.items():
        try:
            analyse_batch(plot_name, batch)
        except (ValueError, KeyError, PermissionError, RuntimeError, FileNotFoundError) as e:
            print(f"Failed to analyse {plot_name} with error {e}")
