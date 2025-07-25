"""Copyright © 2025, Empa.

Functions used for parsing, analysing and plotting.

Takes partial cycling files and combines into one full DataFrame and hdf5 file.

Analyses metadata and data to extract useful information, including protocol
summary information, per-cycle data, and summary statistics.
"""

import contextlib
import json
import logging
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import pytz
from tsdownsample import MinMaxLTTBDownsampler

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import get_batch_details, get_sample_data
from aurora_cycler_manager.utils import (
    c_to_float,
    json_dump_compress_lists,
    max_with_none,
    min_with_none,
    round_c_rate,
    run_from_sample,
    weighted_median,
)
from aurora_cycler_manager.version import __url__, __version__

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN axis encountered")
logger = logging.getLogger(__name__)

CONFIG = get_config()
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
    else:
        df["loop_number"] = df["loop_number"].fillna(0)

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
            "original_file_provenance": {str(f): m["provenance"] for f, m in zip(job_files, metadatas, strict=False)},
        },
        "sample_data": sample_data,
        "job_data": [m.get("job_data", {}) for m in metadatas],
        "glossary": glossary,
    }

    return df, metadata


def extract_voltage_crates(job_data: dict) -> dict:
    """Extract min and max voltage, C-rate, and formation cycle count from job data."""
    form_C = None
    form_max_V = None
    form_min_V = None
    cycle_C = None
    cycle_max_V = None
    cycle_min_V = None
    form_cycle_count = None

    voltage = None
    min_V = None
    max_V = None
    global_max_V = None
    current = None
    new_current = None
    rate = None
    new_rate = None
    is_already_rate = False

    # Iterate through jobs, behave differently depending on the job type
    for job in job_data:
        job_type = job.get("job_type")

        # TOMATO 0.2.3 using biologic driver
        if job_type == "tomato_0_2_biologic":
            try:
                capacity = float(job.get("Payload", {}).get("sample", {}).get("capacity", 0))  # in Ah
            except ValueError:
                capacity = 0
            for method in job.get("Payload", {}).get("method", []):
                if not isinstance(method, dict):
                    continue
                if method.get("technique") == "constant_current":
                    try:
                        new_current = abs(c_to_float(method.get("current")))
                        is_already_rate = True
                    except ValueError:
                        with contextlib.suppress(ValueError, TypeError):
                            new_current = float(method.get("current"))
                            is_already_rate = False
                    if new_current:
                        current = new_current

                    with contextlib.suppress(ValueError, TypeError):
                        voltage = method.get("limit_voltage_min")
                    min_V = min_with_none([min_V, voltage])
                    with contextlib.suppress(ValueError, TypeError):
                        voltage = method.get("limit_voltage_max")
                    max_V = max_with_none([max_V, voltage])
                    global_max_V = max_with_none([global_max_V, voltage])
                if method.get("technique") == "loop":
                    if method.get("n_gotos", 9) < 9:
                        if not form_C and current:
                            if is_already_rate:
                                form_C = round_c_rate(current, 10)
                            elif capacity:
                                form_C = round_c_rate(current / capacity, 10)
                        if not form_max_V and max_V:
                            form_max_V = max_V
                        if not form_min_V and min_V:
                            form_min_V = min_V
                        if not form_cycle_count:
                            form_cycle_count = method["n_gotos"] + 1  # Ec-lab ngotos is cycles-1

                    elif method.get("n_gotos", 0) > 9:
                        if not cycle_C and current:
                            if is_already_rate:
                                cycle_C = round_c_rate(current, 10)
                            elif capacity:
                                cycle_C = round_c_rate(current / capacity, 10)
                        if not cycle_max_V and max_V:
                            cycle_max_V = max_V
                        if not cycle_min_V and min_V:
                            cycle_min_V = min_V

                    if (form_C and cycle_C) or (form_max_V and cycle_max_V):
                        break
                    max_V, min_V = None, None
                    current, new_current = None, None

        # Neware xlsx or ndax
        elif job_type in ("neware_xlsx", "neware_ndax"):
            try:
                capacity = float(job.get("MultCap", 0))  # in mAs
            except ValueError:
                capacity = 0
            for method in job["Payload"]:
                if not isinstance(method, dict):
                    continue
                # Remember the last cycling current and voltage
                if method.get("Step Name") == "CC Chg":
                    with contextlib.suppress(ValueError):
                        new_current = abs(float(method.get("Current (A)", 0)))
                    if new_current:
                        current = new_current
                    with contextlib.suppress(ValueError):
                        voltage = float(method.get("Cut-off voltage (V)", 0))
                    max_V = max_with_none([max_V, voltage])
                    global_max_V = max_with_none([global_max_V, voltage])
                if method.get("Step Name") == "CC DChg":
                    with contextlib.suppress(ValueError):
                        new_current = abs(float(method.get("Current (A)", 0)))
                    current = max_with_none([current, new_current])
                    with contextlib.suppress(ValueError):
                        voltage = float(method.get("Cut-off voltage (V)", 0))
                    min_V = min_with_none([min_V, voltage])

                # If there is a cycle step, assign formation or longterm C-rate and voltage
                if method.get("Step Name") == "Cycle" and method.get("Cycle count"):
                    try:
                        cycle_count = int(method["Cycle count"])
                    except ValueError:
                        continue
                    # First time less than 10 cycles, assume formation
                    if cycle_count < 10 and not (form_C or form_max_V or form_min_V or form_cycle_count):
                        form_C = round_c_rate(current / (capacity / 3.6e6), 10) if (current and capacity) else None
                        form_max_V = max_V
                        form_min_V = min_V
                        form_cycle_count = cycle_count
                    # First time more than 10 cycles, assume longterm
                    elif cycle_count >= 10 and not (cycle_C or cycle_max_V or cycle_min_V):
                        cycle_C = round_c_rate(current / (capacity / 3.6e6), 10) if (current and capacity) else None
                        cycle_max_V = max_V if max_V else None
                        cycle_min_V = min_V if min_V else None
                    if (cycle_C and form_C) or (cycle_max_V and form_max_V):
                        break
                    # Reset current and voltage
                    max_V, min_V = None, None
                    current, new_current = None, None

        # EC-lab mpr
        elif job_type == "eclab_mpr":
            capacity = 0
            capacity_units = job.get("settings", {}).get("battery_capacity_unit")
            if capacity_units == 1:  # mAh
                capacity = job.get("settings", {}).get("battery_capacity", 0)  # in mAh
            if capacity_units and capacity_units != 1:
                logger.warning("Unknown capacity units from ec-lab: %s", capacity_units)

            if isinstance(job.get("params", []), dict):  # it may be a dict of lists instead of a list of dicts
                try:
                    n_techniques = len(job["params"].get("Is") or job["params"].get("N"))
                    job["params"] = [{k: val[i] for k, val in job["params"].items()} for i in range(n_techniques)]
                except (ValueError, TypeError, KeyError, AttributeError):
                    logger.exception("EC-lab params not in expected format, should be list of dicts or dict of lists")

            if job.get("settings", {}).get("technique", "") == "GCPL":
                for method in job.get("params", []):
                    if not isinstance(method, dict):
                        continue
                    current_mode = method.get("set_I/C") or method.get("Set I/C")
                    current = method.get("Is")
                    if current_mode == "C":
                        new_rate = method.get("N")
                        rate = 1 / new_rate if new_rate else None
                    elif current_mode == "I" and capacity:
                        current_units = method.get("I_unit") or method.get("unit Is")
                        if current and current_units:
                            if current_units == "A":
                                current = current * 1000
                            elif current_units != "mA":
                                logger.warning("EC-lab current unit unknown: %s", current_units)
                            rate = abs(current) / capacity
                    # Get voltage
                    discharging = None
                    Isign = method.get("I_sign") or method.get("I sign")
                    if current_mode == "C":
                        discharging = Isign
                    elif current_mode == "I" and current:
                        if Isign:  # noqa: SIM108
                            discharging = 1 if current * (1 - 2 * Isign) < 0 else 0
                        else:
                            discharging = 1 if current < 0 else 0
                    voltage = method.get("EM") or method.get("EM (V)")
                    global_max_V = max_with_none([global_max_V, voltage])
                    if voltage:
                        if discharging == 1:
                            min_V = min_with_none([min_V, voltage])
                        elif discharging == 0:
                            max_V = max_with_none([max_V, voltage])
                    # Get cycles and set values
                    cycles = method.get("nc_cycles") or method.get("nc cycles")
                    if cycles and cycles >= 1:
                        # Less than 10 cycles, assume formation
                        if cycles and cycles < 9:
                            if rate and not form_C:
                                form_C = round_c_rate(rate, 10)
                            if max_V and not form_max_V:
                                form_max_V = round(max_V, 6)
                            if min_V and not form_min_V:
                                form_min_V = round(min_V, 6)
                            if not form_cycle_count:
                                form_cycle_count = cycles + 1
                        # First time more than 10 cycles, assume longterm
                        elif cycles and cycles > 9 and not (cycle_C or cycle_max_V or cycle_min_V):
                            cycle_C = round_c_rate(rate, 10) if rate else None
                            cycle_max_V = round(max_V, 6) if max_V else None
                            cycle_min_V = round(min_V, 6) if min_V else None
                        # If we have both formation and cycle values, stop
                        if (cycle_C and form_C) or (cycle_max_V and form_max_V):
                            break
                        # Otherwise reset values and continue
                        max_V, min_V = None, None
                        current, new_current = None, None
                        rate, new_rate = None, None

            elif job.get("settings", {}).get("technique", "") == "MB":
                for method in job.get("params", []):
                    if not isinstance(method, dict):
                        continue
                    if method.get("ctrl_type") == 0:  # CC
                        # Get rate
                        current_mode = method.get("Apply I/C")
                        if current_mode == "C":
                            new_rate = method.get("N")
                            rate = 1 / new_rate if new_rate else None
                        elif current_mode == "I" and capacity:
                            current = method.get("ctrl1_val")
                            current_unit = method.get("ctrl1_val_unit")
                            if current and current_unit:
                                if current_unit == 1:  # mA
                                    pass
                                else:
                                    logger.warning("EC-lab current unit unknown: %s", current_unit)
                                rate = abs(current) / capacity
                        # Get voltage limits
                        for lim in [1, 2, 3]:
                            if method.get(f"lim{lim}_type") == 1:  # Voltage limit
                                voltage = method.get(f"lim{lim}_val")
                                voltage_unit = method.get(f"lim{lim}_val_unit")
                                lim_comp = method.get(f"lim{lim}_comp")
                                if voltage:
                                    if voltage_unit == 0:  # V
                                        pass
                                    else:
                                        logger.warning("EC-lab voltage unit unknown: %s", voltage_unit)
                                if lim_comp == 0:  # Charge
                                    max_V = max_with_none([max_V, voltage])
                                elif lim_comp == 1:  # Discharge
                                    min_V = min_with_none([min_V, voltage])
                                global_max_V = max_with_none([global_max_V, voltage])
                    # Get cycles and set values
                    cycles = method.get("ctrl_repeat")
                    if cycles and cycles >= 1:
                        # Less than 10 cycles, assume formation
                        if cycles and cycles < 9:
                            if rate and not form_C:
                                form_C = round_c_rate(rate, 10)
                            if max_V and not form_max_V:
                                form_max_V = round(max_V, 6)
                            if min_V and not form_min_V:
                                form_min_V = round(min_V, 6)
                            if not form_cycle_count:
                                form_cycle_count = cycles + 1
                        # First time more than 10 cycles, assume longterm
                        elif cycles and cycles > 9 and not (cycle_C or cycle_max_V or cycle_min_V):
                            cycle_C = round_c_rate(rate, 10) if rate else None
                            cycle_max_V = round(max_V, 6) if max_V else None
                            cycle_min_V = round(min_V, 6) if min_V else None
                        # If we have both formation and cycle values, stop
                        if (cycle_C and form_C) or (cycle_max_V and form_max_V):
                            break
                        # Otherwise reset values and continue
                        max_V, min_V = None, None
                        current, new_current = None, None
                        rate, new_rate = None, None
    global_max_V = round(global_max_V, 6) if global_max_V else None
    return {
        "form_C": form_C,
        "form_max_V": form_max_V,
        "form_min_V": form_min_V,
        "cycle_C": cycle_C,
        "cycle_max_V": cycle_max_V,
        "cycle_min_V": cycle_min_V,
        "global_max_V": global_max_V,
        "form_cycle_count": form_cycle_count,
    }


def analyse_cycles(
    job_files: list[Path],
    voltage_lower_cutoff: float = 0,
    voltage_upper_cutoff: float = 5,
    save_cycle_dict: bool = False,
    save_merged_hdf: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    """Take multiple dataframes, merge and analyse the cycling data.

    Args:
        job_files (List[Path]): list of paths to the hdf5 job files
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
    sampleid = sample_data.get("Sample ID")
    job_data = metadata.get("job_data")
    snapshot_status = job_data[-1].get("Snapshot status") if job_data else None  # Used in tomato
    finished = job_data[-1].get("Finished") if job_data else None  # Used in Newares
    snapshot_pipeline = job_data[-1].get("Pipeline") if job_data else None
    last_snapshot = job_data[-1].get("Last snapshot") if job_data else None

    # Extract useful information from the metadata
    mass_mg = sample_data.get("Cathode active material mass (mg)", np.nan)

    # Get voltage and C-rates
    protocol_summary = extract_voltage_crates(job_data) if job_data else {}

    # Check current status and pipeline (may be more recenty than snapshot)
    pipeline, status = None, None
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

    # Analyse each cycle in the cycling data
    charge_capacity_mAh = []
    discharge_capacity_mAh = []
    charge_avg_V = []
    discharge_avg_V = []
    charge_energy_mWh = []
    discharge_energy_mWh = []
    charge_avg_I = []
    discharge_avg_I = []
    cycle_median_I = []
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
            cycle_median_I.append(weighted_median(abs(charge_data["Iavg (A)"]), abs(charge_data["dQ (mAh)"])))

    # Try to guess the number of formation cycles if it was not found from the job data
    form_cycle_count = protocol_summary.get("form_cycle_count")
    if not form_cycle_count:
        form_cycle_count = 3
        # Check median current up to 10 cycles, if it changes assume that is the formation cycle
        rounded_current = [f"{x:.2g}" for x in cycle_median_I[: min(10, len(cycle_median_I))]]
        if len(rounded_current) > 2 and len(set(rounded_current)) > 1:
            idx = next((i for i, x in enumerate(rounded_current) if x != rounded_current[0]), None)
            if idx is not None:
                form_cycle_count = idx
    initial_cycle = form_cycle_count + 1

    formed = len(charge_capacity_mAh) >= initial_cycle

    # A row is added if charge data is complete and discharge started, but it may have incomplete discharge data
    # If the job is not complete but a discharge has started, set the last discharge data to NaN
    complete = 1
    if (
        started_charge
        and started_discharge
        and (snapshot_status in ["r", "cd", "ce"] or finished is False)  # job is still running
    ):
        discharge_capacity_mAh[-1] = np.nan
        complete = 0

    # Create a dictionary with the cycling data
    cycle_dict = {
        "Sample ID": sampleid,
        "Cycle": list(range(1, len(charge_capacity_mAh) + 1)),
        "Charge capacity (mAh)": charge_capacity_mAh,
        "Discharge capacity (mAh)": discharge_capacity_mAh,
        "Charge energy (mWh)": charge_energy_mWh,
        "Discharge energy (mWh)": discharge_energy_mWh,
        "Charge average voltage (V)": [e / c for e, c in zip(charge_energy_mWh, charge_capacity_mAh, strict=False)],
        "Discharge average voltage (V)": [
            e / c for e, c in zip(discharge_energy_mWh, discharge_capacity_mAh, strict=False)
        ],
        "Coulombic efficiency (%)": [
            100 * d / c for d, c in zip(discharge_capacity_mAh, charge_capacity_mAh, strict=False)
        ],
        "Energy efficiency (%)": [100 * d / c for d, c in zip(discharge_energy_mWh, charge_energy_mWh, strict=False)],
        "Voltage efficiency (%)": [100 * d / c for d, c in zip(discharge_avg_V, charge_avg_V, strict=False)],
        "Specific charge capacity (mAh/g)": [c / (mass_mg * 1e-3) for c in charge_capacity_mAh] if mass_mg else None,
        "Specific discharge capacity (mAh/g)": [d / (mass_mg * 1e-3) for d in discharge_capacity_mAh]
        if mass_mg
        else None,
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
        "Delta V (V)": [c - d for c, d in zip(charge_avg_V, discharge_avg_V, strict=False)],
        "Charge average current (A)": charge_avg_I,
        "Discharge average current (A)": discharge_avg_I,
        "Formation max voltage (V)": protocol_summary["form_max_V"],
        "Formation min voltage (V)": protocol_summary["form_min_V"],
        "Cycle max voltage (V)": protocol_summary["cycle_max_V"],
        "Cycle min voltage (V)": protocol_summary["cycle_min_V"],
        "Max voltage (V)": protocol_summary["global_max_V"],
        "Formation C": protocol_summary["form_C"] if protocol_summary["form_C"] else 0,  # for backwards compatibility
        "Cycle C": protocol_summary["cycle_C"],
        "Formation cycles": form_cycle_count,
    }

    # Add other columns from sample table to cycle_dict
    for col in SAMPLE_METADATA_TO_DATA:
        cycle_dict[col] = sample_data.get(col)

    # Calculate additional quantities from cycling data and add to cycle_dict
    if not cycle_dict or not cycle_dict["Cycle"]:
        logger.info("No cycles found for %s", sampleid)
    elif len(cycle_dict["Cycle"]) == 1 and not complete:
        logger.info("No complete cycles found for %s", sampleid)
    else:  # Analyse the cycling data
        last_idx = -1 if complete else -2

        cycle_dict["First formation coulombic efficiency (%)"] = cycle_dict["Coulombic efficiency (%)"][0]
        cycle_dict["First formation specific discharge capacity (mAh/g)"] = (
            cycle_dict["Specific discharge capacity (mAh/g)"][0] if mass_mg else None
        )
        cycle_dict["Initial specific discharge capacity (mAh/g)"] = (
            (cycle_dict["Specific discharge capacity (mAh/g)"][initial_cycle - 1] if formed else None)
            if mass_mg
            else None
        )
        cycle_dict["Initial coulombic efficiency (%)"] = (
            cycle_dict["Coulombic efficiency (%)"][initial_cycle - 1] if formed else None
        )
        cycle_dict["Capacity loss (%)"] = (
            100 - cycle_dict["Normalised discharge capacity (%)"][last_idx] if formed else None
        )
        cycle_dict["Last specific discharge capacity (mAh/g)"] = (
            cycle_dict["Specific discharge capacity (mAh/g)"][last_idx] if mass_mg else None
        )
        cycle_dict["Last coulombic efficiency (%)"] = cycle_dict["Coulombic efficiency (%)"][last_idx]
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
            cycle_dict[f"Cycles to {pcent}% capacity"] = None
            if formed:
                abs_cycle = _find_first_element(norm < pcent, form_cycle_count)
                if abs_cycle is not None:
                    cycle_dict[f"Cycles to {pcent}% capacity"] = abs_cycle - form_cycle_count
        norm = np.array(cycle_dict["Normalised discharge energy (%)"])
        for pcent in pcents:
            cycle_dict[f"Cycles to {pcent}% energy"] = None
            if formed:
                abs_cycle = _find_first_element(norm < pcent, form_cycle_count)
                if abs_cycle is not None:
                    cycle_dict[f"Cycles to {pcent}% energy"] = abs_cycle - form_cycle_count

        cycle_dict["Run ID"] = run_from_sample(sampleid)

        # If assembly history is available, calculate times between steps
        assembly_history = sample_data.get("Assembly history", [])
        if isinstance(assembly_history, str):
            assembly_history = json.loads(assembly_history)
        if assembly_history and isinstance(assembly_history, list):
            job_start = df["uts"].iloc[0]
            press = next((step.get("uts") for step in assembly_history if step["Step"] == "Press"), None)
            electrolyte_ind = [i for i, step in enumerate(assembly_history) if step["Step"] == "Electrolyte"]
            if electrolyte_ind:
                first_electrolyte = next(
                    (step.get("uts") for step in assembly_history if step["Step"] == "Electrolyte"),
                    None,
                )
                history_after_electrolyte = assembly_history[max(electrolyte_ind) :]
                cover_electrolyte = next(
                    (step.get("uts") for step in history_after_electrolyte if step["Step"] in ["Anode", "Cathode"]),
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
        if pipeline:
            if formed and (cap_loss := cycle_dict.get("Capacity loss (%)")) and cap_loss > 50:
                flag = "🪫"
            if (form_eff := cycle_dict.get("First formation coulombic efficiency (%)")) and form_eff < 50:
                flag = "🚩"
            if formed and (init_eff := cycle_dict.get("Initial coulombic efficiency (%)")) and init_eff < 50:
                flag = "🚩"
            if formed and (init_cap := cycle_dict.get("Initial specific discharge capacity (mAh/g)")) and init_cap < 50:
                flag = "🚩"
        update_row = {
            "Pipeline": pipeline,
            "Status": status,
            "Flag": flag,
            "Number of cycles": int(max(cycle_dict["Cycle"])),
            "Capacity loss (%)": cycle_dict["Capacity loss (%)"],
            "Max voltage (V)": max_with_none([protocol_summary["form_max_V"], protocol_summary["cycle_max_V"]]),
            "Formation C": cycle_dict["Formation C"],
            "Cycling C": cycle_dict["Cycle C"],
            "First formation efficiency (%)": cycle_dict["First formation coulombic efficiency (%)"],
            "Initial specific discharge capacity (mAh/g)": cycle_dict["Initial specific discharge capacity (mAh/g)"],
            "Initial efficiency (%)": cycle_dict["Initial coulombic efficiency (%)"],
            "Last specific discharge capacity (mAh/g)": cycle_dict["Last specific discharge capacity (mAh/g)"],
            "Last efficiency (%)": cycle_dict["Last coulombic efficiency (%)"],
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
                f"UPDATE results SET {columns} WHERE `Sample ID` = ?",  # noqa: S608
                (*update_row.values(), sampleid),
            )

    # Glossary for analysed cycles
    cycles_glossary = {
        "Sample ID": "Sample ID (YYMMDD_name_tag_XX)",
        "Cycle": "Cycle number",
        "Charge capacity (mAh)": "Charge capacity in milliampere-hours (mAh)",
        "Discharge capacity (mAh)": "Discharge capacity in milliampere-hours (mAh)",
        "Charge energy (mWh)": "Charge energy in milliwatt-hours (mWh)",
        "Discharge energy (mWh)": "Discharge energy in milliwatt-hours (mWh)",
        "Charge average voltage (V)": "Average charge voltage (V), current-weighted",
        "Discharge average voltage (V)": "Average discharge voltage (V), current-weighted",
        "Coulombic efficiency (%)": "Discharge capacity / Charge capacity * 100",
        "Energy efficiency (%)": "Discharge energy / Charge energy * 100",
        "Voltage efficiency (%)": "Discharge average voltage / Charge average voltage * 100",
        "Specific charge capacity (mAh/g)": "Charge capacity per gram of active material",
        "Specific discharge capacity (mAh/g)": "Discharge capacity per gram of active material",
        "Normalised discharge capacity (%)": "Discharge capacity normalized to post-formation first cycle",
        "Normalised discharge energy (%)": "Discharge energy normalized to post-formation first cycle",
        "Delta V (V)": "Charge average voltage - Discharge average voltage",
        "Charge average current (A)": "Average current during charge weighted by current (sum(I*dQ)/sum(dQ))",
        "Discharge average current (A)": "Average current during discharge weighted by current (sum(I*dQ)/sum(dQ))",
        "Formation max voltage (V)": "Max voltage during formation from protocol (not measured data)",
        "Formation min voltage (V)": "Min voltage during formation from protocol (not measured data)",
        "Cycle max voltage (V)": "Max voltage during long-term cycling found from protocol (not measured data)",
        "Cycle min voltage (V)": "Min voltage during long-term cycling found from protocol (not measured data)",
        "Max voltage (V)": "Max voltage in entire protocol found from protocol (not measured data)",
        "Formation C": "C-rate during formation cycling, found from protocol (not measured data)",
        "Cycle C": "C-rate during cycling, found from protocol (not measured data)",
        "Formation cycles": "Number of formation cycles determined from protocol (assumes <10 and same C-rate)",
        "First formation coulombic efficiency (%)": "Coulombic efficiency in first formation cycle",
        "First formation specific discharge capacity (mAh/g)": "Specific discharge capacity in first formation cycle",
        "Initial specific discharge capacity (mAh/g)": "Specific discharge capacity in first post-formation cycle",
        "Initial coulombic efficiency (%)": "Coulombic efficiency in first post-formation cycle",
        "Capacity loss (%)": "Capacity loss from first post-formation to last cycle",
        "Last specific discharge capacity (mAh/g)": "Specific discharge capacity in last cycle",
        "Last coulombic efficiency (%)": "Coulombic efficiency in last cycle",
        "Formation average voltage (V)": "Average voltage in formation (current-weighted)",
        "Formation average current (A)": "Average. current in formation (current-weighted)",
        "Initial delta V (V)": "Charge - Discharge average voltage in first post-formation cycle",
    }
    cycles_metadata = metadata.copy()
    cycles_metadata["glossary"] = cycles_glossary
    if save_cycle_dict or save_merged_hdf:
        save_folder = job_files[0].parent
        if save_cycle_dict:
            with (save_folder / f"cycles.{sampleid}.json").open("w", encoding="utf-8") as f:
                json_dump_compress_lists({"data": cycle_dict, "metadata": cycles_metadata}, f, indent=4)
        if save_merged_hdf:
            df = df.drop(columns=["dt (s)", "Iavg (A)"])
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
    return df, cycle_dict, metadata


def analyse_sample(sample: str) -> tuple[pd.DataFrame, dict, dict]:
    """Analyse a single sample.

    Will search for the sample in the processed snapshots folder and analyse the cycling data.

    """
    run_id = run_from_sample(sample)
    file_location = Path(CONFIG["Processed snapshots folder path"]) / run_id / sample
    job_files = list(file_location.glob("snapshot.*.h5"))
    df, cycle_dict, metadata = analyse_cycles(
        job_files,
        save_cycle_dict=True,
        save_merged_hdf=True,
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
            logger.warning("File %s not found", hdf5_file)
            continue
        with h5py.File(hdf5_file, "a") as f:
            # check the keys data and metadata exist
            if "data" not in f or "metadata" not in f:
                logger.warning("File %s has incorrect format", hdf5_file)
                continue
            metadata = json.loads(f["metadata"][()])
            sample_data = get_sample_data(sample_id)
            metadata["sample_data"] = sample_data
            f["metadata"][()] = json.dumps(metadata)
        # JSON cycles file
        json_file = sample_folder / f"cycles.{sample_id}.json"
        if not json_file.exists():
            logger.warning("File %s not found", json_file)
            continue
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # check it has keys data and metadata
            if "data" not in data or "metadata" not in data:
                logger.warning("File %s has incorrect format", json_file)
                continue
            data["metadata"]["sample_data"] = sample_data
            for col in SAMPLE_METADATA_TO_DATA:
                data["data"][col] = sample_data.get(col)
        with json_file.open("w", encoding="utf-8") as f:
            json_dump_compress_lists(data, f, indent=4)


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
    # Calculate derivative - impossible to do after downsampling
    df["Q (mAh)"] = df["dQ (mAh)"].cumsum()
    # Group by cycle and calculate the derivative
    dqdv = np.full(len(df), np.nan)
    for idx in df.groupby("Cycle").indices.values():
        group = df.iloc[np.array(idx)]
        dqdv[idx] = calc_dqdv(group["V (V)"].to_numpy(), group["Q (mAh)"].to_numpy(), group["dQ (mAh)"].to_numpy())
    df["dQ/dV (mAh/V)"] = dqdv

    # Reduce precision of some columns
    for col in ["V (V)", "I (A)", "dQ (mAh)", "dQ/dV (mAh/V)", "Q (mAh)"]:
        df[col] = df[col].astype(np.float32)
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

    # Downsample the dataframe
    df = df.iloc[ind]

    # Recalculate dQ so it cumulates correctly after downsampling
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
                    logger.info("Shrunk %s", sample.name)
                except (KeyError, ValueError, PermissionError, RuntimeError, FileNotFoundError):
                    logger.exception("Failed to shrink %s", sample.name)


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
                    logger.info("Analysed %s", sample.name)
                except (KeyError, ValueError, PermissionError, RuntimeError, FileNotFoundError, TypeError):
                    logger.exception("Failed to analyse %s", sample.name)


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
        if not sample_folder.exists():
            logger.warning("Folder %s does not exist", sample_folder)
            continue
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
                logger.warning("No cycling data found for %s", sample)
                continue
        except StopIteration:
            # Handle the case where no file starts with 'cycles'
            logger.warning("No files starting with 'cycles' found in %s", sample_folder)
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
    batches = get_batch_details()
    for plot_name, batch in batches.items():
        try:
            analyse_batch(plot_name, batch)
        except (ValueError, KeyError, PermissionError, RuntimeError, FileNotFoundError):  # noqa: PERF203
            logger.exception("Failed to analyse %s", plot_name)


def moving_average(x, npoints: int = 11) -> np.ndarray:
    """Calculate moving window average of a 1D array."""
    if npoints % 2 == 0:
        npoints += 1  # Ensure npoints is odd for a symmetric window
    window = np.ones(npoints) / npoints
    xav = np.convolve(x, window, mode="same")
    xav[: npoints // 2] = np.nan
    xav[-npoints // 2 :] = np.nan
    return xav


def deriv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate dy/dx for 1D arrays, ignore division by zero errors."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dydx = np.zeros(len(y))
        dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
        dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

        # for any 3 points where x direction changes sign set to nan
        mask = (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0
        dydx[1:-1][mask] = np.nan
    return dydx


def smoothed_derivative(
    x: np.ndarray,
    y: np.ndarray,
    npoints: int = 21,
) -> np.ndarray:
    """Calculate dy/dx with moving window average."""
    x_smooth = moving_average(x, npoints)
    y_smooth = moving_average(y, npoints)
    return deriv(x_smooth, y_smooth)


def calc_dqdv(v: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """Calculate dQ/dV from V, Q, and dQ."""
    # Preallocate output array
    dvdq = np.full_like(v, np.nan, dtype=float)

    # Split into positive and negative dq, work on slices
    pos_mask = dq >= 0
    neg_mask = ~pos_mask

    if np.sum(pos_mask) > 5:
        v_pos = v[pos_mask]
        q_pos = q[pos_mask]
        dq_pos = dq[pos_mask]
        # Remove end points which can be problematic, e.g. with CV steps
        bad_pos = (v_pos > np.max(v_pos) * 0.999) | (v_pos < np.min(v_pos) * 1.001) | (np.abs(dq_pos) < 1e-9)
        npoints = max(5, np.sum(~bad_pos) // 25)
        dvdq_pos = smoothed_derivative(q_pos, v_pos, npoints=npoints)
        dvdq_pos[bad_pos] = np.nan
        dvdq[pos_mask] = dvdq_pos

    if np.sum(neg_mask) > 5:
        v_neg = v[neg_mask]
        q_neg = q[neg_mask]
        dq_neg = dq[neg_mask]
        # Remove end points which can be problematic, e.g. with CV steps
        bad_neg = (v_neg > np.max(v_neg) * 0.999) | (v_neg < np.min(v_neg) * 1.001) | (np.abs(dq_neg) < 1e-9)
        npoints = max(5, np.sum(~bad_neg) // 25)
        dvdq_neg = smoothed_derivative(q_neg, v_neg, npoints=npoints)
        dvdq_neg[bad_neg] = np.nan
        dvdq[neg_mask] = -dvdq_neg

    return 1 / dvdq
