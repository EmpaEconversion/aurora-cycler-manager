"""Copyright © 2025-2026, Empa.

Functions used for parsing, analysing and plotting.

Takes partial cycling files and combines into one full DataFrame and parquet file.

Analyses metadata and data to extract useful information, including protocol
summary information, per-cycle data, and summary statistics.
"""

import contextlib
import json
import logging
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import polars as pl
from tsdownsample import MinMaxLTTBDownsampler
from xlsxwriter import Workbook

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.data_bundle import (
    SampleDataBundle,
    get_cycles_summary,
    get_cycling,
    get_metadata,
    get_overall_summary,
    get_sample_folder,
    read_cycling,
    read_metadata,
)
from aurora_cycler_manager.database_funcs import get_batch_details, get_sample_data
from aurora_cycler_manager.stdlib_utils import (
    json_dump_compress_lists,
    max_with_none,
    min_with_none,
    round_c_rate,
    run_from_sample,
)
from aurora_cycler_manager.utils import (
    parse_datetime,
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


def merge_metadata(job_files: list[Path], metadatas: list[dict]) -> dict:
    """Merge several job metadata, add provenance, replace sample data with latest from db."""
    sample_id = metadatas[0].get("sample_data", {}).get("Sample ID", "")
    sample_data = get_sample_data(sample_id)
    # Merge glossary dicts
    glossary = {}
    for g in [m.get("glossary", {}) for m in metadatas]:
        glossary.update(g)
    return {
        "provenance": {
            "aurora_metadata": {
                "data_merging": {
                    "job_files": [str(f) for f in job_files],
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "analysis.merge_dfs",
                    "datetime": datetime.now(timezone.utc).isoformat(),
                },
            },
            "original_file_provenance": {str(f): m["provenance"] for f, m in zip(job_files, metadatas, strict=True)},
        },
        "sample_data": sample_data,
        "job_data": [m.get("job_data", {}) for m in metadatas],
        "glossary": glossary,
    }


def read_and_order_job_files(job_files: list[Path]) -> tuple[list[Path], list[pl.DataFrame], list[dict]]:
    """Take list of job files, reorder by time, return lists of dataframes and metadata."""
    dfs = [read_cycling(f) for f in job_files]
    metadatas = [read_metadata(f) for f in job_files]
    if len(dfs) == 0:
        msg = "No valid cycling files provided"
        raise ValueError(msg)
    sampleids = [m.get("sample_data", {}).get("Sample ID", "") for m in metadatas]
    if len(set(sampleids)) > 1:
        msg = "All files must be from the same sample"
        raise ValueError(msg)
    start_times = [df["uts"][0] for df in dfs]
    end_times = [df["uts"][-1] for df in dfs]
    order = _sort_times(start_times, end_times)
    job_files = [job_files[i] for i in order]
    dfs = [dfs[i] for i in order]
    metadatas = [metadatas[i] for i in order]
    return job_files, dfs, metadatas


def calc_dq(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate and add dQ (mAh) column."""
    # TODO: this should be smarter - maybe assert 0 between steps
    return (
        df.with_columns(
            [
                pl.col("uts").diff().fill_null(0).alias("dt (s)"),
                ((pl.col("I (A)") + pl.col("I (A)").shift(1)) / 2).fill_null(0).alias("Iavg (A)"),
            ]
        )
        .with_columns([(1e3 * pl.col("Iavg (A)") * pl.col("dt (s)") / 3600).alias("dQ (mAh)")])
        .with_columns([pl.when(pl.col("dt (s)") > 600).then(0).otherwise(pl.col("dQ (mAh)")).alias("dQ (mAh)")])
        .drop("Iavg (A)", "dt (s)")
    )


def merge_dfs(dfs: list[pl.DataFrame]) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Merge cycling dataframes and add cycles. Seperate out EIS."""
    for i, df in enumerate(dfs):
        dfs[i] = df.with_columns(pl.lit(i).alias("job_number"))

    df = pl.concat(dfs, how="diagonal")

    # If EIS exists, filter into its own df
    eis_df = None
    if "f (Hz)" in df.columns:
        eis_mask = (df["f (Hz)"].is_not_null()) & (df["f (Hz)"] != 0)
        eis_df = df.filter(eis_mask)
        df = df.filter(~eis_mask).drop("f (Hz)", "Re(Z) (ohm)", "Im(Z) (ohm)")
        if eis_df.is_empty():
            eis_df = None

    if not df.is_empty():
        df = df.sort("uts")
        if "loop_number" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("loop_number"))
        else:
            df = df.with_columns(pl.col("loop_number").fill_null(0))

        if "dQ (mAh)" not in df.columns:
            df = calc_dq(df)

        # Increment step if any job, cycle, or loop changes
        df = df.with_columns(pl.struct(["job_number", "cycle_number", "loop_number"]).rle_id().add(1).alias("Step"))

        # Drop columns
        df = df.drop("job_number", "cycle_number", "loop_number", "index", strict=False)

        # Calculate criteria for each Step group
        step_stats = df.group_by("Step").agg(
            [
                pl.len().alias("count"),
                (pl.col("I (A)") > 0).sum().alias("positive_count"),
                (pl.col("I (A)") < 0).sum().alias("negative_count"),
            ]
        )

        # Determine which steps are valid cycles
        step_stats = step_stats.with_columns(
            ((pl.col("count") > 10) & (pl.col("positive_count") > 5) & (pl.col("negative_count") > 5)).alias("is_cycle")
        )

        # Assign cycle numbers (cumsum of is_cycle, 0 for non-cycles)
        step_stats = step_stats.sort("Step").with_columns(
            pl.when(pl.col("is_cycle")).then(pl.col("is_cycle").cum_sum()).otherwise(0).alias("Cycle")
        )

        # Join back to main dataframe
        df = df.join(step_stats.select(["Step", "Cycle"]), on="Step", how="left")

        # EIS merge - find last non-zero cycle before the EIS
        if eis_df is not None:
            eis_df = eis_df.join_asof(
                df.filter(pl.col("Cycle") != 0).select(["uts", "Cycle"]), on="uts", strategy="backward"
            ).with_columns(pl.col("Cycle").fill_null(0))

    else:
        df = df.with_columns(
            [
                pl.Series("dQ (mAh)", [], dtype=pl.Float32),
                pl.Series("Step", [], dtype=pl.Int32),
                pl.Series("Cycle", [], dtype=pl.Int32),
            ]
        )
        if eis_df is not None:
            eis_df = eis_df.with_columns(pl.lit(0).alias("Cycle"))
    return df, eis_df


def extract_voltage_crates(job_data: list[dict]) -> dict:
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

    # Iterate through jobs, behave differently depending on the job type
    for job in job_data:
        job_type = job.get("job_type")

        # Neware xlsx or ndax
        if job_type in ("neware_xlsx", "neware_ndax"):
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
                    n_techniques = len(next(iter(job["params"].values())))
                    job["params"] = [{k: val[i] for k, val in job["params"].items()} for i in range(n_techniques)]
                except (ValueError, TypeError, KeyError, AttributeError, StopIteration):
                    logger.exception("EC-lab params not in expected format, should be list of dicts or dict of lists")
                    continue
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
                        discharging = (1 if current * (1 - 2 * Isign) < 0 else 0) if Isign else 1 if current < 0 else 0
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

    finished = job_data[-1].get("Finished") if job_data else None

    return {
        "form_C": form_C,
        "form_max_V": form_max_V,
        "form_min_V": form_min_V,
        "cycle_C": cycle_C,
        "cycle_max_V": cycle_max_V,
        "cycle_min_V": cycle_min_V,
        "global_max_V": global_max_V,
        "form_cycle_count": form_cycle_count,
        "finished": finished,
    }


def analyse_cycles(
    df: pl.DataFrame,
    mass_mg: float | None,
    protocol_summary: dict | None,
) -> tuple[pl.DataFrame, dict]:
    """Analyse time-series dataframe, return per-cycle summary."""
    # Analyse each cycle in the cycling data
    protocol_summary = protocol_summary or {}
    form_cycle_count = protocol_summary.get("form_cycle_count")
    finished = protocol_summary.get("finished")

    # TODO: Do we need to filter voltages? e.g. only use 0-5 V

    # Get summary stats
    summary_df = (
        df.group_by("Cycle")
        .agg(
            [
                pl.col("dQ (mAh)").clip(lower_bound=0).sum().alias("Charge capacity (mAh)"),
                -pl.col("dQ (mAh)").clip(upper_bound=0).sum().alias("Discharge capacity (mAh)"),
                (pl.col("V (V)") * pl.col("dQ (mAh)").clip(lower_bound=0)).sum().alias("Charge energy (mWh)"),
                -(pl.col("V (V)") * pl.col("dQ (mAh)").clip(upper_bound=0)).sum().alias("Discharge energy (mWh)"),
                (
                    (pl.col("I (A)") * pl.col("dQ (mAh)").clip(lower_bound=0)).sum()
                    / pl.col("dQ (mAh)").clip(lower_bound=0).sum()
                ).alias("Charge average current (A)"),
                -(
                    (pl.col("I (A)") * pl.col("dQ (mAh)").clip(upper_bound=0)).sum()
                    / pl.col("dQ (mAh)").clip(upper_bound=0).sum()
                ).alias("Discharge average current (A)"),
            ]
        )
        .sort("Cycle")
        .filter(pl.col("Cycle") > 0)
    )

    # Try to guess the number of formation cycles if it was not found from the job data
    if not form_cycle_count:
        form_cycle_count = 3
        # Check median current up to 10 cycles, if it changes assume that is the formation cycle
        median_currents = []
        for cycle in range(1, 11):
            df_filtered = df.filter(pl.col("Cycle") == cycle, pl.col("I (A)") > 0)
            median_currents.append(weighted_median(df_filtered["I (A)"], df_filtered["dQ (mAh)"]))
        rounded_current = [f"{x:.2g}" for x in median_currents if x]
        if len(rounded_current) > 2 and len(set(rounded_current)) > 1:
            idx = next((i for i, x in enumerate(rounded_current) if x != rounded_current[0]), None)
            if idx is not None:
                form_cycle_count = idx
        protocol_summary["form_cycle_count"] = form_cycle_count

    formed = len(summary_df) > form_cycle_count if form_cycle_count else False

    # A cycle can exist if there is charge and discharge data
    # If discharge has started, but measurement hasn't finished, then set last discharge to None
    if finished is False and df.filter(pl.col("Cycle") == pl.max("Cycle"), pl.col("I (A)") < 0).height > 5:
        summary_df = summary_df.with_columns(
            pl.when(pl.int_range(pl.len()) == pl.len() - 1)
            .then(None)
            .otherwise(pl.col("Discharge capacity (mAh)"))
            .alias("Discharge capacity (mAh)")
        )

    # Create a dictionary with the cycling data
    summary_df = summary_df.with_columns(
        [
            (pl.col("Charge energy (mWh)") / pl.col("Charge capacity (mAh)")).alias("Charge average voltage (V)"),
            (pl.col("Discharge energy (mWh)") / pl.col("Discharge capacity (mAh)")).alias(
                "Discharge average voltage (V)"
            ),
        ]
    ).with_columns(
        [
            (100 * pl.col("Discharge capacity (mAh)") / pl.col("Charge capacity (mAh)")).alias(
                "Coulombic efficiency (%)"
            ),
            (100 * pl.col("Discharge energy (mWh)") / pl.col("Charge energy (mWh)")).alias("Energy efficiency (%)"),
            (100 * pl.col("Discharge average voltage (V)") / pl.col("Charge average voltage (V)")).alias(
                "Voltage efficiency (%)"
            ),
            (pl.col("Charge average voltage (V)") - pl.col("Discharge average voltage (V)")).alias("Delta V (V)"),
        ]
    )
    if mass_mg:
        mass_g = mass_mg / 1000
        summary_df = summary_df.with_columns(
            [
                (pl.col("Charge capacity (mAh)") / mass_g).alias("Specific charge capacity (mAh/g)"),
                (pl.col("Discharge capacity (mAh)") / mass_g).alias("Specific discharge capacity (mAh/g)"),
                (pl.col("Charge energy (mWh)") / mass_g).alias("Specific charge energy (mWh/g)"),
                (pl.col("Discharge energy (mWh)") / mass_g).alias("Specific discharge energy (mWh/g)"),
            ]
        )
    if formed:
        dc100 = summary_df["Discharge capacity (mAh)"][form_cycle_count]
        de100 = summary_df["Discharge energy (mWh)"][form_cycle_count]
        summary_df = summary_df.with_columns(
            [
                (100 * pl.col("Discharge capacity (mAh)") / dc100).alias("Normalised discharge capacity (%)"),
                (100 * pl.col("Discharge energy (mWh)") / de100).alias("Normalised discharge energy (%)"),
            ]
        )

    return summary_df, protocol_summary


def analyse_overall(
    df: pl.DataFrame,
    eis_df: pl.DataFrame | None,
    metadata: dict,
    protocol_summary: dict,
    cycle_summary_df: pl.DataFrame,
) -> dict:
    """Get overall summary."""
    sample_data = metadata.get("sample_data", {})
    sample_id = sample_data.get("Sample ID")
    formation_cycles = protocol_summary.get("form_cycle_count")
    overall = {
        "Sample ID": sample_id,
        "Run ID": run_from_sample(sample_id),
        "Formation max voltage (V)": protocol_summary.get("form_max_V"),
        "Formation min voltage (V)": protocol_summary.get("form_min_V"),
        "Cycle max voltage (V)": protocol_summary.get("cycle_max_V"),
        "Cycle min voltage (V)": protocol_summary.get("cycle_min_V"),
        "Max voltage (V)": protocol_summary.get("global_max_V"),
        "Formation C": protocol_summary.get("form_C") or 0,  # for backwards compatibility
        "Cycle C": protocol_summary.get("cycle_C"),
        "Formation cycles": formation_cycles,
    }
    # Add some sample metadata to overall summary for convenience
    for col in SAMPLE_METADATA_TO_DATA:
        overall[col] = sample_data.get(col)

    # Calculate additional quantities from cycle_summary and add to overall_summary
    if cycle_summary_df.is_empty():
        logger.info("No cycles found for %s", sample_id)
    elif len(cycle_summary_df["Cycle"]) == 1 and cycle_summary_df["Discharge capacity (mAh)"][-1].is_null():
        logger.info("No complete cycles found for %s", sample_id)
    else:  # Analyse the cycling data
        formed = cycle_summary_df["Cycle"][-1] > formation_cycles if formation_cycles else False
        has_mass = "Specific discharge capacity (mAh/g)" in cycle_summary_df.columns

        last_idx = -2 if cycle_summary_df["Discharge capacity (mAh)"][-1] is None else -1
        overall["Number of cycles"] = cycle_summary_df["Cycle"].max()
        overall["First formation coulombic efficiency (%)"] = cycle_summary_df["Coulombic efficiency (%)"][0]
        overall["Last coulombic efficiency (%)"] = cycle_summary_df["Coulombic efficiency (%)"][last_idx]

        if has_mass:
            overall["First formation specific discharge capacity (mAh/g)"] = cycle_summary_df[
                "Specific discharge capacity (mAh/g)"
            ][0]
            overall["Last specific discharge capacity (mAh/g)"] = cycle_summary_df[
                "Specific discharge capacity (mAh/g)"
            ][last_idx]

        if formed:
            assert isinstance(formation_cycles, int)  # noqa: S101
            overall["Initial coulombic efficiency (%)"] = cycle_summary_df["Coulombic efficiency (%)"][formation_cycles]
            overall["Capacity loss (%)"] = 100 - cycle_summary_df["Normalised discharge capacity (%)"][last_idx]
            overall["Formation average voltage (V)"] = cycle_summary_df["Charge average voltage (V)"][
                :formation_cycles
            ].mean()
            overall["Formation average current (A)"] = cycle_summary_df["Charge average current (A)"][
                :formation_cycles
            ].mean()

            overall["Initial delta V (V)"] = cycle_summary_df["Delta V (V)"][formation_cycles]
            if has_mass:
                overall["Initial specific discharge capacity (mAh/g)"] = cycle_summary_df[
                    "Specific discharge capacity (mAh/g)"
                ][formation_cycles]

            # Calculate cycles to x% of initial discharge capacity
            def _find_first_element(arr: np.ndarray, start_idx: int) -> int | None:
                """Find first element in array that is 1 where at least 1 of the next 2 elements are also 1.

                Cycles are 1-indexed and arrays are 0-indexed, so this gives the first cycle BEFORE a condition is met.
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
            norm = np.array(cycle_summary_df["Normalised discharge capacity (%)"])
            for pcent in pcents:
                abs_cycle = _find_first_element(norm < pcent, formation_cycles)
                if abs_cycle is not None:
                    overall[f"Cycles to {pcent}% capacity"] = abs_cycle - formation_cycles
            norm = np.array(cycle_summary_df["Normalised discharge energy (%)"])
            for pcent in pcents:
                abs_cycle = _find_first_element(norm < pcent, formation_cycles)
                if abs_cycle is not None:
                    overall[f"Cycles to {pcent}% energy"] = abs_cycle - formation_cycles

    # If assembly history is available, calculate times between steps
    assembly_history = sample_data.get("Assembly history", [])
    if isinstance(assembly_history, str):
        assembly_history = json.loads(assembly_history)
    job_start = min_with_none(
        [
            df["uts"][0] if not df.is_empty() else None,
            eis_df["uts"][0] if eis_df is not None and not eis_df.is_empty() else None,
        ]
    )
    if assembly_history and isinstance(assembly_history, list) and job_start is not None:
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
            overall["Electrolyte to press (s)"] = press - first_electrolyte if first_electrolyte and press else None
            overall["Electrolyte to electrode (s)"] = (
                cover_electrolyte - first_electrolyte if first_electrolyte and cover_electrolyte else None
            )
            overall["Electrode to protection (s)"] = job_start - cover_electrolyte if cover_electrolyte else None
        overall["Press to protection (s)"] = job_start - press if press else None

    return overall


def update_results(overall: dict, job_data: list[dict]) -> None:
    """Update results table with overall info."""
    # Check current status and pipeline (may be more recenty than snapshot)

    snapshot_pipeline = job_data[-1].get("Pipeline") if job_data else None
    last_snapshot = job_data[-1].get("Last snapshot") if job_data else None

    sample_id = overall.get("Sample ID")
    pipeline, status = None, None
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `Pipeline`, `Job ID` FROM pipelines WHERE `Sample ID` = ?", (sample_id,))
        row = cursor.fetchone()
        if row:
            pipeline = row[0]
            job_id = row[1]
            if job_id:
                cursor.execute("SELECT `Status` FROM jobs WHERE `Job ID` = ?", (f"{job_id}",))
                status = cursor.fetchone()[0]

    # Update the database with some of the results
    flag = None
    if pipeline:
        if (cap_loss := overall.get("Capacity loss (%)")) and cap_loss > 50:
            flag = "🪫"
        if (form_eff := overall.get("First formation coulombic efficiency (%)")) and form_eff < 50:
            flag = "🚩"
        if (init_eff := overall.get("Initial coulombic efficiency (%)")) and init_eff < 50:
            flag = "🚩"
        if (init_cap := overall.get("Initial specific discharge capacity (mAh/g)")) and init_cap < 50:
            flag = "🚩"
    update_row = {
        "Pipeline": pipeline,
        "Status": status,
        "Flag": flag,
        "Number of cycles": overall.get("Number of cycles"),
        "Capacity loss (%)": overall.get("Capacity loss (%)"),
        "Max voltage (V)": overall.get("Max voltage (V)"),
        "Formation C": overall.get("Formation C"),
        "Cycling C": overall.get("Cycle C"),
        "First formation efficiency (%)": overall.get("First formation coulombic efficiency (%)"),
        "Initial specific discharge capacity (mAh/g)": overall.get("Initial specific discharge capacity (mAh/g)"),
        "Initial efficiency (%)": overall.get("Initial coulombic efficiency (%)"),
        "Last specific discharge capacity (mAh/g)": overall.get("Last specific discharge capacity (mAh/g)"),
        "Last efficiency (%)": overall.get("Last coulombic efficiency (%)"),
        "Last analysis": datetime.now(timezone.utc).isoformat(),
        # Only add the following keys if they are not None, otherwise they set to NULL in database
        **({"Last snapshot": last_snapshot} if last_snapshot else {}),
        **({"Snapshot pipeline": snapshot_pipeline} if snapshot_pipeline else {}),
    }
    # round any floats to 3 decimal places
    for k, v in update_row.items():
        if isinstance(v, float):
            update_row[k] = round(v, 3)

    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cursor = conn.cursor()
        # insert a row with sampleid if it doesn't exist
        cursor.execute("INSERT OR IGNORE INTO results (`Sample ID`) VALUES (?)", (sample_id,))
        # update the row
        columns = ", ".join([f"`{k}` = ?" for k in update_row])
        cursor.execute(
            f"UPDATE results SET {columns} WHERE `Sample ID` = ?",  # noqa: S608
            (*update_row.values(), sample_id),
        )


def analyse_sample(sample_id: str) -> SampleDataBundle:
    """Analyse a single sample.

    Will search for the sample in the processed snapshots folder and analyse the cycling data.

    """
    sample_folder = get_sample_folder(sample_id)
    job_files = list(sample_folder.rglob("snapshot.*"))

    # Read dfs into the correct order
    job_files, dfs, metadatas = read_and_order_job_files(job_files)

    # Merge into one df, plus optional eis df
    df, eis_df = merge_dfs(dfs)

    # Merge metadatas together
    metadata = merge_metadata(job_files, metadatas)

    # Get sample and job data
    sample_data = metadata.get("sample_data", {})
    job_data = metadata.get("job_data")

    # Extract info from the protocol information
    protocol_summary = extract_voltage_crates(job_data) if job_data else {}

    # Get the per-cycle dataframe
    summary_df, protocol_summary = analyse_cycles(
        df,
        mass_mg=sample_data.get("Cathode active material mass (mg)"),
        protocol_summary=protocol_summary,
    )

    overall = analyse_overall(
        df,
        eis_df,
        metadata,
        protocol_summary,
        summary_df,
    )

    # Get the shrunk dataframe
    shrunk_df = shrink_df(df)

    # Save the data
    df.write_parquet(sample_folder / f"full.{sample_id}.parquet", metadata={"AURORA:metadata": json.dumps(metadata)})
    if shrunk_df is not None:
        shrunk_df.write_parquet(sample_folder / f"shrunk.{sample_id}.parquet")
    if eis_df is not None:
        eis_df.write_parquet(sample_folder / f"eis.{sample_id}.parquet")
    if summary_df is not None:
        summary_df.write_parquet(sample_folder / f"cycles.{sample_id}.parquet")
    if overall is not None:
        with (sample_folder / f"overall.{sample_id}.json").open("w") as f:
            json.dump(overall, f, indent=4)
    if metadata is not None:
        with (sample_folder / f"metadata.{sample_id}.json").open("w") as f:
            json.dump(metadata, f, indent=4)

    return SampleDataBundle(
        sample_id=sample_id,
        cycling=df,
        cycling_shrunk=shrunk_df,
        eis=eis_df,
        cycles_summary=summary_df,
        overall_summary=overall,
        metadata=metadata,
    )


def update_sample_metadata(sample_ids: str | list[str]) -> None:
    """Update "sample_data" in metadata of all files.

    Updates full.*.h5, full.*.parquet, cycles.*.json, overall.*.json, metadata.*.json.

    Args:
        sample_ids: sample id or list of sample ids to update

    """
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    for sample_id in sample_ids:
        # Get updated database data
        sample_data = get_sample_data(sample_id)

        sample_folder = get_sample_folder(sample_id)

        # HDF5 full file
        hdf5_file = sample_folder / f"full.{sample_id}.h5"
        if hdf5_file.exists():
            with h5py.File(hdf5_file, "a") as f:
                metadata = json.loads(f["metadata"][()])
                metadata["sample_data"] = sample_data
                del f["metadata"]
                f.create_dataset("metadata", data=json.dumps(metadata))

        # Parquet full file
        # Doesn't seem like there is a way to do this without a full file rewrite
        full_file = sample_folder / f"full.{sample_id}.parquet"
        metadata = read_metadata(full_file)
        df = pl.read_parquet(full_file)
        metadata["sample_data"] = sample_data
        df = read_cycling(full_file)
        df.write_parquet(
            sample_folder / f"full.{sample_id}.parquet", metadata={"AURORA:metadata": json.dumps(metadata)}
        )

        # JSON cycles file
        json_file = sample_folder / f"cycles.{sample_id}.json"
        if json_file.exists():
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                data["metadata"]["sample_data"] = sample_data
                for col in SAMPLE_METADATA_TO_DATA:
                    data["data"][col] = sample_data.get(col)
            with json_file.open("w", encoding="utf-8") as f:
                json_dump_compress_lists(data, f, indent=4)

        # JSON overall file
        overall_file = sample_folder / f"overall.{sample_id}.json"
        if overall_file.exists():
            with overall_file.open("r", encoding="utf-8") as f:
                overall = json.load(f)
                for col in SAMPLE_METADATA_TO_DATA:
                    overall[col] = sample_data.get(col)
            with overall_file.open("w", encoding="utf-8") as f:
                json.dump(overall, f, indent=4)

        # JSON metadata file
        metadata_file = sample_folder / f"metadata.{sample_id}.json"
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
                metadata["sample_data"] = sample_data
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)


def shrink_df(df: pl.DataFrame) -> pl.DataFrame:
    """Find the full.x.h5 file for the sample and save a lossy, compressed version."""
    # Only keep a few columns
    df = df.select("uts", "V (V)", "I (A)", "dQ (mAh)", "Cycle")

    # Use the LTTB downsampler to reduce the number of data points
    original_length = len(df)
    new_length = min(original_length, original_length // 20 + 1000, 50000)
    if new_length < 3:
        return df.with_columns(pl.lit(None).alias("dQ/dV (mAh/V)"))

    # Calculate cumulative sum
    df = df.with_columns(pl.col("dQ (mAh)").cum_sum().alias("Q (mAh)"))

    # Define function to apply per group
    def compute_dqdv(group_df: pl.DataFrame) -> pl.DataFrame:
        dqdv = calc_dqdv(group_df["V (V)"].to_numpy(), group_df["Q (mAh)"].to_numpy(), group_df["dQ (mAh)"].to_numpy())
        return group_df.with_columns(pl.Series("dQ/dV (mAh/V)", dqdv))

    # Apply to each cycle group
    df = df.group_by("Cycle", maintain_order=True).map_groups(compute_dqdv).sort("uts")

    # Reduce precision of some columns
    df.cast(
        {
            "Cycle": pl.Int16,
            "V (V)": pl.Float32,
            "I (A)": pl.Float32,
            "dQ (mAh)": pl.Float32,
            "dQ/dV (mAh/V)": pl.Float32,
            "Q (mAh)": pl.Float32,
        }
    )
    s_ds_V = MinMaxLTTBDownsampler().downsample(df["uts"], df["V (V)"], n_out=new_length)
    s_ds_I = MinMaxLTTBDownsampler().downsample(df["uts"], df["I (A)"], n_out=new_length)
    ind = np.sort(np.concatenate([s_ds_V, s_ds_I]))

    # Downsample the dataframe
    df = df[ind]

    # Recalculate dQ so it cumulates correctly after downsampling
    return df.with_columns(pl.col("Q (mAh)").diff().fill_null(0).alias("dQ (mAh)")).drop("Q (mAh)")


def shrink_all_samples(sampleid_contains: str = "") -> None:
    """Shrink all samples in the processed snapshots folder.

    Args:
        sampleid_contains (str, optional): only shrink samples with this string in the sampleid

    """
    for batch_folder in Path(CONFIG["Processed snapshots folder path"]).iterdir():
        if batch_folder.is_dir():
            for sample_folder in batch_folder.iterdir():
                sample_id = sample_folder.name
                if sampleid_contains and sampleid_contains not in sample_id:
                    continue
                try:
                    df = get_cycling(sample_id)
                    df = shrink_df(df)
                    df.write_parquet(sample_folder / f"shrunk.{sample_id}.parquet")
                    logger.info("Shrunk %s", sample_id)
                except (KeyError, ValueError, PermissionError, RuntimeError, FileNotFoundError):
                    logger.exception("Failed to shrink %s", sample_id)


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
        samples_to_analyse = [
            r[0] for r in results if r[0] and (not r[1] or not r[2] or parse_datetime(r[1]) > parse_datetime(r[2]))
        ]
    elif mode == "if_not_exists":
        with sqlite3.connect(CONFIG["Database path"]) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT `Sample ID` FROM results WHERE `Last analysis` IS NULL")
            results = cursor.fetchall()
        samples_to_analyse = [r[0] for r in results]
    else:
        samples_to_analyse = []

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
    summary_dfs = []
    overall_dicts = []
    metadata: dict[str, dict] = {"sample_metadata": {}}
    for sample in samples:
        # get the anaylsed data
        summary_df = get_cycles_summary(sample)
        if summary_df is not None:
            summary_df = summary_df.with_columns(pl.lit(sample).alias("Sample ID"))
            summary_dfs.append(summary_df)
            overall_dicts.append(get_overall_summary(sample))
            metadata["sample_metadata"][sample] = get_metadata(sample)
    if len(summary_dfs) == 0:
        msg = "No cycling data found for any sample"
        raise ValueError(msg)
    summary_df = pl.concat(summary_dfs, how="diagonal")
    overall_df = pl.DataFrame(overall_dicts, strict=False)

    # update the metadata
    metadata["provenance"] = {
        "aurora_metadata": {
            "batch_analysis": {
                "repo_url": __url__,
                "repo_version": __version__,
                "method": "analysis.analyse_batch",
                "datetime": datetime.now(timezone.utc).isoformat(),
            },
        },
    }
    with Workbook(save_location / f"batch.{plot_name}.xlsx") as wb:
        summary_df.write_excel(
            workbook=wb,
            worksheet="Data by cycle",
            table_style="Table Style Medium 16",
            autofit=True,
        )
        overall_df.write_excel(
            workbook=wb,
            worksheet="Results by sample",
            table_style="Table Style Medium 16",
            autofit=True,
        )

    with (save_location / f"batch.{plot_name}.json").open("w", encoding="utf-8") as f:
        json.dump({"data": summary_df.to_dict(as_series=False), "metadata": metadata}, f, indent=4)


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
        except (ValueError, KeyError, PermissionError, RuntimeError, FileNotFoundError):
            logger.exception("Failed to analyse %s", plot_name)


def moving_average(x: np.ndarray, npoints: int = 11) -> np.ndarray:
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
