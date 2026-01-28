"""Copyright © 2025-2026, Empa.

Functions for converting between Aurora-style and BDF-style dataframes/files.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from aurora_cycler_manager.data_bundle import read_cycling, read_metadata

logger = logging.getLogger(__name__)

bdf_mapping = {
    "uts": "Unix Time / s",
    "I (A)": "Current / A",
    "V (V)": "Voltage / V",
    "Step": "Step Count / 1",
    "Cycle": "Cycle Count / 1",
}
hdf_mapping = {v: k for k, v in bdf_mapping.items()}
# also include 'machine readable' BDF codes
hdf_mapping = {
    **hdf_mapping,
    "unix_time_seconds": "uts",
    "current_ampere": "I (A)",
    "voltage_volt": "V (V)",
    "step_count_dimensionless": "Step",
    "cycle_count_dimensionless": "Cycle",
}


def aurora_to_bdf(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Convert an Aurora dataframe to BDF compliant dataframe."""
    df = pd.DataFrame(df[[c for c in df.columns if c in bdf_mapping]])
    df = df.rename(columns=bdf_mapping)
    df["Test Time / s"] = (df["Unix Time / s"] - df["Unix Time / s"].iloc[0]).round(3)
    return df


def bdf_to_aurora(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a BDF compliant dataframe to Aurora."""
    df = df[[c for c in df.columns if c in hdf_mapping]]
    df = df.rename(columns=hdf_mapping)
    if "uts" not in df:
        msg = "Aurora dataframes must include unix time in seconds."
        raise ValueError(msg)
    # Need to recalculate dQ (mAh)
    df = df.astype({"uts": "f8", "I (A)": "f8"})
    dt_s = np.concatenate([[0], df["uts"].to_numpy()[1:] - df["uts"].to_numpy()[:-1]])
    Iavg_A = np.concatenate([[0], (df["I (A)"].to_numpy()[1:] + df["I (A)"].to_numpy()[:-1]) / 2])
    df["dQ (mAh)"] = 1e3 * Iavg_A * dt_s / 3600
    return df


def aurora_hdf_to_bdf_parquet(hdf5_file: str | Path, bdf_file: str | Path | None = None) -> None:
    """Convert Aurora HDF5 file to BDF parquet file."""
    hdf5_file = Path(hdf5_file)
    df = read_cycling(hdf5_file)
    metadata = read_metadata(hdf5_file)

    # Convert to BDF style columns
    df = aurora_to_bdf(pd.DataFrame(df))

    # In parquet, metadata can be stored in pandas attrs
    df.attrs["metadata"] = metadata

    # Save parquet file
    if not bdf_file:
        bdf_file = hdf5_file.with_suffix(".bdf.parquet")
    else:
        bdf_file = Path(bdf_file).with_suffix(".bdf.parquet")
        bdf_file.parent.mkdir(exist_ok=True)
    df.to_parquet(bdf_file, compression="brotli")


def bdf_parquet_to_aurora_hdf(bdf_file: str | Path, hdf5_file: str | Path | None = None) -> None:
    """Convert BDF parquet file to Aurora hdf5 file."""
    bdf_file = Path(bdf_file)
    df = pd.read_parquet(bdf_file)
    metadata = df.attrs.get("metadata")
    df = bdf_to_aurora(df)

    # Remove both suffixes if there are two
    if bdf_file.name.endswith(".bdf.parquet"):
        bdf_file = bdf_file.with_suffix("")
    if not hdf5_file:
        hdf5_file = bdf_file.with_suffix(".h5")
    else:
        hdf5_file = Path(hdf5_file).with_suffix(".h5")
        hdf5_file.parent.mkdir(exist_ok=True)
    df.to_hdf(hdf5_file, key="data/cycling", mode="w")
    with h5py.File(hdf5_file, "a") as f:
        f.create_dataset("metadata", data=json.dumps(metadata))
