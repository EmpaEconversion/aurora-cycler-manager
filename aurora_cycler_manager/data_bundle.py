"""Copyright © 2025-2026, Empa.

DataBundle Typed dict and functions to read/write hdf5.
"""

import json
from pathlib import Path
from typing import TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python 3.10
    from typing_extensions import NotRequired

import h5py
import pandas as pd

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.stdlib_utils import run_from_sample

CONFIG = get_config()


class DataBundle(TypedDict):
    """Bundle of cycling data and metadata mimicking hdf5 layout."""

    data_cycling: pd.DataFrame
    data_cycling_shrunk: NotRequired[pd.DataFrame]
    data_eis: NotRequired[pd.DataFrame]
    data_cycle_summary: NotRequired[pd.DataFrame]
    data_overall_summary: NotRequired[dict[str, str | int | float]]
    metadata: dict


def read_hdf_cycling(file: str | Path) -> pd.DataFrame:
    """Read cycling data from aurora-style hdf5 file to DataFrame."""
    keys = ["data/cycling", "cycling", "data"]
    with h5py.File(file, "r") as f:
        for key in keys:
            if key in f:
                df = pd.read_hdf(file, key=key)
                if isinstance(df, pd.DataFrame):
                    return df
    msg = f"Time-series key not found, tried {', '.join(keys)}"
    raise ValueError(msg)


def read_hdf_cycling_shrunk(file: str | Path) -> pd.DataFrame | None:
    """Read cycling data from aurora-style hdf5 file to DataFrame."""
    keys = ["data/cycling_shrunk", "data_cycling_shrunk", "cycling_shrunk"]
    with h5py.File(file, "r") as f:
        for key in keys:
            if key in f:
                df = pd.read_hdf(file, key=key)
                if isinstance(df, pd.DataFrame):
                    return df
    return None


def read_hdf_eis(file: str | Path) -> pd.DataFrame | None:
    """Read EIS data from aurora-style hdf5 file to DataFrame."""
    with h5py.File(file, "r") as f:
        for key in ["data/eis", "data-eis", "eis"]:
            if key in f:
                df = pd.read_hdf(file, key=key)
                if isinstance(df, pd.DataFrame):
                    return df
    return None


def read_hdf_cycle_summary(file: str | Path) -> pd.DataFrame | None:
    """Read cycle summary from aurora-style hdf5 file to DataFrame."""
    with h5py.File(file, "r") as f:
        for key in ["data/cycle_summary", "data-cycle-summary", "data_cycle_summary"]:
            if key in f:
                df = pd.read_hdf(file, key=key)
                if isinstance(df, pd.DataFrame):
                    return df
    return None


def read_hdf_overall_summary(file: str | Path) -> dict | None:
    """Read overall summary from aurora-style hdf5 file to a dict."""
    with h5py.File(file, "r") as f:
        for key in ["data/overall_summary", "data-overall-summary", "data_overall_summary"]:
            if key in f:
                return json.loads(f[key][()])
    return None


def read_hdf_combined_summary(file: str | Path) -> pd.DataFrame | None:
    """Read cycle and overall summary from aurora-style hdf5 file, combine to one DataFrame."""
    df = read_hdf_cycle_summary(file)
    summary_dict = read_hdf_overall_summary(file)
    if df is not None and summary_dict is not None:
        df.assign(**summary_dict)
    return df


def read_hdf_metadata(file: str | Path) -> dict:
    """Read metadata from aurora-style hdf5 file."""
    with h5py.File(file, "r") as f:
        return json.loads(f["metadata"][()])


def read_hdf_data_bundle(file: str | Path) -> DataBundle:
    """Read hdf data and metadata into a data bundle."""
    data: DataBundle = {
        "data_cycling": read_hdf_cycling(file),
        "metadata": read_hdf_metadata(file),
    }
    if (eis := read_hdf_eis(file)) is not None:
        data["data_eis"] = eis
    if (cycle_summary := read_hdf_cycle_summary(file)) is not None:
        data["data_cycle_summary"] = cycle_summary
    if (overall_summary := read_hdf_overall_summary(file)) is not None:
        data["data_overall_summary"] = overall_summary
    if (shrunk := read_hdf_cycling_shrunk(file)) is not None:
        data["data_cycling_shrunk"] = shrunk
    return data


def write_hdf(data: DataBundle, output_file: str | Path) -> None:
    """Write a data bundle to hdf."""
    data["data_cycling"].to_hdf(
        output_file,
        key="data/cycling",
        mode="w",
        complib="blosc",
        complevel=9,
    )
    if (eis_df := data.get("data_eis")) is not None:
        eis_df.to_hdf(
            output_file,
            key="data/eis",
            mode="a",
            complib="blosc",
            complevel=9,
        )
    if (summary_df := data.get("data_cycle_summary")) is not None:
        summary_df.to_hdf(
            output_file,
            key="data/cycle_summary",
            mode="a",
            complib="blosc",
            complevel=9,
        )
    with h5py.File(output_file, "a") as f:
        if (overall := data.get("data_overall_summary")) is not None:
            f.create_dataset("data/overall_summary", data=json.dumps(overall))
        f.create_dataset("metadata", data=json.dumps(data["metadata"]))


def get_full_file(sample_id: str) -> Path:
    """Get Path to sample data."""
    run_id = run_from_sample(sample_id)
    return CONFIG["Processed snapshots folder path"] / run_id / sample_id / f"full.{sample_id}.h5"


def get_data_bundle(sample_id: str) -> DataBundle:
    """Get data bundle from Sample ID."""
    return read_hdf_data_bundle(get_full_file(sample_id))


def get_cycling(sample_id: str) -> pd.DataFrame:
    """Get cycling data from Sample ID."""
    return read_hdf_cycling(get_full_file(sample_id))


def get_cycling_shrunk(sample_id: str) -> pd.DataFrame | None:
    """Get cycling data from Sample ID."""
    return read_hdf_cycling_shrunk(get_full_file(sample_id))


def get_eis(sample_id: str) -> pd.DataFrame | None:
    """Get EIS data from Sample ID."""
    return read_hdf_eis(get_full_file(sample_id))


def get_cycle_summary(sample_id: str) -> pd.DataFrame | None:
    """Get per-cycle summary data from Sample ID."""
    return read_hdf_cycle_summary(get_full_file(sample_id))


def get_overall_summary(sample_id: str) -> dict | None:
    """Get per-cycle summary data from Sample ID."""
    return read_hdf_overall_summary(get_full_file(sample_id))


def get_combined_summary(sample_id: str) -> pd.DataFrame | None:
    """Get per-cycle summary data with extra info from Sample ID."""
    return read_hdf_combined_summary(get_full_file(sample_id))


def get_metadata(sample_id: str) -> dict | None:
    """Get metadata from a sample."""
    return read_hdf_metadata(get_full_file(sample_id))
