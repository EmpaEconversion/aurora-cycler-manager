"""Copyright © 2025-2026, Empa.

Functions to read and write data files. Provides SampleDataBundle interface.
"""

import json
from functools import cached_property
from pathlib import Path

import h5py
import pandas as pd
import polars as pl

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.dicts import bdf_to_aurora_map
from aurora_cycler_manager.stdlib_utils import run_from_sample

CONFIG = get_config()


def read_cycling(file: str | Path) -> pl.DataFrame:
    """Read cycling data from aurora-style parquet/hdf5 file to DataFrame."""
    file = Path(file)
    if file.suffix == ".parquet":
        df = pl.read_parquet(file)
        if "voltage_volt" in df.columns:  # bdf
            return df.rename(bdf_to_aurora_map, strict=False)
        return df
    if file.suffix == ".h5":
        return pl.DataFrame(pd.read_hdf(file))
    msg = f"Unsupported file format {file.suffix}"
    raise ValueError(msg)


def read_metadata(file: str | Path) -> dict:
    """Read metadata from aurora-style parquet/hdf5 file."""
    file = Path(file)
    if file.suffix == ".parquet":
        return json.loads(pl.read_parquet_metadata(file).get("AURORA:metadata", "{}"))
    if file.suffix == ".h5":
        with h5py.File(file, "r") as f:
            return json.loads(f["metadata"][()])
    msg = f"Unsupported file format {file.suffix}"
    raise ValueError(msg)


def get_sample_folder(sample_id: str) -> Path:
    """Get sample data folder."""
    run_id = run_from_sample(sample_id)
    return CONFIG["Processed snapshots folder path"] / run_id / sample_id


def get_cycling(sample_id: str) -> pl.DataFrame:
    """Get cycling data from Sample ID."""
    folder = get_sample_folder(sample_id)
    if (data_path := folder / f"full.{sample_id}.parquet").exists():
        return read_cycling(data_path)
    if (data_path := folder / f"full.{sample_id}.h5").exists():
        return read_cycling(data_path)
    msg = "No data found."
    raise FileNotFoundError(msg)


def get_cycling_shrunk(sample_id: str) -> pl.DataFrame | None:
    """Get shrunk cycling data from Sample ID."""
    folder = get_sample_folder(sample_id)
    if (data_path := folder / f"shrunk.{sample_id}.parquet").exists():
        return read_cycling(data_path)
    if (data_path := folder / f"shrunk.{sample_id}.h5").exists():
        return read_cycling(data_path)
    return None


def get_eis(sample_id: str) -> pl.DataFrame | None:
    """Get EIS data from Sample ID."""
    folder = get_sample_folder(sample_id)
    if (data_path := folder / f"eis.{sample_id}.parquet").exists():
        return read_cycling(data_path)
    return None


def get_cycles_summary(sample_id: str) -> pl.DataFrame | None:
    """Get per-cycle summary data from Sample ID."""
    folder = get_sample_folder(sample_id)
    if (data_path := folder / f"cycles.{sample_id}.parquet").exists():
        return pl.read_parquet(data_path)
    if (data_path := folder / f"cycles.{sample_id}.json").exists():
        with data_path.open("r") as f:
            data = json.load(f)["data"]
            data = {k: v for k, v in data.items() if isinstance(v, list)}
        return pl.DataFrame(data).cast({"Cycle": pl.UInt32})
    return None


def get_overall_summary(sample_id: str) -> dict | None:
    """Get overall data, single scalar quantites from cycling."""
    folder = get_sample_folder(sample_id)
    if (data_path := folder / f"overall.{sample_id}.json").exists():
        with data_path.open("r") as f:
            return json.load(f)
    if (data_path := folder / f"cycles.{sample_id}.json").exists():
        with data_path.open("r") as f:
            data = json.load(f)["data"]
            return {k: v for k, v in data.items() if not isinstance(v, list)}
    return None


def get_metadata(sample_id: str) -> dict | None:
    """Get sample metadata dictionary."""
    folder = get_sample_folder(sample_id)
    if (data_path := folder / f"metadata.{sample_id}.json").exists():
        with data_path.open("r") as f:
            return json.load(f)
    if (data_path := folder / f"cycles.{sample_id}.json").exists():
        with data_path.open("r") as f:
            return json.load(f)["metadata"]
    if (data_path := folder / f"full.{sample_id}.h5").exists():
        with h5py.File(data_path, "r") as f:
            return json.loads(f["metadata"][()])
    return None


class SampleDataBundle:
    """Lazy-loading wrapper for sample data with support for pre-loaded data."""

    def __init__(
        self,
        sample_id: str,
        *,
        cycling: pl.DataFrame | None = None,
        cycling_shrunk: pl.DataFrame | None = None,
        eis: pl.DataFrame | None = None,
        cycles_summary: pl.DataFrame | None = None,
        overall_summary: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Initialize with sample_id and optionally pre-loaded data.

        Args:
            sample_id: Sample identifier
            cycling: Pre-loaded cycling data (optional)
            cycling_shrunk: Pre-loaded shrunk cycling data (optional)
            eis: Pre-loaded electrochemical impedance (optional)
            cycles_summary: Pre-loaded cycles summary (optional)
            overall_summary: Pre-loaded overall summary (optional)
            metadata: Pre-loaded metadata (optional)

        """
        self.sample_id = sample_id
        self._preloaded = {
            "cycling": cycling,
            "cycling_shrunk": cycling_shrunk,
            "eis": eis,
            "cycles_summary": cycles_summary,
            "overall_summary": overall_summary,
            "metadata": metadata,
        }

    @cached_property
    def cycling(self) -> pl.DataFrame | None:
        """Time series cycling data."""
        if self._preloaded["cycling"] is not None:
            return self._preloaded["cycling"]
        return get_cycling(self.sample_id)

    @cached_property
    def cycling_shrunk(self) -> pl.DataFrame | None:
        """Shrunk time series cycling data."""
        if self._preloaded["cycling_shrunk"] is not None:
            return self._preloaded["cycling_shrunk"]
        return get_cycling_shrunk(self.sample_id)

    @cached_property
    def eis(self) -> pl.DataFrame | None:
        """Shrunk time series cycling data."""
        if self._preloaded["eis"] is not None:
            return self._preloaded["eis"]
        return get_cycling_shrunk(self.sample_id)

    @cached_property
    def cycles_summary(self) -> pl.DataFrame | None:
        """Per-cycle summary data."""
        if self._preloaded["cycles_summary"] is not None:
            return self._preloaded["cycles_summary"]
        return get_cycles_summary(self.sample_id)

    @cached_property
    def overall_summary(self) -> dict | None:
        """Overall summary stats."""
        if self._preloaded["overall_summary"] is not None:
            return self._preloaded["overall_summary"]
        return get_overall_summary(self.sample_id)

    @cached_property
    def metadata(self) -> dict | None:
        """Metadata."""
        if self._preloaded["metadata"] is not None:
            return self._preloaded["metadata"]
        return get_metadata(self.sample_id)
