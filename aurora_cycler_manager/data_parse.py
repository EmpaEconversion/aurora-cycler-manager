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
from aurora_cycler_manager.dicts import aurora_dtypes, aurora_to_bdf_map, bdf_to_aurora_map
from aurora_cycler_manager.stdlib_utils import run_from_sample

CONFIG = get_config()


def read_cycling(file: str | Path) -> pl.DataFrame:
    """Read cycling data from aurora-style parquet/hdf5 file to DataFrame."""
    file = Path(file)
    if file.suffix == ".parquet":
        df = pl.read_parquet(file)
        if "voltage_volt" in df.columns:  # bdf
            return bdf_to_aurora(df)
        return df.cast({k: v for k, v in aurora_dtypes.items() if k in df.columns}, strict=False)
    if file.suffix == ".h5":
        df = pl.DataFrame(pd.read_hdf(file))
        return df.cast({k: v for k, v in aurora_dtypes.items() if k in df.columns}, strict=False)
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
    return CONFIG["Data folder path"] / run_id / sample_id


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


##### BDF convertsion #####


def aurora_to_bdf(df: pl.DataFrame) -> pl.DataFrame:
    """Convert an Aurora dataframe to BDF compliant dataframe."""
    df.select([k for k in aurora_to_bdf_map if k in df.columns])
    df = df.rename(aurora_to_bdf_map, strict=False)
    if df.is_empty():
        return df.with_columns(pl.lit(None).alias("test_time_second"))
    t0 = df["unix_time_second"][0]
    return df.with_columns((pl.col("unix_time_second") - t0).alias("test_time_second"))


def bdf_to_aurora(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a BDF compliant dataframe to Aurora."""
    exprs = []
    if "test_time_millisecond" in df.columns:
        exprs += [(pl.col("test_time_millisecond") / 1000).alias("test_time_second")]
    if "date_time_millisecond" in df.columns:
        exprs += [(pl.col("date_time_millisecond") / 1000).alias("unix_time_second")]
    if "cycle_dimensionless" in df.columns:
        exprs += [(pl.col("cycle_dimensionless")).alias("cycle_count")]
    df = df.with_columns(exprs)
    df = df.select([k for k in bdf_to_aurora_map if k in df.columns])
    df = df.rename(bdf_to_aurora_map, strict=False)
    if "uts" not in df:
        msg = "Aurora dataframes must include unix time in seconds."
        raise ValueError(msg)
    return df.cast({k: v for k, v in aurora_dtypes.items() if k in df.columns}, strict=False)


def aurora_to_bdf_parquet(aurora_full_file: str | Path, bdf_file: str | Path | None = None) -> None:
    """Convert Aurora full file to BDF parquet file."""
    aurora_full_file = Path(aurora_full_file)
    df = read_cycling(aurora_full_file)
    metadata = read_metadata(aurora_full_file)

    # Convert to BDF style columns
    df = aurora_to_bdf(df)

    # Save parquet file
    if not bdf_file:
        bdf_file = aurora_full_file.with_suffix(".bdf.parquet")
    else:
        bdf_file = Path(bdf_file).with_suffix(".bdf.parquet")
        bdf_file.parent.mkdir(exist_ok=True)
    df.write_parquet(bdf_file, compression="brotli", metadata={"AURORA:metadata": json.dumps(metadata)})
