"""Copyright © 2025-2026, Empa.

Functions for converting between Aurora-style and BDF-style dataframes/files.
"""

import json
import logging
from pathlib import Path

import polars as pl

from aurora_cycler_manager.analysis import calc_dq
from aurora_cycler_manager.data_bundle import read_cycling, read_metadata

logger = logging.getLogger(__name__)

aurora_to_bdf_map: dict[str, str] = {
    "uts": "unix_time_second",
    "V (V)": "voltage_volt",
    "I (A)": "current_ampere",
    "Step": "step_count",
    "Cycle": "cycle_count",
    "f (Hz)": "frequency_hertz",
    "Re(Z) (ohm)": "real_impedance_ohm",
    "Im(Z) (ohm)": "imaginary_impedance_ohm",
}

bdf_to_aurora_map_extras: dict[str, str] = {
    "Unix Time / s": "uts",
    "Current / A": "I (A)",
    "Voltage / V": "V (V)",
    "Step Count / 1": "Step",
    "Cycle Count / 1": "Cycle",
    "Freqency / Hz": "f (Hz)",
    "Real Impedance / ohm": "Re(Z) (ohm)",
    "Imaginary Impedance / ohm": "Im(Z) (ohm)",
}

bdf_to_aurora_map: dict[str, str] = {
    **{v: k for k, v in aurora_to_bdf_map.items()},
    **bdf_to_aurora_map_extras,
}


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
    df = df.select([k for k in bdf_to_aurora_map if k in df.columns])
    df = df.rename(bdf_to_aurora_map, strict=False)
    if "uts" not in df:
        msg = "Aurora dataframes must include unix time in seconds."
        raise ValueError(msg)
    # Need to recalculate dQ (mAh), lost in round trip
    return calc_dq(df)


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
