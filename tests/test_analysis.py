"""Test analysis.py."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from aurora_cycler_manager.analysis import (
    analyse_cycles,
    analyse_overall,
    analyse_sample,
    calc_dqdv,
    extract_voltage_crates,
    merge_dfs,
    merge_metadata,
    read_and_order_job_files,
    shrink_df,
    update_results,
    update_sample_metadata,
)
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.data_bundle import get_sample_folder, read_cycling, read_metadata
from aurora_cycler_manager.database_funcs import update_sample_label
from aurora_cycler_manager.eclab_harvester import convert_all_mprs
from aurora_cycler_manager.neware_harvester import convert_all_neware_data


class TestAnalysis:
    """Test the analysis functions."""

    def test_analysis_steps_eclab(self, reset_all) -> None:
        """Test individual analysis steps using EC-lab data."""
        convert_all_mprs()

        sample_id = "250116_kigr_gen6_01"

        folder = get_sample_folder("250116_kigr_gen6_01")
        snapshots_folder = folder / "snapshots"
        assert snapshots_folder.exists()
        assert any(snapshots_folder.glob("snapshot.*.parquet"))

        lens = []
        for file in snapshots_folder.glob("snapshot.*.parquet"):
            df = read_cycling(file)
            assert isinstance(df, pl.DataFrame)
            lens.append(len(df))

        job_files, dfs, metadatas = read_and_order_job_files(list(snapshots_folder.glob("snapshot.*.parquet")))
        assert all(isinstance(j, Path) for j in job_files)
        assert len(job_files) == len(list(snapshots_folder.glob("snapshot.*.parquet")))
        assert all(isinstance(m, dict) for m in metadatas)

        df, eis_df = merge_dfs(dfs)
        assert eis_df is None
        assert isinstance(df, pl.DataFrame)
        assert len(df) == sum(lens)
        assert df["Cycle"][-1] == 3

        metadata = merge_metadata(job_files, metadatas)
        assert isinstance(metadata, dict)
        assert metadata.get("Sample ID") == metadatas[0].get("Sample ID")

        # Get sample and job data
        sample_data = metadata.get("sample_data", {})
        job_data = metadata.get("job_data")
        assert isinstance(job_data, list)

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
        assert isinstance(overall, dict)
        assert overall.get("Sample ID") == "250116_kigr_gen6_01"
        assert overall.get("Number of cycles") == 3
        assert overall.get("Formation cycles") == 3
        assert overall.get("Max voltage (V)") == 4.9

        # Get the shrunk dataframe
        shrunk_df = shrink_df(df)
        assert len(shrunk_df) < len(df)
        assert "dQ/dV (mAh/V)" in shrunk_df.columns

        update_results(overall, job_data)

        with sqlite3.connect(get_config()["Database path"]) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM results WHERE `Sample ID` = ?",
                (sample_id,),
            )
            result = cursor.fetchone()
            assert result is not None
            result = dict(result)
            assert result["Sample ID"] == sample_id
            assert result["Number of cycles"] == 3
            assert result["First formation efficiency (%)"] == 76.112

        assert not (folder / f"full.{sample_id}.parquet").exists()
        assert not (folder / f"shrunk.{sample_id}.parquet").exists()
        assert not (folder / f"cycles.{sample_id}.parquet").exists()
        assert not (folder / f"overall.{sample_id}.json").exists()
        assert not (folder / f"metadata.{sample_id}.json").exists()

        analyse_sample(sample_id)

        assert (folder / f"full.{sample_id}.parquet").exists()
        assert (folder / f"shrunk.{sample_id}.parquet").exists()
        assert (folder / f"cycles.{sample_id}.parquet").exists()
        assert (folder / f"overall.{sample_id}.json").exists()
        assert (folder / f"metadata.{sample_id}.json").exists()

        df2 = pl.read_parquet(folder / f"full.{sample_id}.parquet")
        assert_frame_equal(df, df2)

        shrunk_df2 = pl.read_parquet(folder / f"shrunk.{sample_id}.parquet")
        assert_frame_equal(shrunk_df, shrunk_df2)

        summary_df2 = pl.read_parquet(folder / f"cycles.{sample_id}.parquet")
        assert_frame_equal(summary_df, summary_df2)

    def test_analyse_eclab_sample(self, reset_all) -> None:
        """Generate test data, run analysis."""
        convert_all_mprs()
        results = analyse_sample("250116_kigr_gen6_01")
        df = results.cycling
        cycle_df = results.cycles_summary
        metadata = results.metadata

        # DataFrame checks
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()
        assert all(k in df.columns for k in ["uts", "V (V)", "I (A)", "Cycle"])
        assert all(df["uts"] > 1.7e9)
        assert all(df["V (V)"] > 0)
        assert all(df["V (V)"] < 5)

        # cycle dict checks
        assert isinstance(cycle_df, pl.DataFrame)
        assert len(cycle_df["Cycle"]) == cycle_df["Cycle"][-1]

        # DataFrame-cycle consistency
        assert df["Cycle"].max() == cycle_df["Cycle"][-1]

        # metadata checks
        assert isinstance(metadata, dict)
        assert all(k in metadata for k in ["sample_data", "job_data", "provenance"])
        assert metadata["sample_data"]["Sample ID"] == "250116_kigr_gen6_01"

    def test_analyse_neware_sample(self, reset_all) -> None:
        """Generate test data, run analysis."""
        convert_all_neware_data()
        results = analyse_sample("commercial_cell_009")
        df = results.cycling
        cycle_df = results.cycles_summary
        metadata = results.metadata

        # DataFrame checks
        assert isinstance(df, pl.DataFrame)
        assert not df.is_empty()
        assert all(k in df.columns for k in ["uts", "V (V)", "I (A)", "Cycle"])
        assert all(df["uts"] > 1.7e9)
        assert all(df["V (V)"] > 0)
        assert all(df["V (V)"] < 5)

        # cycle dict checks
        assert isinstance(cycle_df, pl.DataFrame)
        assert len(cycle_df["Cycle"]) == cycle_df["Cycle"][-1]

        # DataFrame-cycle consistency
        assert df["Cycle"].max() == cycle_df["Cycle"][-1]

        # metadata checks
        assert isinstance(metadata, dict)
        assert all(k in metadata for k in ["sample_data", "job_data", "provenance"])
        assert metadata["sample_data"]["Sample ID"] == "commercial_cell_009"

    def test_update_sample_metadata(self, reset_all, test_dir: Path) -> None:
        """Test update sample metadata."""
        sample_id = "250116_kigr_gen6_01"
        sample_folder = get_sample_folder(sample_id)

        # Files which will be written to
        full_file = sample_folder / f"full.{sample_id}.parquet"
        cycles_file = sample_folder / f"cycles.{sample_id}.parquet"
        overall_file = sample_folder / f"overall.{sample_id}.json"
        metadata_file = sample_folder / f"metadata.{sample_id}.json"

        # Convert the data to cycles.*.json and full.*.h5 and read the data
        convert_all_mprs()
        analyse_sample(sample_id)

        full_data_before = read_cycling(full_file)
        full_metadata_before = read_metadata(full_file)
        cycles_before = pl.read_parquet(cycles_file)
        with overall_file.open("r") as f:
            overall_before = json.load(f)
        with metadata_file.open("r") as f:
            metadata_before = json.load(f)

        # Change the sample metadata
        update_sample_label("250116_kigr_gen6_01", "This should be written to the file")
        update_sample_metadata("250116_kigr_gen6_01")

        # Reread the data files
        full_data_after = read_cycling(full_file)
        full_metadata_after = read_metadata(full_file)
        cycles_after = pl.read_parquet(cycles_file)
        with overall_file.open("r") as f:
            overall_after = json.load(f)
        with metadata_file.open("r") as f:
            metadata_after = json.load(f)

        # Check that the label has been updated
        assert overall_before["Label"] != "This should be written to the file"
        assert overall_after["Label"] == "This should be written to the file"

        assert metadata_before["sample_data"]["Label"] != "This should be written to the file"
        assert metadata_after["sample_data"]["Label"] == "This should be written to the file"

        assert full_metadata_before["sample_data"]["Label"] != "This should be written to the file"
        assert full_metadata_after["sample_data"]["Label"] == "This should be written to the file"

        # # The rest should be the same
        overall_before.pop("Label")
        overall_after.pop("Label")
        assert overall_before == overall_after

        metadata_before["sample_data"].pop("Label")
        metadata_after["sample_data"].pop("Label")
        assert metadata_before == metadata_after

        full_metadata_before["sample_data"].pop("Label")
        full_metadata_after["sample_data"].pop("Label")
        assert full_metadata_before == full_metadata_after

        assert_frame_equal(cycles_before, cycles_after)
        assert_frame_equal(full_data_before, full_data_after)

    def test_dqdv(self) -> None:
        """Test the dQ/dV calculation against analytical derivative."""
        V = np.concatenate([np.linspace(0, 100, 101), np.linspace(100, 0, 101)])
        Q = np.concatenate([np.linspace(0, 10, 101) ** 2, np.linspace(10, 0, 101) ** 2])
        dQ = Q - np.pad(Q, (1, 0), mode="edge")[:-1]
        res = calc_dqdv(V, Q, dQ)

        # Analytical derivative
        dQdV_expected = np.concatenate([np.linspace(0, 2, 101), np.linspace(-2, 0, 101)])

        # Skip first and last points of charge/discharge - they are nan due to moving window average
        np.testing.assert_almost_equal(res[5:95], dQdV_expected[5:95], decimal=6)
        np.testing.assert_almost_equal(res[105:195], dQdV_expected[105:195], decimal=6)
