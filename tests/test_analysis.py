"""Test analysis.py."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aurora_cycler_manager.analysis import analyse_sample, calc_dqdv, update_sample_metadata
from aurora_cycler_manager.data_bundle import read_hdf_cycling, read_hdf_metadata
from aurora_cycler_manager.database_funcs import update_sample_label
from aurora_cycler_manager.eclab_harvester import convert_all_mprs
from aurora_cycler_manager.neware_harvester import convert_all_neware_data


class TestAnalysis:
    """Test the analysis functions."""

    def test_analyse_eclab_sample(self, reset_all) -> None:
        """Generate test data, run analysis."""
        convert_all_mprs()
        results = analyse_sample("250116_kigr_gen6_01")
        df = results["data_cycling"]
        cycle_df = results.get("data_cycle_summary")
        metadata = results["metadata"]

        # DataFrame checks
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(k in df.columns for k in ["uts", "V (V)", "I (A)", "Cycle"])
        assert all(df["uts"] > 1.7e9)
        assert all(df["V (V)"] > 0)
        assert all(df["V (V)"] < 5)

        # cycle dict checks
        assert isinstance(cycle_df, pd.DataFrame)
        assert len(cycle_df["Cycle"]) == cycle_df["Cycle"].iloc[-1]

        # DataFrame-cycle consistency
        assert df["Cycle"].max() == cycle_df["Cycle"].iloc[-1]

        # metadata checks
        assert isinstance(metadata, dict)
        assert all(k in metadata for k in ["sample_data", "job_data", "provenance"])
        assert metadata["sample_data"]["Sample ID"] == "250116_kigr_gen6_01"

    def test_analyse_neware_sample(self, reset_all) -> None:
        """Generate test data, run analysis."""
        convert_all_neware_data()
        results = analyse_sample("commercial_cell_009")
        df = results["data_cycling"]
        cycle_df = results.get("data_cycle_summary")
        metadata = results["metadata"]

        # DataFrame checks
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(k in df.columns for k in ["uts", "V (V)", "I (A)", "Cycle"])
        assert all(df["uts"] > 1.7e9)
        assert all(df["V (V)"] > 0)
        assert all(df["V (V)"] < 5)

        # cycle dict checks
        assert isinstance(cycle_df, pd.DataFrame)
        assert len(cycle_df["Cycle"]) == cycle_df["Cycle"].iloc[-1]

        # DataFrame-cycle consistency
        assert df["Cycle"].max() == cycle_df["Cycle"].iloc[-1]

        # metadata checks
        assert isinstance(metadata, dict)
        assert all(k in metadata for k in ["sample_data", "job_data", "provenance"])
        assert metadata["sample_data"]["Sample ID"] == "commercial_cell_009"

    def test_update_sample_metadata(self, reset_all, test_dir: Path) -> None:
        """Test update sample metadata."""
        sample_folder = test_dir / "snapshots" / "250116_kigr_gen6" / "250116_kigr_gen6_01"

        # Files which will be written to
        cycles_file = sample_folder / "cycles.250116_kigr_gen6_01.json"
        full_file = sample_folder / "full.250116_kigr_gen6_01.h5"

        # Convert the data to cycles.*.json and full.*.h5 and read the data
        convert_all_mprs()
        analyse_sample("250116_kigr_gen6_01")
        with cycles_file.open("r") as f:
            cycles_data_before = json.load(f)
        full_data_before = read_hdf_cycling(full_file)
        full_metadata_before = read_hdf_metadata(full_file)

        # Change the sample metadata
        update_sample_label("250116_kigr_gen6_01", "This should be written to the file")
        update_sample_metadata("250116_kigr_gen6_01")

        # Reread the data files
        with cycles_file.open("r") as f:
            cycles_data_after = json.load(f)
        full_data_after = read_hdf_cycling(full_file)
        full_metadata_after = read_hdf_metadata(full_file)

        # Check that the label has been updated
        assert cycles_data_after["data"]["Label"] == "This should be written to the file"
        assert cycles_data_after["metadata"]["sample_data"]["Label"] == "This should be written to the file"
        assert cycles_data_before["data"]["Label"] != cycles_data_after["data"]["Label"]

        assert full_metadata_after["sample_data"]["Label"] == "This should be written to the file"
        assert full_metadata_before["sample_data"]["Label"] != full_metadata_after["sample_data"]["Label"]

        # The rest should be the same
        cycles_data_before["data"].pop("Label")
        cycles_data_after["data"].pop("Label")
        cycles_data_before["metadata"]["sample_data"].pop("Label")
        cycles_data_after["metadata"]["sample_data"].pop("Label")
        assert cycles_data_before == cycles_data_after

        assert full_data_before.equals(full_data_after)

        full_metadata_before["sample_data"].pop("Label")
        full_metadata_after["sample_data"].pop("Label")
        assert full_metadata_before == full_metadata_after

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
