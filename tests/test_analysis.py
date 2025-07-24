"""Test analysis.py, generate files with tomato converter."""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from aurora_cycler_manager.analysis import (
    analyse_sample,
    calc_dqdv,
    update_sample_metadata,
)
from aurora_cycler_manager.database_funcs import (
    update_sample_label,
)
from aurora_cycler_manager.tomato_converter import (
    convert_all_tomato_jsons,
)


class TestAnalysis:
    """Test the analysis functions."""

    # Make backup to restore from for each test
    db_path = Path(__file__).parent / "test_data" / "database" / "test_database.db"
    shutil.copyfile(db_path, db_path.with_suffix(".bak"))

    def test_analyse_sample(self) -> None:
        """Generate test data, run analysis."""
        convert_all_tomato_jsons()
        df, cycle_dict, metadata = analyse_sample("240606_svfe_gen1_15")

        # DataFrame checks
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(k in df.columns for k in ["uts", "V (V)", "I (A)", "Cycle"])
        assert all(df["uts"] > 1.7e9)
        assert all(df["V (V)"] > 0)
        assert all(df["V (V)"] < 5)

        # cycle dict checks
        assert isinstance(cycle_dict, dict)
        assert isinstance(cycle_dict["Cycle"], list)
        assert len(cycle_dict["Cycle"]) == cycle_dict["Cycle"][-1]

        # DataFrame-cycle consistency
        assert df["Cycle"].max() == cycle_dict["Cycle"][-1]

        # metadata checks
        assert isinstance(metadata, dict)
        assert all(k in metadata for k in ["sample_data", "job_data", "provenance"])
        assert metadata["sample_data"]["Sample ID"] == "240606_svfe_gen1_15"

    def test_update_sample_metadata(self) -> None:
        """Test update sample metadata."""
        sample_folder = Path(__file__).parent / "test_data" / "snapshots" / "240606_svfe_gen1" / "240606_svfe_gen1_15"
        try:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

            # Files which will be written to
            cycles_file = sample_folder / "cycles.240606_svfe_gen1_15.json"
            full_file = sample_folder / "full.240606_svfe_gen1_15.h5"

            # Convert the data to cyckes.*.json and full.*.h5 and read the data
            convert_all_tomato_jsons()
            with cycles_file.open("r") as f:
                cycles_data_before = json.load(f)
            full_data_before = pd.read_hdf(full_file, "data")
            with h5py.File(full_file, "r") as f:
                full_metadata_before = json.loads(f["metadata"][()])

            # Change the sample metadata
            update_sample_label("240606_svfe_gen1_15", "This should be written to the file")
            update_sample_metadata("240606_svfe_gen1_15")

            # Reread the data files
            with cycles_file.open("r") as f:
                cycles_data_after = json.load(f)
            full_data_after = pd.read_hdf(full_file, "data")
            with h5py.File(full_file, "r") as f:
                full_metadata_after = json.loads(f["metadata"][()])

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

        finally:  # Reset db and remove files
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)
            for file in sample_folder.glob("*.h5"):
                file.unlink()
            for file in sample_folder.glob("cycles.*.json"):
                file.unlink()

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
