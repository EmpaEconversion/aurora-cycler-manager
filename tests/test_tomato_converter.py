"""Tests for tomato_converter.py."""

from pathlib import Path

import pandas as pd

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.tomato_converter import (
    convert_all_tomato_jsons,
    convert_tomato_json,
    get_tomato_snapshot_folder,
)


class TestGetSnapshotFolder:
    """Test the get_snapshot_folder function."""

    def test_get_snapshot_folder(self) -> None:
        """Test the get_snapshot_folder function."""
        result = get_tomato_snapshot_folder()
        expected_path = Path(__file__).parent / "test_data" / "local_snapshots" / "tomato_snapshots"
        assert result == expected_path


class TestConvertTomatoJson:
    """Test the convert_tomato_json function."""

    def test_convert_tomato_json(self) -> None:
        """Test the convert_tomato_json function."""
        snapshot_file_path = (
            Path(__file__).parent
            / "test_data"
            / "local_snapshots"
            / "tomato_snapshots"
            / "240606_svfe_gen1"
            / "240606_svfe_gen1_15"
            / "snapshot.tt1-100.json"
        )
        df, metadata = convert_tomato_json(snapshot_file_path, output_hdf_file=False)
        # DataFrame checks
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        expected_cols = ["uts", "V (V)", "I (A)", "cycle_number", "loop_number", "index", "technique"]
        assert all(col in df.columns for col in expected_cols)
        # Check time is sensible
        assert all(df["uts"] > 1.7e9)
        assert all(df["uts"] < 1.8e9)
        assert all(df["I (A)"] > -1e-3)
        assert all(df["I (A)"] < 1e-3)
        assert all(df["V (V)"] > 0)
        assert all(df["V (V)"] < 5)
        # Metadata checks
        assert isinstance(metadata, dict)
        expected_keys = ["sample_data", "job_data", "provenance"]
        assert all(key in metadata for key in expected_keys)


class TestConvertAllTomatoJson:
    """Test convert all function."""

    def test_convert_all_tomato_jsons(self) -> None:
        """Test the convert_all_tomato_jsons function."""
        snapshot_folder = get_config().get("Processed snapshots folder path")
        if snapshot_folder is None:
            msg = "Snapshots folder path not set in config"
            raise ValueError(msg)
        expected_snapshots = [
            Path(snapshot_folder) / "240606_svfe_gen1" / "240606_svfe_gen1_15" / "snapshot.tt1-100.h5",
            Path(snapshot_folder) / "240606_svfe_gen1" / "240606_svfe_gen1_15" / "snapshot.tt1-68.h5",
            Path(snapshot_folder) / "240606_svfe_gen1" / "240606_svfe_gen1_16" / "snapshot.tt1-69.h5",
        ]
        try:
            convert_all_tomato_jsons()
            # check that the hdf5 outputs were created
            assert snapshot_folder is not None, "Snapshots folder path not set in config"
            assert all(snapshot.exists() for snapshot in expected_snapshots), "Not all expected snapshots were created"
        finally:
            # Clean up the created files
            for snapshot in expected_snapshots:
                if snapshot.exists():
                    snapshot.unlink()
