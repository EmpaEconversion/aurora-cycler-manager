"""Unit tests for database_funcs.py."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from aurora_cycler_manager.database_funcs import (
    _pre_check_sample_file,
    _recalculate_sample_data,
    add_samples_from_file,
    add_samples_from_object,
    delete_samples,
    get_all_sampleids,
    get_batch_details,
    get_job_data,
    get_sample_data,
    modify_batch,
    remove_batch,
    save_or_overwrite_batch,
    update_sample_label,
)


class TestPreCheckSampleFile:
    """Unit tests for database functions."""

    def test_non_existent(self) -> None:
        """Should raise error if file not found."""
        # Test with a non-existent file
        non_existent_file = Path("non_existent_file.json")
        with pytest.raises(FileNotFoundError, match=".*does not exist.*"):
            _pre_check_sample_file(non_existent_file)

    def test_too_big(self) -> None:
        """Should raise error if file is over 2 MB."""
        try:
            large_file = Path("large_file.json")
            with large_file.open("wb") as f:
                f.write(b"0" * (2 * 1024 * 1024 + 1))
            with pytest.raises(ValueError, match=".*is over 2 MB.*"):
                _pre_check_sample_file(large_file)
        finally:
            # Clean up the dummy file
            large_file.unlink(missing_ok=True)

    def test_not_json(self) -> None:
        """Should raise error if file is not JSON."""
        sample_file = Path(__file__).parent / "test_data" / "samples" / "240620_kigr_gen2.json"
        # copy to a temp file with a different extension
        try:
            temp_file = sample_file.with_suffix(".txt")
            temp_file.write_text(sample_file.read_text())
            with pytest.raises(ValueError, match=".*not a json file.*"):
                _pre_check_sample_file(temp_file)
        finally:
            # Clean up the dummy file
            temp_file.unlink(missing_ok=True)

    def test_valid_file(self) -> None:
        """Should not raise error if file is valid."""
        sample_file = Path(__file__).parent / "test_data" / "samples" / "240620_kigr_gen2.json"
        _pre_check_sample_file(sample_file)  # Should not raise any error


class TestRecalculateSampleData:
    """Takes a dataframe and recalculates/adds some columns."""

    sample_file = Path(__file__).parent / "test_data" / "samples" / "240620_kigr_gen2.json"
    df = pd.read_json(sample_file, orient="records")

    def test_missing_sampleid(self) -> None:
        """Should raise error if sampleid is missing."""
        df = self.df.copy()
        df = df.drop(columns=["Sample ID"])
        with pytest.raises(ValueError, match=".*does not contain a 'Sample ID' column.*"):
            _recalculate_sample_data(df)

    def test_duplicate_row(self) -> None:
        """Should raise error if there are duplicate rows."""
        df = self.df.copy()
        df = pd.concat([df, df])
        with pytest.raises(ValueError, match=".*contains duplicate.*"):
            _recalculate_sample_data(df)

    def test_nan_sampleid(self) -> None:
        """Should raise error if sampleid is NaN."""
        df = self.df.copy()
        df.loc[0, "Sample ID"] = None
        with pytest.raises(ValueError, match=".*contains NaN.*"):
            _recalculate_sample_data(df)

    def test_backticks(self) -> None:
        """Should raise error if any column name contains backticks."""
        df = self.df.copy()
        df = df.rename(columns={"Anode Type": "Bobby tables `; DROP TABLE samples"})
        with pytest.raises(ValueError, match=".*cannot contain backticks.*"):
            _recalculate_sample_data(df)

    def test_column_config(self) -> None:
        """Should raise error if any column name is not in the config."""
        df = self.df.copy()
        new_df = _recalculate_sample_data(df)
        # Columns should be switched to the column names in the config
        assert "Anode Weight (mg)" in df.columns
        assert "Anode mass (mg)" in new_df.columns

    def fill_run_id(self) -> None:
        """Fill in empty run ID."""
        df = self.df.copy()
        df.loc[0, "Run ID"] = None
        new_df = _recalculate_sample_data(df)
        assert all(run_id == "240620_kigr_gen2" for run_id in new_df["Run ID"])

    def test_recalculate(self) -> None:
        """Should recalculate the sample data."""
        df = self.df.copy()

        df.loc[0, "Anode Weight (mg)"] = 223
        df.loc[0, "Anode Current Collector Weight (mg)"] = 23
        df.loc[0, "Anode Active Material Weight Fraction"] = 0.9
        df.loc[0, "Anode Balancing Specific Capacity (mAh/g)"] = 1000

        df.loc[0, "Cathode Weight (mg)"] = 123
        df.loc[0, "Cathode Current Collector Weight (mg)"] = 23
        df.loc[0, "Cathode Active Material Weight Fraction"] = 0.9
        df.loc[0, "Cathode Balancing Specific Capacity (mAh/g)"] = 1000

        df.loc[0, "Anode Diameter (mm)"] = 100
        df["Cathode Diameter (mm)"] = df["Cathode Diameter (mm)"].astype(float)
        df.loc[0, "Cathode Diameter (mm)"] = 100 / 2**0.5

        new_df = _recalculate_sample_data(df)

        assert new_df.loc[0, "Anode active material mass (mg)"] == 180.0
        assert new_df.loc[0, "Anode balancing capacity (mAh)"] == 180.0

        assert new_df.loc[0, "Cathode active material mass (mg)"] == 90.0
        assert new_df.loc[0, "Cathode balancing capacity (mAh)"] == 90.0

        assert new_df.loc[0, "N:P ratio overlap factor"] == pytest.approx(0.5)

        assert new_df.loc[0, "N:P ratio"] == pytest.approx(1)


class TestSampleFunctions:
    """Test the various functions for manipulating the samples table."""

    # Make backup to restore from for each test
    db_path = Path(__file__).parent / "test_data" / "database" / "test_database.db"
    sample_file = Path(__file__).parent / "test_data" / "samples" / "240620_kigr_gen2.json"
    shutil.copyfile(db_path, db_path.with_suffix(".bak"))

    def test_update_sample_label(self) -> None:
        """Add sample from file and manipulate the samples table."""
        try:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

            # Add samples from file
            add_samples_from_file(self.sample_file)

            # Update a label
            update_sample_label("240620_kigr_gen2_01", "foo")
            sample_data = get_sample_data("240620_kigr_gen2_01")
            assert sample_data["Label"] == "foo"

            update_sample_label("240620_kigr_gen2_01", "bar")
            sample_data = get_sample_data("240620_kigr_gen2_01")
            assert sample_data["Label"] == "bar"

            # Delete some samples
            sample_ids = get_all_sampleids()
            assert "240620_kigr_gen2_01" in sample_ids
            delete_samples("240620_kigr_gen2_01")
            sample_ids = get_all_sampleids()
            assert "240620_kigr_gen2_01" not in sample_ids
            delete_samples(["240620_kigr_gen2_02", "240620_kigr_gen2_03"])
            sample_ids = get_all_sampleids()
            assert "240620_kigr_gen2_02" not in sample_ids
            assert "240620_kigr_gen2_03" not in sample_ids

        finally:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

    def test_add_samples_from_object(self) -> None:
        """Test thats samples can be added from a dict."""
        try:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

            with self.sample_file.open("r") as f:
                sample_dict = json.load(f)
            add_samples_from_object(sample_dict)
            sample_ids = get_all_sampleids()
            assert "240620_kigr_gen2_01" in sample_ids

        finally:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

    def test_batch_operations(self) -> None:
        """Create, modify, delete batches in the database."""
        try:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

            # Add samples from file
            add_samples_from_file(self.sample_file)

            # Create a batch
            save_or_overwrite_batch(
                "Batch please",
                "A test batch for testing",
                [
                    "240620_kigr_gen2_01",
                    "240620_kigr_gen2_02",
                    "240620_kigr_gen2_03",
                ],
            )

            # Check the batch exists
            batch_details = get_batch_details()
            assert "Batch please" in batch_details
            assert batch_details["Batch please"]["description"] == "A test batch for testing"

            # Try overwriting - it should raise a ValueError
            with pytest.raises(ValueError, match=".*already exists.*"):
                save_or_overwrite_batch(
                    "Batch please",
                    "A test batch for testing",
                    [
                        "240620_kigr_gen2_01",
                        "240620_kigr_gen2_02",
                        "240620_kigr_gen2_03",
                    ],
                )

            # Check the batch didn't change
            batch_details = get_batch_details()
            assert batch_details["Batch please"]["description"] == "A test batch for testing"

            # Try overwriting with same name and force overwrite
            save_or_overwrite_batch(
                "Batch please",
                "It has the same name but I'm forcing it to overwrite",
                [
                    "240620_kigr_gen2_04",
                    "240620_kigr_gen2_05",
                    "240620_kigr_gen2_06",
                ],
                overwrite=True,
            )
            # Confirm it overwrites
            batch_details = get_batch_details()
            assert (
                batch_details["Batch please"]["description"] == "It has the same name but I'm forcing it to overwrite"
            )
            assert "240620_kigr_gen2_04" in batch_details["Batch please"]["samples"]

            # Modify everything in the batch
            modify_batch(
                "Batch please",
                "Batch but different label",
                "And a different description",
                [
                    "240620_kigr_gen2_07",
                    "240620_kigr_gen2_08",
                    "240620_kigr_gen2_09",
                ],
            )
            batch_details = get_batch_details()
            assert "Batch please" not in batch_details
            assert "Batch but different label" in batch_details
            assert batch_details["Batch but different label"]["description"] == "And a different description"
            assert "240620_kigr_gen2_08" in batch_details["Batch but different label"]["samples"]

            # Remove the batch
            remove_batch("Batch but different label")
            batch_details = get_batch_details()
            assert "Batch but different label" not in batch_details

        finally:
            shutil.copyfile(self.db_path.with_suffix(".bak"), self.db_path)

    def test_get_job_data(self) -> None:
        """Test getting job data from database."""
        job_data = get_job_data("nw4-120-1-1-48")
        assert job_data["Job ID"] == "nw4-120-1-1-48"
        assert job_data["Server label"] == "nw4"
        assert isinstance(job_data["Payload"], list)
