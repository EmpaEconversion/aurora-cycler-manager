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
        with pytest.raises(FileNotFoundError, match=r".*does not exist.*"):
            _pre_check_sample_file(non_existent_file)

    def test_too_big(self, tmp_path) -> None:
        """Should raise error if file is over 2 MB."""
        large_file = tmp_path / "large_file.json"
        with large_file.open("wb") as f:
            f.write(b"0" * (2 * 1024 * 1024 + 1))
        with pytest.raises(ValueError, match=r".*is over 2 MB.*"):
            _pre_check_sample_file(large_file)

    def test_not_json(self, test_dir: Path, tmp_path: Path) -> None:
        """Should raise error if file is not JSON."""
        sample_file = test_dir / "samples" / "240620_kigr_gen2.json"
        temp_file = tmp_path / sample_file.with_suffix(".txt").name
        shutil.copy(sample_file, temp_file)
        with pytest.raises(ValueError, match=r".*not a json file.*"):
            _pre_check_sample_file(temp_file)

    def test_valid_file(self, test_dir: Path) -> None:
        """Should not raise error if file is valid."""
        sample_file = test_dir / "samples" / "240620_kigr_gen2.json"
        _pre_check_sample_file(sample_file)  # Should not raise any error


@pytest.fixture
def sample_file(test_dir) -> Path:
    """Return samples path."""
    return test_dir / "samples" / "240620_kigr_gen2.json"


@pytest.fixture
def sample_df(sample_file) -> pd.DataFrame:
    """Return samples file as DataFrame."""
    return pd.read_json(sample_file, orient="records")


class TestRecalculateSampleData:
    """Takes a dataframe and recalculates/adds some columns."""

    def test_missing_sampleid(self, sample_df: pd.DataFrame) -> None:
        """Should raise error if sampleid is missing."""
        sample_df = sample_df.drop(columns=["Sample ID"])
        with pytest.raises(ValueError, match=r".*does not contain a 'Sample ID' column.*"):
            _recalculate_sample_data(sample_df)

    def test_duplicate_row(self, sample_df: pd.DataFrame) -> None:
        """Should raise error if there are duplicate rows."""
        sample_df = pd.concat([sample_df, sample_df])
        with pytest.raises(ValueError, match=r".*contains duplicate.*"):
            _recalculate_sample_data(sample_df)

    def test_nan_sampleid(self, sample_df: pd.DataFrame) -> None:
        """Should raise error if sampleid is NaN."""
        sample_df.loc[0, "Sample ID"] = None
        with pytest.raises(ValueError, match=r".*contains NaN.*"):
            _recalculate_sample_data(sample_df)

    def test_backticks(self, sample_df: pd.DataFrame) -> None:
        """Should raise error if any column name contains backticks."""
        sample_df = sample_df.rename(columns={"Anode Type": "Bobby tables `; DROP TABLE samples"})
        with pytest.raises(ValueError, match=r".*cannot contain backticks.*"):
            _recalculate_sample_data(sample_df)

    def test_column_config(self, sample_df: pd.DataFrame) -> None:
        """Should raise error if any column name is not in the config."""
        new_df = _recalculate_sample_data(sample_df)
        # Columns should be switched to the column names in the config
        assert "Anode Weight (mg)" in sample_df.columns
        assert "Anode mass (mg)" in new_df.columns

    def fill_run_id(self, sample_df: pd.DataFrame) -> None:
        """Fill in empty run ID."""
        sample_df.loc[0, "Run ID"] = None
        sample_df = _recalculate_sample_data(sample_df)
        assert all(run_id == "240620_kigr_gen2" for run_id in sample_df["Run ID"])

    def test_recalculate(self, sample_df: pd.DataFrame) -> None:
        """Should recalculate the sample data."""
        sample_df.loc[0, "Anode Weight (mg)"] = 223
        sample_df.loc[0, "Anode Current Collector Weight (mg)"] = 23
        sample_df.loc[0, "Anode Active Material Weight Fraction"] = 0.9
        sample_df.loc[0, "Anode Balancing Specific Capacity (mAh/g)"] = 1000

        sample_df.loc[0, "Cathode Weight (mg)"] = 123
        sample_df.loc[0, "Cathode Current Collector Weight (mg)"] = 23
        sample_df.loc[0, "Cathode Active Material Weight Fraction"] = 0.9
        sample_df.loc[0, "Cathode Balancing Specific Capacity (mAh/g)"] = 1000

        sample_df.loc[0, "Anode Diameter (mm)"] = 100
        sample_df["Cathode Diameter (mm)"] = sample_df["Cathode Diameter (mm)"].astype(float)
        sample_df.loc[0, "Cathode Diameter (mm)"] = 100 / 2**0.5

        new_df = _recalculate_sample_data(sample_df)

        assert new_df.loc[0, "Anode active material mass (mg)"] == 180.0
        assert new_df.loc[0, "Anode balancing capacity (mAh)"] == 180.0

        assert new_df.loc[0, "Cathode active material mass (mg)"] == 90.0
        assert new_df.loc[0, "Cathode balancing capacity (mAh)"] == 90.0

        assert new_df.loc[0, "N:P ratio overlap factor"] == pytest.approx(0.5)

        assert new_df.loc[0, "N:P ratio"] == pytest.approx(1)


class TestSampleFunctions:
    """Test the various functions for manipulating the samples table."""

    def test_update_sample_label(self, reset_all, sample_file: Path) -> None:
        """Add sample from file and manipulate the samples table."""
        # Add samples from file
        add_samples_from_file(sample_file)

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

    def test_add_samples_from_object(self, reset_all, sample_file: Path) -> None:
        """Test thats samples can be added from a dict."""
        with sample_file.open("r") as f:
            sample_dict = json.load(f)
        add_samples_from_object(sample_dict)
        sample_ids = get_all_sampleids()
        assert "240620_kigr_gen2_01" in sample_ids

    def test_batch_operations(self, reset_all, sample_file: Path) -> None:
        """Create, modify, delete batches in the database."""
        # Add samples from file
        add_samples_from_file(sample_file)

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
        with pytest.raises(ValueError, match=r".*already exists.*"):
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
        assert batch_details["Batch please"]["description"] == "It has the same name but I'm forcing it to overwrite"
        assert "240620_kigr_gen2_04" in batch_details["Batch please"]["samples"]

        # Remove the batch
        remove_batch("Batch please")
        batch_details = get_batch_details()
        assert "Batch please" not in batch_details

    def test_get_job_data(self) -> None:
        """Test getting job data from database."""
        job_data = get_job_data("nw4-120-1-1-48")
        assert job_data["Job ID"] == "nw4-120-1-1-48"
        assert job_data["Server label"] == "nw4"
        assert isinstance(job_data["Payload"], list)
