"""Testing functions in the eclab_harvester.py."""

import sqlite3
from pathlib import Path

import pytest

from aurora_cycler_manager.eclab_harvester import convert_mpr


def test_convert_data(reset_all, test_dir: Path) -> None:
    """Should be able to convert mprs from different formats."""
    folder = test_dir / "eclab_harvester"
    mpr_with_date = folder / "test_C01.mpr"

    params = {
        "update_database": False,
        "sample_id": "test",
        "file_name": "test_C01.mpr",
    }

    # convert_mpr should work with Path, str, bytes
    _df, _metadata = convert_mpr(mpr_with_date, **params)
    _df, _metadata = convert_mpr(str(mpr_with_date), **params)
    with mpr_with_date.open("rb") as f:
        _df, _metadata = convert_mpr(f.read(), **params)

    # Without a sample ID it will fail
    with pytest.raises(ValueError):
        _df, _metadata = convert_mpr(mpr_with_date, update_database=False)

    # If there is no way to get the acquisition start time, it will fail
    mpr_without_date = folder / "file_2025-10-17_162649.mpr"
    with pytest.raises(ValueError):
        _df, _metadata = convert_mpr(mpr_without_date, **params)

    # If there is a matching mpl file, it will find it automatically
    mpr_with_sidecar_mpl = folder / "file_2025-10-17_162649-2.mpr"
    _df, _metadata = convert_mpr(mpr_with_sidecar_mpl, **params)

    # An mpl can also be passed manually as a Path, string, bytes
    mpl_path = folder / "file_2025-10-17_162649-2.mpl"
    mpl_bytes = mpl_path.open("rb").read()
    convert_mpr(mpr_without_date, mpl_file=mpl_path, **params)
    convert_mpr(mpr_without_date, mpl_file=str(mpl_path), **params)
    convert_mpr(mpr_without_date, mpl_file=mpl_bytes, **params)


def test_convert_data_update_database(reset_all, test_dir: Path) -> None:
    """Database should be able to accept data from known and unknown sources."""
    # Make backup to restore from for each test
    folder = test_dir / "eclab_harvester"
    test_file_1 = folder / "test_C01.mpr"
    db_path = test_dir / "database" / "test_database.db"
    sample_id = "240701_svfe_gen6_01"

    convert_mpr(
        test_file_1,
        sample_id=sample_id,
        job_id=None,  # e.g. manual upload or harvesting
        update_database=True,
    )
    # Should have made an entry in the dataframes table
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT `Job ID` FROM dataframes WHERE `Sample ID` = ? AND `File stem` = ?",
            (sample_id, test_file_1.stem),
        )
        result = cursor.fetchone()
        assert result is not None
        job_id = result[0]
        cursor.execute(
            "SELECT `Job ID` FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        result = cursor.fetchone()
        assert result is not None
        cursor.close()

    # If same data is submitted from a 'known source', it overwrites
    convert_mpr(
        test_file_1,
        sample_id=sample_id,
        job_id="known_source_123",
        update_database=True,
    )
    # Should have made an entry in the dataframes table
    previous_job_id = job_id
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT `Job ID` FROM dataframes WHERE `Sample ID` = ? AND `File stem` = ?",
            (sample_id, test_file_1.stem),
        )
        result = cursor.fetchone()
        assert result is not None
        job_id = result[0]
        assert job_id == "known_source_123"

        cursor.execute(
            "SELECT `Job ID` FROM jobs WHERE `Job ID` = ?",
            (previous_job_id,),
        )
        result = cursor.fetchone()
        assert result is None
        cursor.execute(
            "SELECT `Job ID` FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        result = cursor.fetchone()
        assert result is not None
        cursor.close()

    # If manually uploaded again, it will keep the known source job ID
    convert_mpr(
        test_file_1,
        sample_id=sample_id,
        job_id=None,  # e.g. manual upload or harvesting
        update_database=True,
    )
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT `Job ID` FROM dataframes WHERE `Sample ID` = ? AND `File stem` = ?",
            (sample_id, test_file_1.stem),
        )
        result = cursor.fetchone()
        assert result is not None
        job_id = result[0]
        assert job_id == "known_source_123"

        cursor.execute(
            "SELECT `Job ID` FROM jobs WHERE `Job ID` = ?",
            (previous_job_id,),
        )
        result = cursor.fetchone()
        assert result is None
        cursor.execute(
            "SELECT `Job ID` FROM jobs WHERE `Job ID` = ?",
            (job_id,),
        )
        result = cursor.fetchone()
        assert result is not None
        cursor.close()
