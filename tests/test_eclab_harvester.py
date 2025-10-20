"""Testing functions in the eclab_harvester.py."""

from datetime import datetime
from pathlib import Path

import pytest

from aurora_cycler_manager.eclab_harvester import convert_mpr, hash_dataframe


def test_convert_data() -> None:
    """Should be able to convert mprs from different formats."""
    folder = Path(__file__).resolve().parent / "test_data" / "eclab_harvester"
    mpr_with_date = folder / "test_C01.mpr"

    params = {
        "output_hdf5_file": False,
        "sample_id": "test",
    }

    # convert_mpr should work with Path, str, bytes
    _df, _metadata = convert_mpr(mpr_with_date, **params)
    _df, _metadata = convert_mpr(str(mpr_with_date), **params)
    with mpr_with_date.open("rb") as f:
        _df, _metadata = convert_mpr(f.read(), **params)

    # Without a sample ID it will fail
    with pytest.raises(ValueError):
        _df, _metadata = convert_mpr(mpr_with_date, output_hdf5_file=False)

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


def test_hash_dataframe() -> None:
    """Test that hashes are generated correctly."""
    folder = Path(__file__).resolve().parent / "test_data" / "eclab_harvester"
    file1 = folder / "file_2025-10-17_162649.mpr"
    file2 = folder / "file_2025-10-17_163649.mpr"
    file3 = folder / "file_2025-10-17_164650.mpr"
    file4 = folder / "file_2025-10-17_165650.mpr"
    file5 = folder / "file_2025-10-17_165650 - with mpt convert.mpr"
    file6 = folder / "different_test.mpr"
    fake_mpl_file = b"Acquisition started on : 09/04/2024 16:39:26.844\r\n"

    params = {
        "mpl_file": fake_mpl_file,
        "output_hdf5_file": False,
        "sample_id": "doesn't matter",
        "modified_date": datetime.now(),
    }

    df1, _ = convert_mpr(file1, **params)
    df2, _ = convert_mpr(file2, **params)
    df3, _ = convert_mpr(file3, **params)
    df4, _ = convert_mpr(file4, **params)
    df5, _ = convert_mpr(file5, **params)
    df6, _ = convert_mpr(file6, **params)

    # First file does not have 10 datapoints and throws an error
    with pytest.raises(ValueError):
        hash_dataframe(df1)

    # Hashes should be the same regardless of extra datapoints
    assert hash_dataframe(df2) == hash_dataframe(df3)
    assert hash_dataframe(df2) == hash_dataframe(df4)
    # Hashes should be the same after mpt convert, which can modify the mpr file
    assert hash_dataframe(df2) == hash_dataframe(df5)
    # Hash should be different to a different data file
    assert hash_dataframe(df2) != hash_dataframe(df6)
