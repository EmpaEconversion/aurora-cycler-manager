"""Testing functions in the eclab_harvester.py."""

from pathlib import Path

import pytest
from sqlalchemy import MetaData, Table, create_engine, select

import aurora_cycler_manager.database_funcs as dbf
from aurora_cycler_manager.analysis import analyse_sample
from aurora_cycler_manager.data_parse import get_cycling
from aurora_cycler_manager.eclab_harvester import convert_mpr, get_mpr_data, main
from aurora_cycler_manager.setup_logging import setup_logging


def test_main(reset_all, mock_ssh, test_dir: Path, caplog) -> None:
    """Test file harvesting."""
    # Set up mock response and files
    run_id = "250116_kigr_gen6"
    sample_id = "250116_kigr_gen6_01"
    filename = "250116_kigr_gen6_01_01_GCPL_CD8"
    local_folder = test_dir / "local_snapshots" / "eclab_snapshots"
    files = {
        f"C:\\aurora\\data\\{run_id}\\{sample_id}\\job1.mpl": local_folder / run_id / "1" / (filename + ".mpl"),
        f"C:\\aurora\\data\\{run_id}\\{sample_id}\\job1.mpr": local_folder / run_id / "1" / (filename + ".mpr"),
    }
    mock_ssh.add_command_response(
        command="Get-ChildItem -Path 'C:/aurora/data/'",
        stdout="\n".join([k for k in files if k.endswith(".mpr")]),
    )
    for remote_path, local_path in files.items():
        content = local_path.read_bytes()
        mock_ssh.add_sftp_file(remote_path, content)

    # Make fake jobs for both files
    dbf.add_or_update_job("job1", {"Sample ID": sample_id, "Job ID on server": "bio-job1", "Server label": "bio"})
    setup_logging()
    main()

    # Should have copied the files as is
    for local_file in [
        local_folder / run_id / sample_id / "job1.mpr",
        local_folder / run_id / sample_id / "job1.mpl",
    ]:
        assert local_file.exists()
        local_file.unlink()

    # Should not warn/fail
    assert caplog.text == ""

    # Analysed data exists
    df = get_cycling(sample_id)
    assert df is not None


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

    engine = create_engine(f"sqlite:///{db_path.as_posix()}")
    metadata = MetaData()
    jobs_table = Table("jobs", metadata, autoload_with=engine)
    dataframes_table = Table("dataframes", metadata, autoload_with=engine)

    convert_mpr(
        test_file_1,
        sample_id=sample_id,
        job_id=None,  # e.g. manual upload or harvesting
        update_database=True,
    )
    # Should have made an entry in the dataframes table
    with engine.connect() as conn:
        result = (
            conn.execute(
                select(dataframes_table.c["Job ID"])
                .where(dataframes_table.c["Sample ID"] == sample_id)
                .where(dataframes_table.c["File stem"] == test_file_1.stem)
            )
            .mappings()
            .first()
        )
        assert result is not None
        job_id = result["Job ID"]

        result = conn.execute(select(jobs_table.c["Job ID"]).where(jobs_table.c["Job ID"] == job_id)).fetchone()

        assert result is not None

    # If same data is submitted from a 'known source', it overwrites
    convert_mpr(
        test_file_1,
        sample_id=sample_id,
        job_id="known_source_123",
        update_database=True,
    )
    # Should have made an entry in the dataframes table
    previous_job_id = job_id
    with engine.connect() as conn:
        result = (
            conn.execute(
                select(dataframes_table.c["Job ID"])
                .where(dataframes_table.c["Sample ID"] == sample_id)
                .where(dataframes_table.c["File stem"] == test_file_1.stem)
            )
            .mappings()
            .first()
        )
        assert result is not None
        job_id = result["Job ID"]
        assert job_id == "known_source_123"

        result = conn.execute(
            select(jobs_table.c["Job ID"]).where(jobs_table.c["Job ID"] == previous_job_id)
        ).fetchone()
        assert result is None
        result = conn.execute(select(jobs_table.c["Job ID"]).where(jobs_table.c["Job ID"] == job_id)).fetchone()
        assert result is not None
    # If manually uploaded again, it will keep the known source job ID
    convert_mpr(
        test_file_1,
        sample_id=sample_id,
        job_id=None,  # e.g. manual upload or harvesting
        update_database=True,
    )
    with engine.connect() as conn:
        result = (
            conn.execute(
                select(dataframes_table.c["Job ID"])
                .where(dataframes_table.c["Sample ID"] == sample_id)
                .where(dataframes_table.c["File stem"] == test_file_1.stem)
            )
            .mappings()
            .first()
        )
        assert result is not None
        job_id = result["Job ID"]
        assert job_id == "known_source_123"

        result = conn.execute(
            select(jobs_table.c["Job ID"]).where(jobs_table.c["Job ID"] == previous_job_id)
        ).fetchone()
        assert result is None
        result = conn.execute(select(jobs_table.c["Job ID"]).where(jobs_table.c["Job ID"] == job_id)).fetchone()
        assert result is not None


def test_convert_eis(reset_all, test_dir: Path) -> None:
    """Check EIS works without any cycling data."""
    mpr = test_dir / "misc" / "PEIS.mpr"
    df, _metadata, _yadg_metadata = get_mpr_data(mpr)
    assert all(x in df.columns for x in ["uts", "V (V)", "I (A)", "technique", "f (Hz)", "Re(Z) (ohm)", "Im(Z) (ohm)"])
    # Save to database analyse etc.
    sample_id = "240701_svfe_gen6_01"
    convert_mpr(mpr, sample_id=sample_id)
    data = analyse_sample(sample_id)
    assert data.eis is not None


def test_convert_eis_with_other_data(reset_all, test_dir: Path) -> None:
    """Check EIS works with cycling data."""
    sample_id = "240701_svfe_gen6_01"
    cycling_data = test_dir / "eclab_harvester" / "test_C01.mpr"
    eis_data = test_dir / "misc" / "PEIS.mpr"
    convert_mpr(cycling_data, sample_id=sample_id)
    convert_mpr(eis_data, sample_id=sample_id)
    data = analyse_sample(sample_id)

    assert data.cycling is not None
    assert len(data.cycling) == 18001
    assert data.eis is not None
    assert len(data.eis) == 244
    assert data.cycles_summary is not None
    assert data.eis["Cycle"][0] == max(data.cycling["Cycle"])
