"""Tests for Neware harvester."""

import logging
from pathlib import Path

import aurora_cycler_manager.database_funcs as dbf
from aurora_cycler_manager.data_parse import get_cycling
from aurora_cycler_manager.neware_harvester import convert_all_neware_data, main
from aurora_cycler_manager.setup_logging import setup_logging


def test_main(reset_all, mock_ssh, test_dir: Path, caplog) -> None:
    """Test file harvesting."""
    local_folder = test_dir / "local_snapshots" / "neware_snapshots"
    files = {
        "C:/aurora/data/120_6_1_36.ndax": local_folder / "nw4-120-6-1-36.ndax",
        "C:/aurora/data/120_9_5_33.ndax": local_folder / "nw4-120-9-5-33.ndax",
    }
    mock_ssh.add_command_response(
        command="Get-ChildItem -Path 'C:/aurora/data/'",
        stdout="\n".join(files.keys()),
    )
    mock_ssh.add_command_response(
        command="Get-ChildItem -Path 'C:/Neware data/'",
        stdout="",
    )
    for remote_path, local_path in files.items():
        content = local_path.read_bytes()
        mock_ssh.add_sftp_file(remote_path, content)

    # Make fake jobs for both files
    dbf.add_or_update_job(
        "job1", {"Sample ID": "commercial_cell_009", "Job ID on server": "120-6-1-36", "Server label": "nw"}
    )
    dbf.add_or_update_job(
        "job2", {"Sample ID": "250127_svfe_gen21_01", "Job ID on server": "120-9-5-33", "Server label": "nw"}
    )
    setup_logging(level=logging.INFO)
    main()

    # Should have made the files job1.ndax and job2.ndax
    assert (local_folder / "job1.ndax").exists()
    (local_folder / "job1.ndax").unlink()
    assert (local_folder / "job2.ndax").exists()
    (local_folder / "job2.ndax").unlink()

    # Should not warn/fail
    assert caplog.text == ""

    # Analysed data exists
    df = get_cycling("commercial_cell_009")
    assert df is not None
    df = get_cycling("250127_svfe_gen21_01")
    assert df is not None


def test_convert_all(reset_all, caplog) -> None:
    """Test convert all function."""
    setup_logging(level=logging.WARNING)
    # Make fake jobs for files
    dbf.add_or_update_job(
        "nw4-120-6-1-36", {"Sample ID": "commercial_cell_009", "Job ID on server": "120-6-1-36", "Server label": "nw4"}
    )
    dbf.add_or_update_job(
        "nw4-120-9-5-33", {"Sample ID": "250127_svfe_gen21_01", "Job ID on server": "120-9-5-33", "Server label": "nw"}
    )

    # Convert the data
    convert_all_neware_data()

    # Should not warn/fail
    assert caplog.text == ""

    # Analysed data exists
    df = get_cycling("commercial_cell_009")
    assert df is not None
    df = get_cycling("250127_svfe_gen21_01")
    assert df is not None
