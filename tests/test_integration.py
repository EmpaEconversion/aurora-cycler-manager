"""Test chaining together many high level functions."""

import base64
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from aurora_cycler_manager.analysis import (
    analyse_all_batches,
    analyse_all_samples,
    shrink_all_samples,
)
from aurora_cycler_manager.data_parse import get_cycling, get_sample_folder
from aurora_cycler_manager.database_funcs import get_job_data, get_jobs_from_sample, save_or_overwrite_batch
from aurora_cycler_manager.eclab_harvester import convert_all_mprs
from aurora_cycler_manager.neware_harvester import convert_all_neware_data
from aurora_cycler_manager.visualiser import file_io


def set_progress(x: tuple[float, str, str]) -> None:
    """Simulate a progress bar."""
    i, _message, _color = x
    if i < 0 or i > 100:
        msg = "Progress cannot be outside range 0-100"
        raise ValueError(msg)


def test_analyse_download_eclab_sample(
    reset_all, test_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Generate test data, run analysis."""
    sample_id = "250116_kigr_gen6_01"
    run_id = "250116_kigr_gen6"
    sample_folder = get_sample_folder(sample_id)
    raw_data_folder = test_dir / "local_snapshots" / "eclab_snapshots" / run_id / "1"

    # Test downloading without data - should just have some metadata
    zip_file = tmp_path / "empty.zip"
    file_io.create_rocrate(
        [sample_id],
        {"hdf5", "bdf-parquet", "bdf-csv", "cycles-json", "metadata-jsonld"},
        zip_file,
        None,
        set_progress,
    )
    assert zip_file.exists()
    with ZipFile(zip_file, mode="r") as zf:
        files = zf.namelist()
        assert f"{sample_id}/full.{sample_id}.bdf.parquet" not in files
        assert f"{sample_id}/full.{sample_id}.bdf.csv" not in files
        assert f"{sample_id}/cycles.{sample_id}.parquet" not in files
        assert f"{sample_id}/cycles.{sample_id}.csv" not in files
        assert f"{sample_id}/metadata.{sample_id}.jsonld" in files
        assert "ro-crate-metadata.json" in files

    # Zip the raw data
    zip_path = tmp_path / "data.zip"
    with ZipFile(zip_path, mode="w") as zf:
        # Add an invalid file
        zf.writestr(f"{sample_id}/thing.txt", "hello there")
        # Add an invalid sample ID
        zf.writestr("not_a_sample_in_the_db/thing.txt", "hello there")
        # Add file without sample folder
        zf.writestr("thing.txt", "hello there")

    # 'Upload' the raw data
    res = file_io.determine_file(zip_path, [])
    assert "No valid files" in res[0]
    assert res[3]["file"] is None

    # Zip the raw data
    zip_path = tmp_path / "data.zip"
    with ZipFile(zip_path, mode="w") as zf:
        for file in raw_data_folder.iterdir():
            zf.write(file, arcname=Path(sample_id) / file.name)
        # Add an invalid file
        zf.writestr(f"{sample_id}/thing.txt", "hello there")
        # Add an invalid sample ID
        zf.writestr("not_a_sample_in_the_db/thing.txt", "hello there")
        # Add file without sample folder
        zf.writestr("thing.txt", "hello there")

    # 'Upload' the raw data
    res = file_io.determine_file(zip_path, [])
    assert "zip" in res[0]
    assert res[3]["file"] == "zip"

    # Should make some parquet and cycles json files
    file_io.process_file(res[3], zip_path, [])
    assert (sample_folder / f"full.{sample_id}.parquet").exists()
    assert (sample_folder / f"cycles.{sample_id}.parquet").exists()
    assert len(list(sample_folder.rglob("snapshot.*"))) == 2
    df1 = pl.read_parquet(sample_folder / f"full.{sample_id}.parquet")  # compared later

    # Uploading again should not add new files, should overwrite
    zip_path = tmp_path / "data.zip"
    with ZipFile(zip_path, mode="w") as zf:
        for file in raw_data_folder.iterdir():
            zf.write(file, arcname=Path(sample_id) / file.name)
    res = file_io.determine_file(zip_path, [])
    assert "zip" in res[0]
    assert res[3]["file"] == "zip"
    file_io.process_file(res[3], zip_path, [])
    # Should still only be 2 snapshot files
    assert len(list(sample_folder.rglob("snapshot.*"))) == 2

    # Should make some Job IDs in the database
    jobs = get_jobs_from_sample(sample_id)
    assert len(jobs) > 0

    # Upload a unicycler dict
    unicycler_dict = {
        "unicycler": {"version": "0.4.3"},
        "sample": {"name": "test_sample", "capacity_mAh": "123"},
        "record": {"current_mA": "0.1", "voltage_V": "0.1", "time_s": "10"},
        "safety": {
            "max_voltage_V": "5",
            "min_voltage_V": "-0.1",
            "max_current_mA": "10",
            "min_current_mA": "-10",
            "delay_s": "10",
        },
        "method": [
            {"step": "constant_current", "rate_C": "0.1", "until_time_s": "54000.0", "until_voltage_V": "4.9"},
            {
                "step": "constant_voltage",
                "voltage_V": "4.9",
                "until_time_s": "21600",
                "until_rate_C": "0.05",
            },
            {
                "step": "constant_current",
                "rate_C": "-0.1",
                "until_time_s": "54000.0",
                "until_voltage_V": "3.5",
            },
            {
                "step": "loop",
                "loop_to": 1,
                "cycle_count": 100,
            },
        ],
    }
    unicycler_file = tmp_path / "unicycler.json"
    with unicycler_file.open("w") as f:
        f.write(json.dumps(unicycler_dict))
    # 'Upload' the dict without job selected
    res = file_io.determine_file(unicycler_file, [])
    assert "you must select jobs" in res[0]
    assert res[3]["file"] is None
    assert res[3]["data"] is None

    # 'Upload' the dict
    res = file_io.determine_file(unicycler_file, [{"Sample ID": sample_id, "Job ID": jobs[0]}])
    assert "unicycler" in res[0]
    assert res[3]["file"] == "unicycler-json"
    assert res[3]["data"] == unicycler_dict

    # Should add a unicycler protocol to the database
    file_io.process_file(res[3], unicycler_file, [{"Sample ID": sample_id, "Job ID": jobs[0]}])
    job_data = get_job_data(jobs[0])
    assert job_data["Unicycler protocol"] == unicycler_dict

    # 'Upload' the dict again, should warn
    res = file_io.determine_file(unicycler_file, [job_data])
    assert "unicycler" in res[0]
    assert "this will overwrite data" in res[0]
    assert res[3]["file"] == "unicycler-json"
    assert res[3]["data"] == unicycler_dict

    # Upload a battinfo xlsx
    zenodoinfo_xlsx = test_dir / "misc" / "aurora_zenodo_info.xlsx"
    battinfo_xlsx = test_dir / "misc" / "BattINFO_example.xlsx"

    # Without samples selected warns
    res = file_io.determine_file(battinfo_xlsx, [])
    assert "must select samples" in res[0]
    assert res[3]["file"] is None

    # Wrong xslx complains
    res = file_io.determine_file(zenodoinfo_xlsx, [])
    assert "does not have the expected sheets" in res[0]
    assert res[3]["file"] is None

    # With selected works
    res = file_io.determine_file(battinfo_xlsx, [{"Sample ID": sample_id}])
    assert "BattINFO" in res[0]
    assert res[3]["file"] == "battinfo-xlsx"
    assert res[3]["data"] is None

    # Should add a unicycler protocol to the database
    file_io.process_file(res[3], battinfo_xlsx, [{"Sample ID": sample_id}])
    assert (sample_folder / f"battinfo.{sample_id}.jsonld").exists()

    # Upload an auxiliary jsonld
    aux_jsonld = {
        "@context": "some stuff",
        "@type": "CoinCell",
        "hasThing": "stuff",
    }
    aux_json_file = tmp_path / "aux.json"
    with aux_json_file.open("w") as f:
        f.write(json.dumps(aux_jsonld))
    # 'Upload' the dict
    res = file_io.determine_file(aux_json_file, [{"Sample ID": sample_id}])
    assert "auxiliary" in res[0]
    assert res[3]["file"] == "aux-jsonld"
    assert res[3]["data"] == aux_jsonld

    # Should add a unicycler protocol to the database
    file_io.process_file(res[3], aux_json_file, [{"Sample ID": sample_id}])
    assert (sample_folder / f"aux.{sample_id}.jsonld").exists()

    # Include publication info - this is how data is uploaded to Dash
    zenodo_info_str = f"xlsx,{base64.b64encode(zenodoinfo_xlsx.read_bytes()).decode('utf-8')}"

    # Test downloading the files
    zip_file = tmp_path / "file.zip"
    file_io.create_rocrate(
        [sample_id],
        {"bdf-parquet", "bdf-csv", "cycles-csv", "cycles-parquet", "metadata-jsonld"},
        zip_file,
        zenodo_info_str,
        set_progress,
    )
    assert zip_file.exists()
    with ZipFile(zip_file, mode="r") as zf:
        files = zf.namelist()
        assert f"{sample_id}/full.{sample_id}.bdf.parquet" in files
        assert f"{sample_id}/full.{sample_id}.bdf.csv" in files
        assert f"{sample_id}/cycles.{sample_id}.parquet" in files
        assert f"{sample_id}/cycles.{sample_id}.csv" in files
        assert f"{sample_id}/metadata.{sample_id}.jsonld" in files
        assert "ro-crate-metadata.json" in files

    # Test reuploading the files
    res = file_io.determine_file(zip_file, selected_rows=[])
    assert "zip" in res[0]
    assert res[3]["file"] == "zip"
    file_io.process_file(res[3], zip_file, [])
    assert (sample_folder / f"full.{sample_id}.parquet").exists()
    assert (sample_folder / f"cycles.{sample_id}.parquet").exists()
    assert len(list(sample_folder.rglob("snapshot.*"))) == 3  # It has put the full thing in as one snapshot
    df2 = pl.read_parquet(sample_folder / f"full.{sample_id}.parquet")
    # The dataframe should not have changed
    assert_frame_equal(
        df1.drop("technique", strict=False),
        df2.drop("technique", strict=False),
        check_column_order=False,
    )

    # Test downloading the files when every one breaks
    with (
        patch("zipfile.ZipFile.write", side_effect=ValueError("fail")),
        patch("zipfile.ZipFile.writestr", side_effect=ValueError("fail")),
    ):
        # Test downloading the files
        zip_file = tmp_path / "file.zip"
        with pytest.raises(ValueError, match="Zip has no content"):
            file_io.create_rocrate(
                [sample_id],
                {"hdf5", "bdf-parquet", "bdf-csv", "cycles-json", "metadata-jsonld"},
                zip_file,
                zenodo_info_str,
                set_progress,
            )


def test_analyse_all(reset_all, test_dir: Path) -> None:
    """Run all conversions and analysis."""
    convert_all_neware_data()
    convert_all_mprs()
    analyse_all_samples()
    analyse_all_samples(mode="if_not_exists")
    analyse_all_samples(mode="always")
    for shrunk_file in test_dir.rglob("shrunk.*.parquet"):
        shrunk_file.unlink()
    assert not any(test_dir.rglob("shrunk.*.parquet"))
    shrink_all_samples()
    assert any(test_dir.rglob("shrunk.*.parquet"))
    save_or_overwrite_batch(
        "test",
        "this tests if batch analysis works",
        [
            "250116_kigr_gen6_01",
            "240606_svfe_gen1_15",
            "250127_svfe_gen21_01",
        ],
    )
    analyse_all_batches()
    assert (test_dir / "batches" / "test" / "batch.test.xlsx").exists()
    assert (test_dir / "batches" / "test" / "batch.test.json").exists()
    df = get_cycling("250116_kigr_gen6_01")
    assert len(df) == 4183
    assert df["Cycle"].max() == 3
