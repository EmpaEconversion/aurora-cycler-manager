"""Test server_manager.py module."""

import json
from pathlib import Path
from time import time

import polars as pl
import pytest
from aurora_unicycler import Protocol

import aurora_cycler_manager.database_funcs as dbf
from aurora_cycler_manager.cycler_servers import CyclerServer
from aurora_cycler_manager.data_parse import get_cycling
from aurora_cycler_manager.server_manager import ServerManager, _CyclingJob, _Sample


def test_connections(reset_all, mock_ssh) -> None:
    """Test mock SSH connections."""
    sm = ServerManager()
    assert all(s in sm.servers for s in ["nw", "bio"])
    assert all(isinstance(s, CyclerServer) for s in sm.servers.values())


def test_cycling_job_add_payload(reset_all, mock_ssh, tmp_path: Path) -> None:
    """Test adding payload to cycling job."""
    sample_id = "240701_svfe_gen6_01"
    sm = ServerManager()
    sm.load("10-1-1", sample_id)

    cj = _CyclingJob(_Sample.from_id(sample_id), "test-job", 1, "testing")

    # Biologic
    payload_path = tmp_path / "test.mps"
    with payload_path.open("w") as f:
        f.write("Biologic settings file blah blah blah")
    cj.add_payload(payload_path)
    assert cj.payload == payload_path
    cj.payload = None
    cj.add_payload(str(payload_path))
    assert cj.payload == payload_path
    cj.payload = None

    # Neware
    payload_path = tmp_path / "test.xml"
    with payload_path.open("w") as f:
        f.write("<?xml= some neware stuff")
    cj.add_payload(payload_path)
    assert cj.payload == payload_path
    cj.payload = None
    cj.add_payload(str(payload_path))
    assert cj.payload == payload_path
    cj.payload = None

    # Unicycler
    protocol = Protocol.from_dict(
        {
            "record": {"time_s": 1},
            "method": [{"step": "open_circuit_voltage", "until_time_s": 1}],
        }
    )
    protocol_with_sample = Protocol.from_dict(
        {  # Details should be added automatically
            "record": {"time_s": 1},
            "sample": {"name": "240701_svfe_gen6_01", "capacity_mAh": 1000.0},
            "method": [{"step": "open_circuit_voltage", "until_time_s": 1}],
        }
    )

    # Unicycler JSON
    payload_path = tmp_path / "test.json"
    with payload_path.open("w") as f:
        f.write(protocol.model_dump_json(exclude_none=True))
    cj.add_payload(payload_path)
    assert cj.unicycler_protocol == protocol_with_sample.model_dump_json(exclude_none=True)
    cj.unicycler_protocol = None

    # Unicycler dict
    cj.add_payload(protocol.to_dict())
    assert cj.unicycler_protocol == protocol_with_sample.model_dump_json(exclude_none=True)
    cj.unicycler_protocol = None

    # Unicycler str
    cj.add_payload(protocol.to_json())
    assert cj.unicycler_protocol == protocol_with_sample.model_dump_json(exclude_none=True)
    cj.unicycler_protocol = None

    # Random string allowed, might be e.g. xml or mps string - checked by indiviual cycler servers
    cj.add_payload("hello")
    assert cj.payload == "hello"
    cj.payload = None

    # Wrong path not allowed
    file_path = tmp_path / "test.docx"
    with pytest.raises(AssertionError, match="If payload is a path, it must be"):
        cj.add_payload(file_path)


def test_load_eject(reset_all, mock_ssh) -> None:
    """Test loading and ejecting samples."""
    sm = ServerManager()

    sample1 = "240701_svfe_gen6_01"
    sample2 = "240709_svfe_gen8_01"
    sample3 = "250116_kigr_gen6_01"

    pip1 = "10-1-1"
    pip2 = "10-1-2"
    pip3 = "MPG2-1-1"
    pip4 = "MPG2-1-2"

    sm.load(pip1, sample1)

    with pytest.raises(ValueError, match="already has a sample loaded"):
        sm.load(pip1, sample2)

    with pytest.raises(ValueError, match="already loaded on pipeline"):
        sm.load(pip2, sample1)

    with pytest.raises(ValueError, match=f"has sample {sample1} loaded, not {sample2}"):
        sm.eject(pip1, sample2)

    sm.eject(pip1, sample1)
    sm.load(pip1, sample2)

    with pytest.raises(ValueError, match=f"There is no sample to eject on pipeline {pip2}"):
        sm.eject(pip2, sample2)

    sm.load(pip3, sample3)
    sm.load(pip4, sample1)
    sm.eject(pip3)
    sm.eject(pip4)

    with pytest.raises(ValueError, match="Sample ID 'err' not found in the database"):
        sm.load(pip2, "err")

    with pytest.raises(ValueError, match="Pipeline 'err' not found in the database"):
        sm.load("err", sample1)


def test_start_stop(reset_all, mock_ssh, tmp_path: Path, test_dir: Path) -> None:
    """Test starting and stopping jobs."""
    sm = ServerManager()

    sample1 = "240701_svfe_gen6_01"

    pip1 = "10-1-1"
    pip2 = "10-1-2"

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

    sm.load(pip1, sample1)

    mock_ssh.add_command_response(command="neware start 10-1-1")  # empty = success
    mock_ssh.add_command_response(command="neware get-job-id", stdout='{"10-1-1": "10-1-1-1"}')
    sm.submit(
        sample1,
        tmp_path / "unicycler.json",
        capacity_Ah="mass",
    )
    with pytest.raises(ValueError, match="job is currently running"):
        sm.eject(pip1)

    job_id = dbf.get_job_from_pipeline(pip1)
    assert job_id == "nw-10-1-1-1"
    assert dbf.check_job_running(job_id)

    mock_ssh.add_command_response(command="neware stop 10-1-1")  # empty = success
    mock_ssh.add_command_response(
        command="neware status 10-1-1",
        stdout='{"10-1-1": {"barcode": "240701_svfe_gen6_01", "workstatus": "working"}}',
    )
    sm.cancel(job_id)
    mock_ssh.add_command_response(
        command="neware status 10-1-1",
        stdout='{"10-1-1": {"barcode": "240701_svfe_gen6_01", "workstatus": "stop"}}',
    )
    sm.eject(pip1)

    file_paths = {
        ".ndc": "bts_folder\\NdcFile\\20250128\\20250128_161756_270120_0_69_33.ndc",
        "_step.ndc": "bts_folder\\NdcFile\\20250128\\20250128_161756_270120_0_69_33_step.ndc",
        "_runInfo.ndc": "bts_folder\\NdcFile\\20250128\\20250128_161756_270120_0_69_33_runInfo.ndc",
        "_log.ndc": "bts_folder\\NdcFile\\20250128\\20250128_161756_270120_0_69_33_log.ndc",
        "_es.ndc": "bts_folder\\NdcFile\\20250128\\20250128_161756_270120_0_69_33_es.ndc",
    }
    file_map = {p: test_dir / "ssh" / p.replace("\\", "/") for p in file_paths.values()}
    for remote_path, local_path in file_map.items():
        content = local_path.read_bytes()
        mock_ssh.add_sftp_file(remote_path, content)

    mock_ssh.add_command_response(
        command='$file_endings = @(".ndc", "_step.ndc", "_runInfo.ndc", "_log.ndc", "_es.ndc")',
        stdout=json.dumps(file_paths),
    )

    sm.snapshot(job_id)
    new_ndax_path = test_dir / "local_snapshots" / "neware_snapshots" / "nw-10-1-1-1.ndax"
    assert new_ndax_path.exists()
    new_ndax_path.unlink()  # Remove after analysed

    # Should have analysed data
    df = get_cycling(sample1)

    assert isinstance(df, pl.DataFrame)
    assert "uts" in df
    assert len(df) > 1000

    sm.load(pip2, sample1)
    mock_ssh.add_command_response(command="neware start 10-1-2")  # empty = success
    mock_ssh.add_command_response(command="neware get-job-id", stdout='{"10-1-2": "10-1-2-2"}')
    sm.submit(
        sample1,
        tmp_path / "unicycler.json",
        capacity_Ah="areal",
    )
    job_id = dbf.get_job_from_pipeline(pip2)
    assert job_id == "nw-10-1-2-2"
    assert dbf.check_job_running(job_id)
    mock_ssh.add_command_response(command="neware stop 10-1-2")  # empty = success
    mock_ssh.add_command_response(
        command="neware status 10-1-2",
        stdout='{"10-1-2": {"barcode": "240701_svfe_gen6_01", "workstatus": "working"}}',
    )
    sm.cancel(job_id)
    mock_ssh.add_command_response(
        command="neware status 10-1-2",
        stdout='{"10-1-2": {"barcode": "240701_svfe_gen6_01", "workstatus": "stop"}}',
    )
    sm.eject(pip2)


def test_update_db(reset_all, mock_ssh) -> None:
    """Test querying cyclers and refreshing database."""
    sm = ServerManager()

    last_update = dbf.get_db_last_update()
    assert not last_update

    biologic_response: dict[str, dict] = {
        "MPG2-1-1": {"Status": "Stop", "Ox/Red": "Reduction", "OCV": "OCV", "EIS": "No EIS", "Connection": "Ok"},
        "MPG2-1-2": {"Status": "Run", "Ox/Red": "Reduction", "OCV": "OCV", "EIS": "No EIS", "Connection": "Ok"},
    }
    neware_response: dict[str, dict] = {
        "10-1-1": {
            "dev": "27-10-1-1-0",
            "cycle_id": 12,
            "step_id": 3,
            "step_type": "cv",
            "workstatus": "working",
            "barcode": "240701_svfe_gen6_01",
            "current": 0,
            "voltage": 0.0003,
            "capacity": 0,
            "energy": 0,
            "totaltime": 123.4,
            "relativetime": 0.1,
            "open_or_close": 0,
            "ip": "127.0.0.1",
            "devtype": 27,
            "channel": "true",
        },
        "10-1-2": {
            "dev": "27-10-1-2-0",
            "cycle_id": 0,
            "step_id": 0,
            "step_type": "cv",
            "workstatus": "stop",
            "barcode": "some_other_sample",
            "current": 0,
            "voltage": 0.0003,
            "capacity": 0,
            "energy": 0,
            "totaltime": 123.4,
            "relativetime": 0.1,
            "open_or_close": 0,
            "ip": "127.0.0.1",
            "devtype": 27,
            "channel": "true",
        },
    }
    mock_ssh.add_command_response(
        command="biologic status --ssh",
        stdout=json.dumps(biologic_response),
    )
    mock_ssh.add_command_response(
        command="neware status",
        stdout=json.dumps(neware_response),
    )
    uts_now = time()
    sm.update_db()

    last_update = dbf.get_db_last_update()
    assert isinstance(last_update, float)
    assert abs(uts_now - last_update) < 10

    res = dbf.get_database()
    assert isinstance(res["data"]["pipelines"], dict)
    assert isinstance(res["data"]["pipelines"]["add"], list)
    pips = {p["Pipeline"]: p for p in res["data"]["pipelines"]["add"]}
    assert pips["MPG2-1-1"]["Ready"]
    assert not pips["MPG2-1-2"]["Ready"]
    assert not pips["10-1-1"]["Ready"]
    assert pips["10-1-2"]["Ready"]


def test_partial_update_db(reset_all, mock_ssh) -> None:
    """Test querying cyclers and refreshing database."""
    sm = ServerManager()

    last_update = dbf.get_db_last_update()
    assert not last_update

    res = dbf.get_database()
    assert isinstance(res["data"]["pipelines"], dict)
    assert isinstance(res["data"]["pipelines"].get("add"), list)
    assert not res["data"]["pipelines"].get("remove")
    assert not res["data"]["pipelines"].get("upsert")
    assert len(res["data"]["pipelines"]["add"]) >= 8  # All pipelines

    sample1 = "240701_svfe_gen6_01"
    pip1 = "10-1-1"
    sm.load(pip1, sample1)

    res = dbf.get_database_updates()
    assert isinstance(res["data"]["pipelines"], dict)
    assert not res["data"]["pipelines"].get("add")
    assert not res["data"]["pipelines"].get("remove")
    assert res["data"]["pipelines"].get("upsert")
    assert isinstance(res["data"]["pipelines"]["upsert"], list)
    assert res["data"]["pipelines"]["upsert"][0]["Pipeline"] == pip1

    last_update = dbf.get_db_last_update()
    assert last_update
