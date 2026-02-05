"""Test server_manager.py module."""

from pathlib import Path

import pytest
from aurora_unicycler import Protocol

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.cycler_servers import CyclerServer
from aurora_cycler_manager.server_manager import ServerManager, _CyclingJob, _Sample

CONFIG = get_config()


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
