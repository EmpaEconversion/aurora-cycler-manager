"""Test server_manager.py module."""

import sqlite3
from pathlib import Path

import pytest
from aurora_unicycler import Protocol

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.server_manager import _CyclingJob, _Sample

CONFIG = get_config()


def test_cycling_job_add_payload(reset_all, tmp_path: Path) -> None:
    """Test adding payload to cycling job."""
    with sqlite3.connect(CONFIG["Database path"]) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO pipelines (`Pipeline`, `Server label`, `Sample ID`) VALUES ('test','nw','240701_svfe_gen6_01')"
        )

    cj = _CyclingJob(_Sample.from_id("240701_svfe_gen6_01"), "test-job", 1, "testing")

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
