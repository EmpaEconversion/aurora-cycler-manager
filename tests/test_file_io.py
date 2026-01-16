"""Tests for file_io module."""

import json
from pathlib import Path

import pytest

from aurora_cycler_manager.database_funcs import get_sample_data
from aurora_cycler_manager.visualiser import file_io


class TestDetermineUploaded:
    """Test file_io.determine_upload function."""

    def test_empty(self) -> None:
        """Empty input."""
        res = file_io.determine_uploaded(b"", "", [])
        assert res[2]  # Disabled
        assert res[3] == {"data": None, "file": None}

    def test_samples_json(self) -> None:
        """Samples file."""
        filename = "samples.json"
        data = [
            {"Sample ID": "hello", "some": "params"},
            {"Sample ID": "there", "some": "params"},
        ]
        data_bytes = json.dumps(data).encode("utf-8")
        selected_rows = []

        res = file_io.determine_uploaded(data_bytes, filename, selected_rows)
        assert res[0] == f"Got a samples json\nContains {len(data)} samples"
        assert not res[2]  # Not disabled
        assert res[3] == {"data": data, "file": "samples-json"}

    def test_samples_overwrite(self) -> None:
        """Samples file with overwriting Sample ID."""
        filename = "samples.json"
        data = [{"Sample ID": "240606_svfe_gen1_15"}]
        data_bytes = json.dumps(data).encode("utf-8")
        selected_rows = []

        res = file_io.determine_uploaded(data_bytes, filename, selected_rows)
        assert "overwrite" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "samples-json"}

    def test_battinfo_jsonld(self) -> None:
        """BattINFO JSON-LD."""
        filename = "battinfo.json"
        data = {"@context": "some stuff", "@type": "CoinCell", "rdfs:comment": ["BattINFO converter version"]}
        data_bytes = json.dumps(data).encode("utf-8")
        selected_rows = [{"Sample ID": "a sample"}]

        res = file_io.determine_uploaded(data_bytes, filename, selected_rows)
        assert "BattINFO" in res[0]
        assert "will be merged with" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "battinfo-jsonld"}

    def test_battinfo_no_samples(self) -> None:
        """BattINFO JSON-LD without samples."""
        filename = "battinfo.jsonld"
        data = {"@context": "some stuff", "@type": "CoinCell", "rdfs:comment": ["BattINFO converter version"]}
        data_bytes = json.dumps(data).encode("utf-8")
        selected_rows = []

        res = file_io.determine_uploaded(data_bytes, filename, selected_rows)
        assert "you must select samples" in res[0]
        assert res[2]
        assert res[3] == {"data": None, "file": None}

    def test_aux_jsonld(self) -> None:
        """Auxiliary JSON-LD."""
        filename = "some-other.jsonld"
        data = {"@context": "some stuff", "@type": "CoinCell", "other": "things"}
        data_bytes = json.dumps(data).encode("utf-8")
        selected_rows = [{"Sample ID": "a sample"}]

        res = file_io.determine_uploaded(data_bytes, filename, selected_rows)
        assert "auxiliary json-ld" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "aux-jsonld"}

    def test_aux_jsonld_no_samples(self) -> None:
        """Auxiliary JSON-LD without samples."""
        filename = "some-other.jsonld"
        data = {"@context": "some stuff", "@type": "CoinCell", "other": "things"}
        data_bytes = json.dumps(data).encode("utf-8")
        selected_rows = []

        res = file_io.determine_uploaded(data_bytes, filename, selected_rows)
        assert "you must select samples" in res[0]
        assert res[2]
        assert res[3] == {"data": None, "file": None}


class TestProcessUploaded:
    """Test file_io.process_upload function."""

    def test_empty(self) -> None:
        """Nothing uploaded."""
        with pytest.raises(ValueError):
            file_io.process_uploaded({"data": None, "file": None}, b"", [])

    def test_bad_samples_json(self, reset_all) -> None:
        """Samples json."""
        res = file_io.process_uploaded(
            {"file": "samples-json", "data": {"wrong": "format"}},
            b"something",
            [],
        )
        assert res == 0

    def test_good_samples_json(self, reset_all) -> None:
        """Samples json."""
        res = file_io.process_uploaded(
            {"file": "samples-json", "data": [{"Sample ID": "put me in the db"}]},
            b"something",
            [],
        )
        assert res == 1
        sample = get_sample_data("put me in the db")
        assert sample


class TestCreateRocrate:
    """Test file_io.create_rocrate function."""

    def test_empty(self, tmp_path: Path) -> None:
        """Test nothing selected."""
        zip_path = tmp_path / "file.zip"
        with pytest.raises(ValueError):
            file_io.create_rocrate([], set(), None, zip_path, None)
