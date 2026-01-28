"""Tests for file_io module."""

import json
from pathlib import Path

import pytest

from aurora_cycler_manager.database_funcs import get_sample_data
from aurora_cycler_manager.visualiser import file_io


class TestDetermineUploaded:
    """Test file_io.determine_file function."""

    def test_empty(self, tmp_path: Path) -> None:
        """Empty input."""
        filepath = tmp_path / "doesnt exist.json"
        res = file_io.determine_file(filepath, [])
        assert res[2]  # Disabled
        assert res[3] == {"data": None, "file": None}

    def test_samples_json(self, tmp_path: Path) -> None:
        """Samples file."""
        data = [
            {"Sample ID": "hello", "some": "params"},
            {"Sample ID": "there", "some": "params"},
        ]
        filepath = tmp_path / "samples.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))

        res = file_io.determine_file(filepath, [])
        assert res[0] == f"Got a samples json\nContains {len(data)} samples"
        assert not res[2]  # Not disabled
        assert res[3] == {"data": data, "file": "samples-json"}

    def test_samples_overwrite(self, tmp_path: Path) -> None:
        """Samples file with overwriting Sample ID."""
        data = [{"Sample ID": "240606_svfe_gen1_15"}]
        filepath = tmp_path / "samples.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, [])
        assert "overwrite" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "samples-json"}

    def test_battinfo_jsonld(self, tmp_path: Path) -> None:
        """BattINFO JSON-LD."""
        data = {"@context": "some stuff", "@type": "CoinCell", "rdfs:comment": ["BattINFO converter version"]}
        selected_rows = [{"Sample ID": "a sample"}]
        filepath = tmp_path / "battinfo.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, selected_rows)
        assert "BattINFO" in res[0]
        assert "will be merged with" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "battinfo-jsonld"}

    def test_battinfo_no_samples(self, tmp_path: Path) -> None:
        """BattINFO JSON-LD without samples."""
        data = {"@context": "some stuff", "@type": "CoinCell", "rdfs:comment": ["BattINFO converter version"]}
        filepath = tmp_path / "battinfo.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, [])
        assert "you must select samples" in res[0]
        assert res[2]
        assert res[3] == {"data": None, "file": None}

    def test_aux_jsonld(self, tmp_path: Path) -> None:
        """Auxiliary JSON-LD."""
        data = {"@context": "some stuff", "@type": "CoinCell", "other": "things"}
        selected_rows = [{"Sample ID": "a sample"}]
        filepath = tmp_path / "aux.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, selected_rows)
        assert "auxiliary json-ld" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "aux-jsonld"}

    def test_aux_jsonld_no_samples(self, tmp_path: Path) -> None:
        """Auxiliary JSON-LD without samples."""
        data = {"@context": "some stuff", "@type": "CoinCell", "other": "things"}
        filepath = tmp_path / "aux.jsonld"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, [])
        assert "you must select samples" in res[0]
        assert res[2]
        assert res[3] == {"data": None, "file": None}


class TestProcessUploaded:
    """Test file_io.process_upload function."""

    def test_empty(self, tmp_path: Path) -> None:
        """Nothing uploaded."""
        filepath = tmp_path / "nothing.txt"
        with filepath.open("w") as f:
            f.write("")
        with pytest.raises(ValueError):
            file_io.process_file({"data": None, "file": None}, filepath, [])

    def test_bad_samples_json(self, reset_all, tmp_path: Path) -> None:
        """Samples json."""
        filepath = tmp_path / "samples.json"
        data = {"file": "samples-json", "data": {"wrong": "format"}}
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.process_file(
            data,
            filepath,
            [],
        )
        assert res == 0

    def test_good_samples_json(self, reset_all, tmp_path: Path) -> None:
        """Samples json."""
        filepath = tmp_path / "samples.json"
        data = {"file": "samples-json", "data": [{"Sample ID": "put me in the db"}]}
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.process_file(
            data,
            filepath,
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
            file_io.create_rocrate([], set(), zip_path)
