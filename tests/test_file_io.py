# Copyright © 2025-2026, Empa.
"""Tests for file_io module."""

import json
from pathlib import Path

import pandas as pd
import pytest

from aurora_cycler_manager.database_funcs import get_sample_data
from aurora_cycler_manager.visualiser import file_io

# In the test database, with barcode empa__ccid000605 and assembled 2024-07-11
SAMPLE = "240709_svfe_gen8_01"
OTHER_SAMPLE = "240606_svfe_gen1_15"


def write_json(path: Path, data: dict) -> Path:
    """Write data to a json file."""
    path.write_text(json.dumps(data))
    return path


def battery_test(cell: dict, output: dict | None = None) -> dict:
    """Wrap cell values in a BatteryTest, like BattINFO converter 3.4 does."""
    test = {
        "@context": ["https://w3id.org/emmo/domain/battery/context"],
        "@type": "BatteryTest",
        "hasTestObject": {"@type": "CoinCell", "rdfs:comment": ["Schema name: CoinCellSchema"], **cell},
    }
    if output:
        test["hasOutput"] = {"@type": ["BatteryTestResult", "dcat:Dataset"], **output}
    return test


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
        data = {
            "@context": "some stuff",
            "@type": "CoinCell",
            "rdfs:comment": ["BattINFO Converter version: 3.3.1", "Schema name: CoinCellSchema"],
        }
        selected_rows = [{"Sample ID": SAMPLE}]
        filepath = tmp_path / "battinfo.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, selected_rows)
        assert "BattINFO" in res[0]
        assert "will be merged with" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "battinfo-jsonld"}

    def test_battinfo_single_node_graph(self, tmp_path: Path) -> None:
        """A single-node @graph is unwrapped and accepted."""
        data = {
            "@context": "some stuff",
            "@graph": [
                {
                    "@type": "BatteryTest",
                    "hasTestObject": {
                        "@type": "CoinCell",
                        "rdfs:comment": ["Schema name: CoinCellSchema"],
                    },
                }
            ],
        }
        filepath = tmp_path / "battinfo.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, [{"Sample ID": SAMPLE}])
        assert "will be merged with" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "battinfo-jsonld"}

    def test_battinfo_multi_node_graph_rejected(self, tmp_path: Path) -> None:
        """A multi-node @graph is rejected at upload."""
        data = {
            "@context": "some stuff",
            "@graph": [
                {"@type": "SomethingElse", "note": "an unrelated node"},
                {
                    "@type": "BatteryTest",
                    "hasTestObject": {
                        "@type": "CoinCell",
                        "rdfs:comment": ["Schema name: CoinCellSchema"],
                    },
                },
            ],
        }
        filepath = tmp_path / "battinfo.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, [{"Sample ID": "a sample"}])
        assert "unrecognized structure" in res[0]
        assert "Multi-node @graph" in res[0]
        assert res[1] == "red"
        assert res[2]
        assert res[3] == {"data": None, "file": None}

    def test_battinfo_bad_root_rejected(self, tmp_path: Path) -> None:
        """A CoinCell under an unexpected root is rejected at upload."""
        data = {
            "@context": "some stuff",
            "@type": "Dataset",
            "contains": {"@type": "CoinCell", "rdfs:comment": ["Schema name: CoinCellSchema"]},
        }
        filepath = tmp_path / "battinfo.json"
        with filepath.open("w") as f:
            f.write(json.dumps(data))
        res = file_io.determine_file(filepath, [{"Sample ID": "a sample"}])
        assert "unrecognized structure" in res[0]
        assert res[1] == "red"
        assert res[2]
        assert res[3] == {"data": None, "file": None}

    def test_battinfo_matching_sample(self, tmp_path: Path) -> None:
        """Cell values matching the sample are accepted."""
        data = battery_test(
            {"schema:productID": "empa__ccid000605", "schema:name": SAMPLE, "schema:dateCreated": "2024-07-11"}
        )
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), [{"Sample ID": SAMPLE}])
        assert "will be merged with" in res[0]
        assert not res[2]
        assert res[3] == {"data": data, "file": "battinfo-jsonld"}

    def test_battinfo_other_cell_rejected(self, tmp_path: Path) -> None:
        """Cell values from another cell are rejected, naming each one."""
        data = battery_test(
            {"schema:productID": "empa__ccid_123456", "schema:name": "x", "schema:dateCreated": "2024-06-16"}
        )
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), [{"Sample ID": SAMPLE}])
        assert res[1] == "red"
        assert res[2]
        assert f"Cell ID 'empa__ccid_123456' does not match: {SAMPLE} ('empa__ccid000605')" in res[0]
        assert "Cell name 'x'" in res[0]
        assert "Date of cell assembly '2024-06-16'" in res[0]
        assert res[3] == {"data": None, "file": None}

    def test_battinfo_cell_values_on_many_samples_rejected(self, tmp_path: Path) -> None:
        """A cell value can only match one of several samples."""
        data = battery_test({"schema:productID": "empa__ccid000605"})
        rows = [{"Sample ID": SAMPLE}, {"Sample ID": OTHER_SAMPLE}]
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), rows)
        assert res[2]
        assert f"does not match: {OTHER_SAMPLE} ('empa__ccid000267')" in res[0]
        assert SAMPLE + " (" not in res[0]

    def test_battinfo_figures_on_many_samples_rejected(self, tmp_path: Path) -> None:
        """Publication figures belong to one sample."""
        data = battery_test({}, {"schema:associatedMedia": {"@id": "https://doi.org/x", "rdfs:label": ["Fig. 1"]}})
        rows = [{"Sample ID": SAMPLE}, {"Sample ID": OTHER_SAMPLE}]
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), rows)
        assert res[2]
        assert "figures are for one sample, but it is applied to 2" in res[0]
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), rows[:1])
        assert not res[2]

    def test_battinfo_files_and_db_values_warn(self, tmp_path: Path) -> None:
        """Listed files and values the database replaces are warnings, not errors."""
        data = battery_test(
            {
                "hasPositiveElectrode": {
                    "@type": "PositiveElectrode",
                    "hasMeasuredProperty": {
                        "@type": "Mass",
                        "hasNumericalPart": {"@type": "RealData", "hasNumberValue": 1},
                        "hasMeasurementUnit": "MilliGram",
                    },
                }
            },
            {"dcat:distribution": {"@type": "dcat:Distribution", "dcterms:identifier": "other/file.csv"}},
        )
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), [{"Sample ID": SAMPLE}])
        assert res[1] == "orange"
        assert not res[2]
        assert "Ignoring listed files, aurora adds the files it exports: other/file.csv" in res[0]
        cathode_mass = get_sample_data(SAMPLE)["Cathode mass (mg)"]
        assert f"hasPositiveElectrode/hasMeasuredProperty/Mass: 1 MilliGram replaced by {cathode_mass:g}" in res[0]

    def test_battinfo_unknown_sample(self, tmp_path: Path) -> None:
        """Selected samples must be in the database."""
        data = battery_test({})
        res = file_io.determine_file(write_json(tmp_path / "b.json", data), [{"Sample ID": "not a sample"}])
        assert res[2]
        assert "not found in the database" in res[0]

    def test_not_battinfo_xlsx(self, tmp_path: Path) -> None:
        """An xlsx the converter can't read is rejected with its reason."""
        filepath = tmp_path / "other.xlsx"
        with pd.ExcelWriter(filepath) as writer:
            pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Metadata")
        res = file_io.determine_file(filepath, [{"Sample ID": SAMPLE}])
        assert res[2]
        assert "Excel file is not a BattINFO file" in res[0]

    def test_battinfo_no_samples(self, tmp_path: Path) -> None:
        """BattINFO JSON-LD without samples."""
        data = {
            "@context": "some stuff",
            "@type": "CoinCell",
            "rdfs:comment": ["BattINFO Converter version: 3.3.1", "Schema name: CoinCellSchema"],
        }
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
