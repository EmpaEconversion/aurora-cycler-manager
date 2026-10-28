# Copyright © 2025-2026, Empa.
"""Unit tests for battinfo_utils.py."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aurora_cycler_manager.battinfo_utils import (
    add_data,
    build_export_metadata,
    check_battinfo_for_samples,
    find_coin_cell,
    generate_zenodo_info_xlsx_template,
    has_type,
    insert_dict_in_jsonld,
    make_test_object,
    merge_battinfo_with_db_data,
    pop_sheet_files,
    priority_merge,
    quantity_conflict,
    summarise_assembly,
    unwrap_graph,
)
from aurora_cycler_manager.database_funcs import get_sample_data


def test_find_coin_cell() -> None:
    """Should find @type CoinCell within any JSON-LD structure."""
    jsonld: dict[str, Any] = {
        "@type": "CoinCell",
        "hasPositiveElectrode": {
            "@type": "Electrode",
            "hasCoating": {
                "blah": "blah blah",
            },
        },
    }
    assert find_coin_cell(jsonld) == jsonld

    jsonld = {
        "@context": [
            "https://w3id.org/emmo/domain/battery/context",
            {
                "schema": "https://schema.org/",
                "emmo": "https://w3id.org/emmo#",
                "echem": "https://w3id.org/emmo/domain/electrochemistry#",
                "battery": "https://w3id.org/emmo/domain/battery#",
                "chemical": "https://w3id.org/emmo/domain/chemical-substance#",
                "unit": "https://qudt.org/vocab/unit/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            },
        ],
        "@graph": [
            {
                "@type": "some other thing",
                "with": "some properties",
                "maybe": [
                    {"some": "lists"},
                    {"of": "dicts"},
                ],
            },
            {
                "@type": "BatteryTest",
                "hasTestObject": {
                    "@type": "CoinCell",
                    "schema:version": "1.1.7",
                    "schema:productID": ["240701_svfe_gen6_10", "empa__ccid000010"],
                    "schema:dateCreated": "01/07/24",
                    "schema:creator": {
                        "@type": "schema:Person",
                        "@id": "https://orcid.org/0009-0004-4673-7806",
                        "schema:name": "Enea Svaluto-Ferro",
                    },
                },
            },
        ],
    }
    assert find_coin_cell(jsonld) == jsonld["@graph"][1]["hasTestObject"]

    jsonld = {
        "this": "time",
        "its": {
            "in": "a",
            "very": "nested",
            "dict": {
                "with": "maybe",
                "some": [
                    "lists",
                    "like",
                    "this",
                    {
                        "with": "dicts",
                        "inside": "it",
                        "@type": "CoinCellery",
                        "hah": "tricked you",
                        "okay": {
                            "but": "this",
                            "one": "is",
                            "the real thing": {
                                "@type": "CoinCell",
                                "schema:version": "1.1.7",
                                "Yahaha!": "You found me 🌱",
                            },
                        },
                    },
                ],
            },
        },
    }
    assert find_coin_cell(jsonld) == jsonld["its"]["dict"]["some"][-1]["okay"]["the real thing"]


def test_has_type() -> None:
    """@type may be a string or a list of strings."""
    assert has_type({"@type": "CoinCell"}, "CoinCell")
    assert has_type({"@type": ["CoinCell", "Cell"]}, "CoinCell")
    assert not has_type({"@type": "Electrode"}, "CoinCell")
    assert not has_type({"@type": ["Electrode"]}, "CoinCell")
    assert not has_type({}, "CoinCell")


def test_find_coin_cell_type_list() -> None:
    """Should find the CoinCell when @type is a list."""
    jsonld = {"@type": "BatteryTest", "hasTestObject": {"@type": ["CoinCell", "Cell"]}}
    assert find_coin_cell(jsonld) == jsonld["hasTestObject"]


def test_unwrap_graph() -> None:
    """Single-node @graph should be replaced by the node, carrying @context down."""
    ctx = ["https://w3id.org/emmo/domain/battery/context"]
    assert unwrap_graph({"@context": ctx, "@graph": [{"@type": "BatteryTest"}]}) == {
        "@context": ctx,
        "@type": "BatteryTest",
    }

    # Nothing to unwrap
    plain = {"@context": ctx, "@type": "CoinCell"}
    assert unwrap_graph(plain) == plain

    # A node keeping its own @context has the outer one merged in
    merged = unwrap_graph({"@context": ["outer"], "@graph": [{"@context": ["inner"], "@type": "BatteryTest"}]})
    assert merged["@context"] == ["outer", "inner"]

    with pytest.raises(ValueError, match="not supported"):
        unwrap_graph({"@graph": [{"@type": "BatteryTest"}, {"@type": "CoinCell"}]})


def test_make_test_object() -> None:
    """Every accepted root shape should end up as a BatteryTest with the CoinCell inside."""
    ctx = ["https://w3id.org/emmo/domain/battery/context"]
    coin_cell = {"@type": "CoinCell", "schema:name": "a sample"}

    shapes = [
        {"@context": ctx, **coin_cell},
        {"@context": ctx, "@type": "BatteryTest", "hasTestObject": dict(coin_cell)},
        {"@context": ctx, "@graph": [{"@type": "BatteryTest", "hasTestObject": dict(coin_cell)}]},
        {"@context": ctx, "@graph": [dict(coin_cell)]},
        {"@context": ctx, **coin_cell, "@type": ["CoinCell", "Cell"]},
    ]
    for shape in shapes:
        result = make_test_object(deepcopy(shape))
        assert result["@type"] == "BatteryTest"
        assert result["@context"] == ctx
        assert find_coin_cell(result)["schema:name"] == "a sample"

    with pytest.raises(ValueError, match="must have CoinCell or BatteryTest"):
        make_test_object({"@context": ctx, "@type": "Electrode"})


def test_summarise_assembly() -> None:
    """Check summarise summarises as expected."""
    data = get_sample_data("240709_svfe_gen8_01")
    assembly = data["Assembly history"]
    answer = summarise_assembly(assembly, data)
    expect = (
        "Cell assembly sequence: CellCan, NegativeElectrode, Separator, "
        "100.0 uL Electrolyte, PositiveElectrode, 1.0 mm Spacer, Spring, CellLid"
    )
    assert answer == expect

    data["Electrolyte amount before separator (uL)"] = 15.0
    data["Electrolyte amount after separator (uL)"] = 15.0
    data["Bottom spacer thickness (mm)"] = 1.0
    data["Top spacer thickness (mm)"] = 0.5
    assembly = [
        {"Step": "Bottom"},
        {"Step": "Spacer", "Description": "its a bottom spacer"},
        {"Step": "Cathode"},
        {"Step": "Electrolyte", "Description": "contains before with other stuff"},
        {"Step": "Separator"},
        {"Step": "Electrolyte", "Description": "blah blah after blah blah"},
        {"Step": "Anode"},
        {"Step": "Spacer", "Description": "contains top with some other words"},
        {"Step": "Spring"},
        {"Step": "Top"},
        {"Step": "Press"},
        {"Step": "Other stuff that should be ignored"},
    ]
    answer = summarise_assembly(assembly, data)
    expect = (
        "Cell assembly sequence: CellCan, 1.0 mm Spacer, PositiveElectrode, "
        "15.0 uL Electrolyte, Separator, 15.0 uL Electrolyte, "
        "NegativeElectrode, 0.5 mm Spacer, Spring, CellLid"
    )
    assert answer == expect


def test_merge_battinfo_with_db(test_dir: Path) -> None:
    """Test merging BattINFO JSON-LD with sample data from database."""
    battinfo_jsonld = {
        "@context": ["stuff"],
        "@type": "CoinCell",
        "schema:version": "1.2.0",
        "schema:productID": "this gets deleted",
        "schema:dateCreated": "27/03/2024 this also gets deleted",
        "schema:creator": {"@type": "schema:Person", "schema:name": "Mr Blobby"},
        "schema:manufacturer": {
            "@type": "schema:Organization",
            "@id": "https://www.wikidata.org/wiki/Q683116",
            "schema:name": "Empa",
        },
        "rdfs:comment": [
            "BattINFO Converter version: 1.2.0",
            "Software credit: blah blah blah",
            "BattINFO CoinCellSchema version: 1.2.0",
            "Project: some pytest stuff",
            "Assembled manually or by robot: coneptually",
        ],
    }
    sample_data = get_sample_data("240709_svfe_gen8_01")
    result = merge_battinfo_with_db_data(battinfo_jsonld, sample_data)

    # Should now contains everything it had before plus extras
    assert result["schema:name"] == "240709_svfe_gen8_01"
    assert result["schema:productID"] == "empa__ccid000605"
    assert result["schema:dateCreated"] == "2024-07-11"
    anode_am_props = result["hasNegativeElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]
    assert isinstance(anode_am_props, list)
    anode_am_prop_types = [prop["@type"] for prop in anode_am_props]
    assert "MassLoading" in anode_am_prop_types
    assert "MassFraction" in anode_am_prop_types
    assert {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 15.0},
        "hasMeasurementUnit": "unit:MilliM",
    } in result["hasNegativeElectrode"]["hasMeasuredProperty"]
    assert {
        "@type": "MassLoading",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 6.70010395749819},
        "hasMeasurementUnit": "unit:MilliGM-PER-CentiM2",
    } in result["hasPositiveElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]
    assert {
        "@type": "MassFraction",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 0.955},
        "hasMeasurementUnit": "unit:FRACTION",
    } in result["hasPositiveElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]
    assert {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 14.0},
        "hasMeasurementUnit": "unit:MilliM",
    } in result["hasPositiveElectrode"]["hasMeasuredProperty"]

    # Try with a more complicated file
    sample_file = test_dir / "samples" / "test_battinfo.jsonld"
    with sample_file.open("r") as f:
        battinfo_jsonld = json.load(f)
    result = merge_battinfo_with_db_data(battinfo_jsonld, sample_data)
    result = result["@graph"][0]["hasTestObject"]  # Get to the CoinCell, then it should have the same info as before
    assert result["schema:name"] == "240709_svfe_gen8_01"
    assert result["schema:productID"] == "empa__ccid000605"
    assert result["schema:dateCreated"] == "2024-07-11"
    assert {
        "@type": "MassFraction",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 0.95},
        "hasMeasurementUnit": "unit:FRACTION",
    } in result["hasNegativeElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]
    assert {
        "@type": "MassLoading",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 4.84369026158375},
        "hasMeasurementUnit": "unit:MilliGM-PER-CentiM2",
    } in result["hasNegativeElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]
    assert {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 15.0},
        "hasMeasurementUnit": "unit:MilliM",
    } in result["hasNegativeElectrode"]["hasMeasuredProperty"]
    assert {
        "@type": "Mass",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 28.01},
        "hasMeasurementUnit": "unit:MilliGM",
    } in result["hasNegativeElectrode"]["hasMeasuredProperty"]
    assert {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 14.0},
        "hasMeasurementUnit": "unit:MilliM",
    } in result["hasPositiveElectrode"]["hasMeasuredProperty"]

    # If you have no sample information, it should not change anything
    result = merge_battinfo_with_db_data(
        battinfo_jsonld,
        {
            "Sample ID": "240701_svfe_gen6_10",
            "Barcode": "empa__ccid000010",
            "other": "stuff",
            "is": "relevant",
        },
    )
    assert result == battinfo_jsonld

    battinfo_jsonld = {
        "@context": ["stuff"],
        "@type": "CoinCell",
        "schema:version": "1.2.0",
        "schema:productID": "this gets deleted",
        "schema:dateCreated": "27/03/2024 this also gets deleted",
        "schema:creator": {"@type": "schema:Person", "schema:name": "Mr Blobby"},
        "schema:manufacturer": {
            "@type": "schema:Organization",
            "@id": "https://www.wikidata.org/wiki/Q683116",
            "schema:name": "Empa",
        },
        "rdfs:comment": [
            "BattINFO Converter version: 1.2.0",
            "Software credit: blah blah blah",
            "BattINFO CoinCellSchema version: 1.2.0",
            "Project: some pytest stuff",
            "Assembled manually or by robot: coneptually",
        ],
        "hasNegativeElectrode": {
            "@type": "Electrode",
            "hasCoating": {
                "@type": "Coating",
                "hasActiveMaterial": {
                    "hasMeasuredProperty": {
                        "some": "dict",
                    },
                },
            },
            "hasMeasuredProperty": {"some": "dict"},
        },
        "hasPositiveElectrode": {
            "@type": "Electrode",
            "hasMeasuredProperty": [
                {"a": "list"},
                {"of": "dicts"},
            ],
            "hasCoating": {
                "@type": "Coating",
                "hasActiveMaterial": {
                    "hasMeasuredProperty": [
                        {"another": "list"},
                        {"of": "dicts"},
                    ],
                },
            },
        },
    }
    sample_data = get_sample_data("240709_svfe_gen8_01")
    result = merge_battinfo_with_db_data(battinfo_jsonld, sample_data)

    # Should be able to handle dicts and lists
    assert result["schema:name"] == "240709_svfe_gen8_01"
    assert result["schema:productID"] == "empa__ccid000605"
    assert result["schema:dateCreated"] == "2024-07-11"
    assert result["hasNegativeElectrode"]["hasMeasuredProperty"][-1] == {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 15.0},
        "hasMeasurementUnit": "unit:MilliM",
    }
    assert result["hasPositiveElectrode"]["hasMeasuredProperty"][-1] == {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 14.0},
        "hasMeasurementUnit": "unit:MilliM",
    }


def test_zenodo_xlsx(test_dir: Path) -> None:
    """Check Zenodo xlsx is readable."""
    content = generate_zenodo_info_xlsx_template(["sample1", "sample2"], ["ccid1", "ccid2"])
    pd.read_excel(content, sheet_name="General")
    pd.read_excel(content, sheet_name="Figures")
    pd.read_excel(content, sheet_name="Authors")
    pd.read_excel(content, sheet_name="Institutions")


def quantity(prop: str, value: float, unit: str) -> dict:
    """Build a measured property node."""
    return {
        "@type": prop,
        "hasNumericalPart": {"@type": "RealData", "hasNumberValue": value},
        "hasMeasurementUnit": unit,
    }


def test_quantity_conflict() -> None:
    """Quantities are compared across the units of aurora and the BattINFO converter."""
    assert quantity_conflict(quantity("Diameter", 14, "MilliMetre"), quantity("Diameter", 14.0, "unit:MilliM")) is None
    assert (
        quantity_conflict(quantity("Thickness", 25, "MicroMetre"), quantity("Thickness", 0.025, "unit:MilliM")) is None
    )
    assert (
        quantity_conflict(
            quantity("MassFraction", 95.5, "unit:PERCENT"), quantity("MassFraction", 0.955, "unit:FRACTION")
        )
        is None
    )
    assert (
        quantity_conflict(
            quantity("MassFraction", 96, "unit:PERCENT"), quantity("MassFraction", 0.955, "unit:FRACTION")
        )
        == "96 unit:PERCENT replaced by 0.955 unit:FRACTION"
    )
    # Unknown units only agree if they are the same
    assert quantity_conflict(quantity("Mass", 1, "Kilo"), quantity("Mass", 1, "Kilo")) is None
    assert quantity_conflict(quantity("Mass", 1, "Kilo"), quantity("Mass", 1, "unit:MilliGM")) is not None
    assert quantity_conflict({"@type": "Mass"}, quantity("Mass", 1, "unit:MilliGM")) is None


def test_insert_into_converter_types() -> None:
    """Inserts find nodes with an @type list, and untyped values join the existing node."""
    cell = {
        "hasSeparator": {
            "@type": ["Separator", "Polypropylene"],
            "hasMeasuredProperty": [quantity("Diameter", 16, "MilliMetre")],
        },
        "hasElectrolyte": {"@type": "Electrolyte"},
    }
    conflicts: list[str] = []
    insert_dict_in_jsonld(cell, [("hasSeparator", "Separator")], {"schema:name": "Celgard"}, conflicts=conflicts)
    insert_dict_in_jsonld(
        cell,
        [("hasSeparator", "Separator"), ("hasMeasuredProperty", "Diameter")],
        quantity("Diameter", 17, "unit:MilliM"),
        conflicts=conflicts,
    )
    insert_dict_in_jsonld(cell, [("hasElectrolyte", "OrganicElectrolyte")], {"schema:name": "LP57"})
    assert cell["hasSeparator"]["schema:name"] == "Celgard"
    assert cell["hasSeparator"]["hasMeasuredProperty"] == [quantity("Diameter", 17, "unit:MilliM")]
    assert cell["hasElectrolyte"] == {"@type": "Electrolyte", "schema:name": "LP57"}
    assert conflicts == ["hasSeparator/hasMeasuredProperty/Diameter: 16 MilliMetre replaced by 17 unit:MilliM"]


def test_merge_keeps_capacities_apart() -> None:
    """Areal and specific capacities are separate nodes, the N:P one uses the balancing capacity."""
    sample = {
        "Sample ID": "s",
        "Cathode C-rate definition areal capacity (mAh/cm2)": 2.0,
        "Cathode C-rate definition specific capacity (mAh/g)": 150,
        "Cathode balancing specific capacity (mAh/g)": 160,
    }
    conflicts: list[str] = []
    result = merge_battinfo_with_db_data({}, sample, allow_empty_battinfo=True, conflicts=conflicts)
    props = result["hasPositiveElectrode"]["hasCoating"]["hasMeasuredProperty"]
    assert [(p["hasNumericalPart"]["hasNumberValue"], p["hasMeasurementUnit"]) for p in props] == [
        (2.0, "emmo:MilliAmpereHourPerSquareCentiMetre"),
        (150, "unit:MilliA-HR-PER-GM"),
        (160, "unit:MilliA-HR-PER-GM"),
    ]
    assert conflicts == []


def test_merge_spacers() -> None:
    """A spacer without an electrode is replaced, not duplicated, and contacts the right electrode."""
    cell = {
        "@type": "CoinCell",
        "hasComponent": [
            {"@type": "Spring"},
            {"@type": "Spacer", "hasMeasuredProperty": quantity("Thickness", 1, "MilliMetre")},
        ],
    }
    sample = {
        "Sample ID": "s",
        "Assembly history": [{"Step": "Anode", "Description": ""}, {"Step": "Cathode", "Description": ""}],
        "Top spacer type": "SS",
        "Top spacer thickness (mm)": 0.5,
    }
    conflicts: list[str] = []
    result = merge_battinfo_with_db_data(cell, sample, conflicts=conflicts)
    spacers = [c for c in result["hasComponent"] if has_type(c, "Spacer")]
    assert len(spacers) == 1
    # The cathode goes on top, and it is the positive electrode
    assert spacers[0]["contacts"] == {"@type": "PositiveElectrode"}
    assert "hasComponent/Spacer/Thickness: 1 MilliMetre replaced by 0.5 unit:MilliM" in conflicts


def test_merge_replaces_assembly_sequence() -> None:
    """The database's assembly sequence replaces the sheet's, also when the comment is a string."""
    sample = {"Sample ID": "s", "Assembly history": [{"Step": "Bottom"}, {"Step": "Top"}]}
    conflicts: list[str] = []
    cell = {"@type": "CoinCell", "rdfs:comment": "Cell assembly sequence: CellLid"}
    result = merge_battinfo_with_db_data(cell, sample, conflicts=conflicts)
    assert result["rdfs:comment"] == "Cell assembly sequence: CellCan, CellLid"
    assert len(conflicts) == 1


def converter_output() -> dict:
    """Get a BatteryTest like BattINFO converter 3.4 writes, with details from its references sheet."""
    return {
        "@context": ["https://w3id.org/emmo/domain/battery/context", {"schema": "https://schema.org/"}],
        "@type": "BatteryTest",
        "hasTestObject": {"@type": "CoinCell", "rdfs:comment": ["Schema name: CoinCellSchema"]},
        "hasInput": {
            "@type": "dcat:Dataset",
            "dcat:distribution": {
                "@id": "https://zenodo.org/records/1/files/a.zip#x/x.mps",
                "@type": "dcat:Distribution",
            },
        },
        "hasOutput": {
            "@type": ["BatteryTestResult", "dcat:Dataset"],
            "dcterms:title": "Old dataset",
            "dcterms:license": {"@id": "https://creativecommons.org/licenses/by/4.0/"},
            "dcterms:issued": "2025-06-01",
            "dcterms:creator": [{"@type": "schema:Person", "schema:name": "Sheet Author"}],
            "dcat:accessURL": {"@id": "https://doi.org/10.5281/zenodo.1"},
            "dcat:distribution": [{"@type": "dcat:Distribution", "dcterms:identifier": "x/x.bdf.csv"}],
            "schema:associatedMedia": {"@id": "https://doi.org/old", "rdfs:label": ["Fig. 1a"]},
        },
    }


def test_pop_sheet_files() -> None:
    """Files are removed with their section, a section left with only its @type is removed."""
    jsonld = converter_output()
    files = pop_sheet_files(jsonld)
    assert set(files) == {"x/x.mps", "x/x.bdf.csv"}
    assert files["x/x.mps"].section == "hasInput"
    assert "hasInput" not in jsonld
    assert "dcat:distribution" not in jsonld["hasOutput"]


def test_priority_merge() -> None:
    """High wins and disagreements are reported, but a link equals its IRI and lists of nodes combine."""
    high = {"a": 1, "b": {"@id": "x"}, "dcat:distribution": {"@id": "f1", "c": 1}}
    low = {"a": 2, "b": "x", "dcat:distribution": [{"@id": "f1", "d": 2}, {"@id": "f2"}], "e": 3}
    conflicts: list[str] = []
    merged = priority_merge(high, low, names=("high", "low"), conflicts=conflicts)
    assert merged["a"] == 1
    assert merged["e"] == 3
    assert merged["dcat:distribution"] == [{"@id": "f1", "c": 1, "d": 2}, {"@id": "f2"}]
    assert len(conflicts) == 1
    assert conflicts[0].startswith("high and low disagree at a")


def test_add_data() -> None:
    """Files are identified by their path in the zip, BDF files get the BDF schema."""
    snippet = add_data("s/full.s.bdf.csv", "https://doi.org/10.5281/zenodo.42", zip_filename="my data.zip")
    dist = snippet["hasOutput"]["dcat:distribution"]
    assert dist["@id"] == "s/full.s.bdf.csv"
    assert dist["dcat:mediaType"] == {"@id": "https://www.iana.org/assignments/media-types/text/csv"}
    assert dist["csvw:tableSchema"]["@id"].endswith("battery-data-format/schema")
    assert dist["dcat:downloadURL"] == {"@id": "https://zenodo.org/records/42/files/my%20data.zip#s/full.s.bdf.csv"}
    dist = add_data("s/cycles.s.parquet", None)["hasOutput"]["dcat:distribution"]
    assert "csvw:tableSchema" not in dist
    assert "dcat:downloadURL" not in dist


def test_build_export_metadata() -> None:
    """The export's files and publication info win, the stored BattINFO fills gaps, then defaults."""
    files = [add_data("s/full.s.bdf.parquet", "https://doi.org/10.5281/zenodo.1")]
    pub_info = {
        "zenodo_doi_url": "https://doi.org/10.5281/zenodo.1",
        "citation_string": "Someone et al.",
        "authors": [{"name": "Export Author", "orcid": "https://orcid.org/1", "affiliation": ["Empa"]}],
        "institutions": {"Empa": {"wikidata_url": "http://www.wikidata.org/entity/Q683116"}},
        "publication_doi_url": "https://doi.org/new",
        "sample_to_fig": {"ccid": ["Fig. 3"]},
    }
    result, conflicts = build_export_metadata(converter_output(), files, pub_info, sample_id="s", ccid="ccid")
    output = result["hasOutput"]
    assert next(iter(result)) == "@context"
    assert output["dcat:distribution"]["@id"] == "s/full.s.bdf.parquet"
    assert "hasInput" not in result
    assert output["dcterms:creator"]["schema:name"] == "Export Author"
    assert output["schema:citation"] == "Someone et al."
    # Same record, so the stored dataset details fill what the export doesn't know
    assert output["dcterms:title"] == "Old dataset"
    assert output["dcterms:issued"] == "2025-06-01"
    # The sample is in both publications
    assert {m["@id"] for m in output["schema:associatedMedia"]} == {"https://doi.org/old", "https://doi.org/new"}
    assert output["prov:wasGeneratedBy"]["schema:instrument"]["@type"] == "schema:SoftwareApplication"
    assert conflicts[0].startswith("Ignoring files listed in BattINFO metadata")
    assert any("dcterms:creator" in c for c in conflicts)
    assert not any("license" in c or "accessURL" in c for c in conflicts)


def test_build_export_metadata_other_record() -> None:
    """Dataset details for another Zenodo record are ignored, its publications are kept."""
    pub_info = {"zenodo_doi_url": "https://doi.org/10.5281/zenodo.2"}
    result, conflicts = build_export_metadata(converter_output(), [], pub_info, sample_id="s", ccid="ccid")
    output = result["hasOutput"]
    assert output["dcat:accessURL"] == {"@id": "https://doi.org/10.5281/zenodo.2"}
    assert output["dcterms:title"] == "Cycling data coin cell ccid"
    assert output["dcterms:issued"] != "2025-06-01"
    assert output["schema:associatedMedia"]["@id"] == "https://doi.org/old"
    assert any(c.startswith("Ignoring Zenodo record https://doi.org/10.5281/zenodo.1") for c in conflicts)


def test_check_battinfo_for_samples() -> None:
    """Cell values must match every sample, files and replaced values are warnings."""
    jsonld = converter_output()
    jsonld["hasTestObject"]["schema:productID"] = "ccid1"
    sample = {"Sample ID": "s1", "Barcode": "ccid1"}
    errors, warnings = check_battinfo_for_samples(jsonld, [sample])
    assert errors == []
    assert warnings == ["Ignoring listed files, aurora adds the files it exports: x/x.bdf.csv, x/x.mps"]

    errors, _ = check_battinfo_for_samples(jsonld, [sample, {"Sample ID": "s2"}])
    assert errors == [
        "Cell ID 'ccid1' does not match: s2 (not in database)",
        "Associated publication figures are for one sample, but it is applied to 2",
    ]


def test_merge_keeps_sheet_comments_and_types() -> None:
    """Database values add to comments and @type lists, replaced names are reported."""
    cell = {
        "@type": "CoinCell",
        "hasCase": {"@type": ["R2032", "StainlessSteel"], "rdfs:comment": ["From the sheet"]},
        "hasElectrolyte": {"@type": "Electrolyte", "schema:name": "LP30"},
    }
    sample = {"Sample ID": "s", "Casing material": "SS316", "Electrolyte name": "LP57"}
    conflicts: list[str] = []
    result = merge_battinfo_with_db_data(cell, sample, conflicts=conflicts)
    assert result["hasCase"] == {
        "@type": ["R2032", "StainlessSteel"],
        "rdfs:comment": ["From the sheet", "Material: SS316"],
    }
    assert result["hasElectrolyte"] == {"@type": "Electrolyte", "schema:name": "LP57"}
    assert conflicts == ["hasElectrolyte/schema:name: 'LP30' replaced by 'LP57'"]
