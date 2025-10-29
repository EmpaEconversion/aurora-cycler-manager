"""Unit tests for battinfo_utils.py."""

import json
from pathlib import Path
from typing import Any

from aurora_cycler_manager.battinfo_utils import (
    find_coin_cell,
    merge_battinfo_with_db_data,
    summarise_assembly,
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
                "schema": "https://schema.org",
                "emmo": "https://w3id.org/emmo#",
                "echem": "https://w3id.org/emmo/domain/electrochemistry#",
                "battery": "https://w3id.org/emmo/domain/battery#",
                "chemical": "https://w3id.org/emmo/domain/chemical-substance/context",
                "unit": "https://qudt.org/vocab/unit/",
                "rdfs": "https://www.w3.org/TR/rdf-schema/#ch_comment",
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


def test_merge_battinfo_with_db() -> None:
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
    assert "240709_svfe_gen8_01" in result["schema:productID"]
    assert "empa__ccid000605" in result["schema:productID"]
    assert result["schema:dateCreated"] == "2024-07-11"
    assert (
        result["hasNegativeElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]["@type"]
        == "MassLoading"
    )
    assert result["hasNegativeElectrode"]["hasMeasuredProperty"] == {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 15.0},
        "hasMeasurementUnit": "unit:MilliM",
    }
    assert (
        result["hasPositiveElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]["@type"]
        == "MassLoading"
    )
    assert result["hasPositiveElectrode"]["hasMeasuredProperty"] == {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 14.0},
        "hasMeasurementUnit": "unit:MilliM",
    }

    # Try with a more complicated file
    sample_file = Path(__file__).parent / "test_data" / "samples" / "test_battinfo.jsonld"
    with sample_file.open("r") as f:
        battinfo_jsonld = json.load(f)
    result = merge_battinfo_with_db_data(battinfo_jsonld, sample_data)
    result = result["@graph"][0]["hasTestObject"]  # Get to the CoinCell, then it should have the same info as before
    assert "240709_svfe_gen8_01" in result["schema:productID"]
    assert "empa__ccid000605" in result["schema:productID"]
    assert result["schema:dateCreated"] == "2024-07-11"
    assert (
        result["hasNegativeElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"][-1]["@type"]
        == "MassLoading"
    )
    assert result["hasNegativeElectrode"]["hasMeasuredProperty"][-1] == {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 15.0},
        "hasMeasurementUnit": "unit:MilliM",
    }
    assert (
        result["hasPositiveElectrode"]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"][-1]["@type"]
        == "MassLoading"
    )
    assert result["hasPositiveElectrode"]["hasMeasuredProperty"][-1] == {
        "@type": "Diameter",
        "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": 14.0},
        "hasMeasurementUnit": "unit:MilliM",
    }

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
    assert "240709_svfe_gen8_01" in result["schema:productID"]
    assert "empa__ccid000605" in result["schema:productID"]
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
