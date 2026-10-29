# Copyright © 2025-2026, Empa.
"""BattINFO ontology functions."""

import base64
import io
import json
import logging
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple
from urllib.parse import quote, unquote

import pandas as pd

from aurora_cycler_manager.version import __title__, __url__, __version__

logger = logging.getLogger(__name__)

BDF_SCHEMA_IRI = "https://w3id.org/battery-data-alliance/ontology/battery-data-format/schema"
# IRI form used in the DCAT 3 examples
IANA_MEDIA_TYPES = "http://www.iana.org/assignments/media-types/"
MEDIA_TYPES = {
    ".parquet": "application/vnd.apache.parquet",
    ".csv": "text/csv",
    ".json": "application/json",
}
# Record number in a Zenodo DOI (zenodo.123) or record page URL (/records/123, older /record/123)
_ZENODO_RECORD_RE = re.compile(r"zenodo\.(\d+)|/records?/(\d+)")
FIGURE_COMMENT = "Subfigure of associated peer-reviewed scientific publication containing this data"

# Units written by aurora or the BattINFO converter templates, as (common unit, scale to it)
UNIT_EQUIVALENTS: dict[str, tuple[str, float]] = {
    "unit:MilliGM": ("mg", 1),
    "MilliGram": ("mg", 1),
    "unit:MilliM": ("mm", 1),
    "MilliMetre": ("mm", 1),
    "MicroMetre": ("mm", 1e-3),
    "unit:MilliGM-PER-CentiM2": ("mg/cm2", 1),
    "MilliGramPerSquareCentiMetre": ("mg/cm2", 1),
    "unit:FRACTION": ("fraction", 1),
    "unit:PERCENT": ("fraction", 0.01),
    "unit:MicroL": ("uL", 1),
    "unit:MilliA-HR-PER-GM": ("mAh/g", 1),
    "emmo:MilliAmpereHourPerSquareCentiMetre": ("mAh/cm2", 1),
    "MilliAmpereHourPerSquareCentiMetre": ("mAh/cm2", 1),
}


blank_coin_cell = coin_cell = {
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
    "@type": "CoinCell",
    "hasPositiveElectrode": {
        "@type": "Electrode",
        "hasCurrentCollector": {"@type": "CurrentCollector"},
        "hasCoating": {"@type": "ElectrodeCoating"},
    },
    "hasNegativeElectrode": {
        "@type": "Electrode",
        "hasCurrentCollector": {"@type": "CurrentCollector"},
        "hasCoating": {"@type": "ElectrodeCoating"},
    },
    "hasElectrolyte": {"@type": "OrganicElectrolyte"},
    "hasSeparator": {
        "@type": "Separator",
    },
    "hasCase": {
        "@type": "R2032",
        "hasComponent": [
            {"@type": "CellLid"},
            {"@type": "CellCan"},
        ],
    },
    "hasComponent": [
        {"@type": "Spring"},
    ],
}


def _deep_merge_dicts(target: dict, source: dict) -> dict:
    """Recursively merge source into target, adding to comments and lists instead of replacing them."""
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge_dicts(target[k], v)
        elif k in target and isinstance(target[k], list) and isinstance(v, list):
            target[k].extend(x for x in v if x not in target[k])
        elif k in target and (isinstance(target[k], list) or k == "rdfs:comment"):
            values = as_list(target[k])
            values += [x for x in as_list(v) if x not in values]
            target[k] = values if isinstance(target[k], list) or len(values) > 1 else values[0]
        else:
            target[k] = v
    return target


# Text values from the database that are reported when they replace a different one
REPORTED_TEXT_KEYS = ("schema:name", "schema:description")


def _text_conflicts(old: dict, new: dict) -> list[str]:
    """Describe the text values of new that replace different ones in old."""
    return [
        f"{k}: '{old[k]}' replaced by '{new[k]}'"
        for k in REPORTED_TEXT_KEYS
        if isinstance(old.get(k), str) and k in new and old[k] != new[k]
    ]


def _quantity(node: dict) -> tuple[float, str | None] | None:
    """Get the number and unit of a measured property node, None if it is not one."""
    part = node.get("hasNumericalPart")
    if not isinstance(part, dict):
        return None
    value = part.get("hasNumberValue")
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value), node.get("hasMeasurementUnit")
    except (TypeError, ValueError):
        return None


def _same_quantity(a: tuple[float, str | None], b: tuple[float, str | None]) -> bool:
    """Check two quantities are equal, converting between known units."""
    (value_a, unit_a), (value_b, unit_b) = a, b
    if unit_a in UNIT_EQUIVALENTS and unit_b in UNIT_EQUIVALENTS:
        (common_a, scale_a), (common_b, scale_b) = UNIT_EQUIVALENTS[unit_a], UNIT_EQUIVALENTS[unit_b]
        return common_a == common_b and math.isclose(value_a * scale_a, value_b * scale_b, rel_tol=1e-6)
    return unit_a == unit_b and math.isclose(value_a, value_b, rel_tol=1e-6)


def quantity_conflict(old: dict, new: dict) -> str | None:
    """Describe how two measured property nodes disagree, None if they agree or are not quantities."""
    old_q, new_q = _quantity(old), _quantity(new)
    if old_q is None or new_q is None or _same_quantity(old_q, new_q):
        return None
    return f"{old_q[0]:g} {old_q[1] or ''} replaced by {new_q[0]:g} {new_q[1] or ''}".replace("  ", " ")


def _compatible(old: dict, new: dict) -> bool:
    """Check two nodes of one type could be the same quantity, by their units and comments."""
    old_unit, new_unit = old.get("hasMeasurementUnit"), new.get("hasMeasurementUnit")
    if old_unit and new_unit:
        old_common = UNIT_EQUIVALENTS.get(old_unit, (old_unit, 1))[0]
        new_common = UNIT_EQUIVALENTS.get(new_unit, (new_unit, 1))[0]
        if old_common != new_common:
            return False
    old_comment, new_comment = old.get("rdfs:comment"), new.get("rdfs:comment")
    return not (old_comment and new_comment and old_comment != new_comment)


def as_list(value: object) -> list:
    """Wrap a single JSON-LD value in a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _touches(component: dict, electrode: str | None) -> bool:
    """Check if a component's contacts node is the given electrode."""
    if not electrode:
        return False
    return any(isinstance(c, dict) and has_type(c, electrode) for c in as_list(component.get("contacts")))


def _has_any_type(node: dict, types: str | tuple[str, ...]) -> bool:
    """Check if a node has one of the given types."""
    return any(has_type(node, t) for t in ((types,) if isinstance(types, str) else types))


def _first_type(types: str | tuple[str, ...]) -> str:
    """Get the type a new node is created with."""
    return types if isinstance(types, str) else types[0]


def _describes(node: object, new_dict: dict, expected_type: str | tuple[str, ...] | None) -> bool:
    """Check if new_dict describes node, by its @type, or by the expected type if it has none."""
    if not isinstance(node, dict):
        return False
    if new_type := new_dict.get("@type"):
        return has_type(node, new_type) and _compatible(node, new_dict)
    return expected_type is None or _has_any_type(node, expected_type)


def insert_dict_in_jsonld(
    obj: dict | list,
    keys: list[tuple],
    new_dict: dict,
    *,
    merge: bool = True,
    conflicts: list[str] | None = None,
    _path: str = "",
) -> None:
    """Insert a dict into a nested jsonld.

    Args:
        obj: The JSON-LD object to put the thing in.
        keys: list of tuples with the key and optional expected type, or types any of which match,
            the first is used for a new node, e.g.
            [
                ("hasPositiveElectrode", ("PositiveElectrode", "Electrode")),
                ("hasCoating", ("ElectrodeCoating", "Coating")),
                ("hasActiveMaterial", None),
                ("hasMeasuredProperty", "MassLoading"),
            ]
        new_dict: The dict to insert
        merge (optional): Whether to merge or replace, default True (merge)
        conflicts (optional): list to append a message to when a different value is replaced

    """
    if not keys:
        msg = "Keys list cannot be empty"
        raise ValueError(msg)

    key, expected_type = keys[0]
    path = f"{_path}/{key}".lstrip("/")

    def put(old: dict) -> dict:
        """Merge or replace old with new_dict, noting replaced values."""
        if conflicts is not None:
            label = new_dict.get("@type")
            prefix = f"{path}/{label}" if label else path
            if conflict := quantity_conflict(old, new_dict):
                conflicts.append(f"{prefix}: {conflict}")
            conflicts.extend(f"{prefix}/{c}" for c in _text_conflicts(old, new_dict))
        return _deep_merge_dicts(old, new_dict) if merge else new_dict

    # At the end of the chain - could by empty, a dict, or a list
    if len(keys) == 1:
        if key not in obj or obj[key] is None:
            # A new node gets the expected type, as when descending
            untyped = expected_type and "@type" not in new_dict
            obj[key] = {"@type": _first_type(expected_type), **new_dict} if untyped else new_dict
        elif isinstance(obj[key], list):
            for i, o in enumerate(obj[key]):
                if _describes(o, new_dict, expected_type):
                    obj[key][i] = put(o)
                    break
            else:
                obj[key].append(new_dict)
        elif isinstance(obj[key], dict):
            # A dict without @type adds to whatever node the key already holds
            if _describes(obj[key], new_dict, None):
                obj[key] = put(obj[key])
            else:
                obj[key] = [obj[key], new_dict]
        else:
            msg = f"Unexpected type at end of path: {type(obj[key])}"
            raise TypeError(msg)
        return

    # Otherwise, recursively descend
    if key not in obj or obj[key] is None:
        obj[key] = {"@type": _first_type(expected_type)} if expected_type else {}

    val = obj[key]

    # If dict, descend directly
    if isinstance(val, dict):
        insert_dict_in_jsonld(val, keys[1:], new_dict, merge=merge, conflicts=conflicts, _path=path)

    # If list, select the dict with the expected type, or the only dict if no type is expected
    elif isinstance(val, list):
        nodes = [el for el in val if isinstance(el, dict)]
        if expected_type:
            target = next((el for el in nodes if _has_any_type(el, expected_type)), None)
        else:
            target = nodes[0] if len(nodes) == 1 else None
        if target is None:
            # If no match found, create a new dict with that type
            target = {"@type": _first_type(expected_type)} if expected_type else {}
            val.append(target)
        insert_dict_in_jsonld(target, keys[1:], new_dict, merge=merge, conflicts=conflicts, _path=path)

    else:
        msg = f"Unexpected type in path: {type(val)}"
        raise TypeError(msg)


# Classes a node may have on aurora's paths, the first is used for a new node
COATING_TYPES = ("ElectrodeCoating", "Coating")
ELECTROLYTE_TYPES = ("OrganicElectrolyte", "Electrolyte")


def _mix_active_materials(cell: dict, electrode_key: str) -> list[str]:
    """Group an electrode's active materials under one ActiveMaterialMix, return a note for each blend.

    Aurora's active material values are totals, so a blend needs one node to describe it.
    """
    notes = []
    for electrode in as_list(cell.get(electrode_key)):
        for coating in as_list(electrode.get("hasCoating")) if isinstance(electrode, dict) else []:
            materials = as_list(coating.get("hasActiveMaterial")) if isinstance(coating, dict) else []
            if len(materials) > 1:
                coating["hasActiveMaterial"] = {"@type": "ActiveMaterialMix", "hasConstituent": materials}
                notes.append(
                    f"{electrode_key}/hasCoating/hasActiveMaterial: {len(materials)} active materials grouped "
                    "as constituents of an ActiveMaterialMix, for the total values"
                )
    return notes


def assembly_date(sample_data: dict) -> str | None:
    """Get the ISO 8601 date of the last timestamped assembly step."""
    for step in reversed(sample_data.get("Assembly history") or []):
        if step.get("Timestamp"):
            # It is YYYY-MM-DD hh-mm-ss %z
            return step["Timestamp"][:10]
    return None


def merge_battinfo_with_db_data(
    battinfo_jsonld: dict,
    sample_data: dict,
    *,
    allow_empty_battinfo: bool = False,
    conflicts: list[str] | None = None,
) -> dict:
    """Merge info from the database with BattINFO ontology.

    The database wins where both have a value. Values it replaces are appended to `conflicts`.
    """
    coin_cell = find_coin_cell(battinfo_jsonld)
    if coin_cell is None:
        if allow_empty_battinfo:
            # Make a default coin cell
            battinfo_jsonld = deepcopy(blank_coin_cell)
            coin_cell = battinfo_jsonld
        else:
            msg = "Could not find CoinCell in JSON-LD"
            raise ValueError(msg)

    def insert(keys: list[tuple], new_dict: dict) -> None:
        insert_dict_in_jsonld(coin_cell, keys, new_dict, conflicts=conflicts)

    # Sample ID and CCID (barcode)
    coin_cell["schema:name"] = sample_data["Sample ID"]
    if val := sample_data.get("Barcode"):
        coin_cell["schema:productID"] = val

    # Date created
    if date := assembly_date(sample_data):
        coin_cell["schema:dateCreated"] = date

    # Cell assembly sequence
    if assembly_history := sample_data.get("Assembly history"):
        sequence = summarise_assembly(assembly_history, sample_data)
        comments = coin_cell.get("rdfs:comment", [])
        comments = comments if isinstance(comments, list) else [comments]
        old = next((c for c in comments if isinstance(c, str) and c.startswith("Cell assembly sequence: ")), None)
        if old and old != sequence and conflicts is not None:
            conflicts.append(f"rdfs:comment: '{old}' replaced by '{sequence}'")
        comments = [c for c in comments if c != old]
        coin_cell["rdfs:comment"] = [*comments, sequence] if comments else sequence

    # Electrode specific quantities
    for xode in ("Anode", "Cathode"):
        key = "hasPositiveElectrode" if xode == "Cathode" else "hasNegativeElectrode"
        electrode_types = (
            ("PositiveElectrode", "Electrode") if xode == "Cathode" else ("NegativeElectrode", "Electrode")
        )

        has_am_values = sample_data.get(f"{xode} active material mass fraction") or (
            sample_data.get(f"{xode} active material mass (mg)") and sample_data.get(f"{xode} diameter (mm)")
        )
        if has_am_values:
            notes = _mix_active_materials(coin_cell, key)
            if conflicts is not None:
                conflicts.extend(notes)

        # Name, description
        if name := sample_data.get(f"{xode} type"):
            insert([(key, electrode_types)], {"schema:name": name})
        if desc := sample_data.get(f"{xode} description"):
            insert([(key, electrode_types)], {"schema:description": desc})

        # Electrode mass
        if electrode_mass_mg := sample_data.get(f"{xode} mass (mg)"):
            insert(
                [(key, electrode_types), ("hasMeasuredProperty", "Mass")],
                {
                    "@type": "Mass",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": electrode_mass_mg},
                    "hasMeasurementUnit": "unit:MilliGM",
                },
            )

        if electrode_cc_mass_mg := sample_data.get(f"{xode} current collector mass (mg)"):
            insert(
                [(key, electrode_types), ("hasCurrentCollector", "CurrentCollector"), ("hasMeasuredProperty", "Mass")],
                {
                    "@type": "Mass",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": electrode_cc_mass_mg},
                    "hasMeasurementUnit": "unit:MilliGM",
                },
            )

        # Electrode active material mass loading
        if sample_data.get(f"{xode} active material mass (mg)") and sample_data.get(f"{xode} diameter (mm)"):
            mass_g = sample_data[f"{xode} active material mass (mg)"] / 1000
            area_cm = 3.14159 * (sample_data[f"{xode} diameter (mm)"] / 20) ** 2
            mass_loading_g_per_cm2 = 1000 * mass_g / area_cm
            mass_dict = {
                "@type": "MassLoading",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": mass_loading_g_per_cm2},
                "hasMeasurementUnit": "unit:MilliGM-PER-CentiM2",
            }
            insert(
                [
                    (key, electrode_types),
                    ("hasCoating", COATING_TYPES),
                    ("hasActiveMaterial", None),
                    ("hasMeasuredProperty", "MassLoading"),
                ],
                mass_dict,
            )

        # Electrode diameter
        if sample_data.get(f"{xode} diameter (mm)"):
            diameter_dict = {
                "@type": "Diameter",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": sample_data[f"{xode} diameter (mm)"]},
                "hasMeasurementUnit": "unit:MilliM",
            }
            insert(
                [(key, electrode_types), ("hasMeasuredProperty", "Diameter")],
                diameter_dict,
            )

        # Active material mass fraction
        if am_mass_frac := sample_data.get(f"{xode} active material mass fraction"):
            insert(
                [
                    (key, electrode_types),
                    ("hasCoating", COATING_TYPES),
                    ("hasActiveMaterial", None),
                    ("hasMeasuredProperty", "MassFraction"),
                ],
                {
                    "@type": "MassFraction",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": am_mass_frac},
                    "hasMeasurementUnit": "unit:FRACTION",
                },
            )

        # Areal capacity
        if a_cap := sample_data.get(f"{xode} C-rate definition areal capacity (mAh/cm2)"):
            insert(
                [
                    (key, electrode_types),
                    ("hasCoating", COATING_TYPES),
                    ("hasMeasuredProperty", "RatedCapacity"),
                ],
                {
                    "@type": "RatedCapacity",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": a_cap},
                    "hasMeasurementUnit": "emmo:MilliAmpereHourPerSquareCentiMetre",
                    "rdfs:comment": "Areal capacity provided by user to define C-rate",
                },
            )

        # Specific capacity
        m_cap1 = sample_data.get(f"{xode} C-rate definition specific capacity (mAh/g)")
        m_cap2 = sample_data.get(f"{xode} balancing specific capacity (mAh/g)")
        if (m_cap1 == m_cap2) or (bool(m_cap1) ^ bool(m_cap2)):
            m_cap = m_cap1 or m_cap2
            insert(
                [
                    (key, electrode_types),
                    ("hasCoating", COATING_TYPES),
                    ("hasMeasuredProperty", "RatedCapacity"),
                ],
                {
                    "@type": "RatedCapacity",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": m_cap},
                    "hasMeasurementUnit": "unit:MilliA-HR-PER-GM",
                    "rdfs:comment": "Specific capacity provided by user to define C-rate",
                },
            )
        elif m_cap1 != m_cap2:
            insert(
                [
                    (key, electrode_types),
                    ("hasCoating", COATING_TYPES),
                    ("hasMeasuredProperty", "RatedCapacity"),
                ],
                {
                    "@type": "RatedCapacity",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": m_cap1},
                    "hasMeasurementUnit": "unit:MilliA-HR-PER-GM",
                    "rdfs:comment": "Specific capacity provided by user to define C-rate",
                },
            )
            insert(
                [
                    (key, electrode_types),
                    ("hasCoating", COATING_TYPES),
                    ("hasMeasuredProperty", "RatedCapacity"),
                ],
                {
                    "@type": "RatedCapacity",
                    "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": m_cap2},
                    "hasMeasurementUnit": "unit:MilliA-HR-PER-GM",
                    "rdfs:comment": "Specific capacity provided by user for calculating N:P ratio",
                },
            )

    # Electrolyte volume
    if vol := sample_data.get("Electrolyte amount (uL)"):
        insert(
            [
                ("hasElectrolyte", ELECTROLYTE_TYPES),
                ("hasMeasuredProperty", "Volume"),
            ],
            {
                "@type": "Volume",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": vol},
                "hasMeasurementUnit": "unit:MicroL",
            },
        )

    # Electrolyte name, desc
    if name := sample_data.get("Electrolyte name"):
        insert([("hasElectrolyte", ELECTROLYTE_TYPES)], {"schema:name": name})
    if desc := sample_data.get("Electrolyte description"):
        insert([("hasElectrolyte", ELECTROLYTE_TYPES)], {"schema:description": desc})

    # Separator
    if name := sample_data.get("Separator type"):
        insert([("hasSeparator", "Separator")], {"schema:name": name})
    if diam := sample_data.get("Separator diameter (mm)"):
        insert(
            [("hasSeparator", "Separator"), ("hasMeasuredProperty", "Diameter")],
            {
                "@type": "Diameter",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": diam},
                "hasMeasurementUnit": "unit:MilliM",
            },
        )
    if thick := sample_data.get("Separator  thickness (mm)"):
        insert(
            [("hasSeparator", "Separator"), ("hasMeasuredProperty", "Thickness")],
            {
                "@type": "Thickness",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": thick},
                "hasMeasurementUnit": "unit:MilliM",
            },
        )

    # Casing
    if name := sample_data.get("Casing type"):
        insert(
            [("hasCase", None)],
            {
                "schema:name": name,
            },
        )
    if material := sample_data.get("Casing material"):
        insert(
            [("hasCase", None)],
            {
                "rdfs:comment": f"Material: {material}",
            },
        )

    # Spacers
    bottom_electrode: str | None = None
    if history := sample_data.get("Assembly history"):
        bottom_electrode = next((a["Step"] for a in history if a["Step"] in {"Anode", "Cathode"}), None)
    top_electrode = {"Anode": "Cathode", "Cathode": "Anode"}.get(bottom_electrode)
    rename = {"Anode": "NegativeElectrode", "Cathode": "PositiveElectrode"}
    bottom_electrode = rename.get(bottom_electrode)
    top_electrode = rename.get(top_electrode)

    # Check for existing spacers
    components = coin_cell.get("hasComponent", [])
    components = components if isinstance(components, list) else [components]
    unclaimed = [i for i, el in enumerate(components) if isinstance(el, dict) and has_type(el, "Spacer")]

    for sp, electrode in (("Top spacer", top_electrode), ("Bottom spacer", bottom_electrode)):
        name = sample_data.get(f"{sp} type")
        thickness = (
            {
                "@type": "Thickness",
                "hasNumericalPart": {
                    "@type": "emmo:RealData",
                    "hasNumberValue": sample_data.get(f"{sp} thickness (mm)"),
                },
                "hasMeasurementUnit": "unit:MilliM",
            }
            if sample_data.get(f"{sp} thickness (mm)")
            else None
        )
        if not name and not thickness:
            continue
        diameter = (
            {
                "@type": "Diameter",
                "hasNumericalPart": {
                    "@type": "emmo:RealData",
                    "hasNumberValue": sample_data.get(f"{sp} diameter (mm)"),
                },
                "hasMeasurementUnit": "unit:MilliM",
            }
            if sample_data.get(f"{sp} diameter (mm)")
            else None
        )
        spacer_dict: dict[str, str | list | dict] = {"@type": "Spacer", "rdfs:comment": f"{sp} during assembly"}
        if name:
            spacer_dict["schema:name"] = name
        if electrode:
            spacer_dict["contacts"] = {"@type": electrode}
        if mat := sample_data.get(f"{sp} material"):
            spacer_dict["rdfs:comment"] = [spacer_dict["rdfs:comment"], f"Material: {mat}"]
        if thickness and diameter:
            spacer_dict["hasMeasuredProperty"] = [thickness, diameter]
        elif thickness:
            spacer_dict["hasMeasuredProperty"] = thickness
        elif diameter:
            spacer_dict["hasMeasuredProperty"] = diameter

        # Fill the spacer touching the same electrode, else one with no electrode given
        idx = next((i for i in unclaimed if _touches(components[i], electrode)), None)
        if idx is None:
            idx = next((i for i in unclaimed if not components[i].get("contacts")), None)
        if idx is None:
            components.append(spacer_dict)
        else:
            unclaimed.remove(idx)
            spacer = components[idx]
            for prop in as_list(spacer_dict.pop("hasMeasuredProperty", None)):
                insert_dict_in_jsonld(
                    spacer,
                    [("hasMeasuredProperty", prop["@type"])],
                    prop,
                    conflicts=conflicts,
                    _path="hasComponent/Spacer",
                )
            if conflicts is not None:
                conflicts.extend(f"hasComponent/Spacer/{c}" for c in _text_conflicts(spacer, spacer_dict))
            _deep_merge_dicts(spacer, spacer_dict)
        coin_cell["hasComponent"] = components

    # N:P Ratio
    if np := sample_data.get("N:P ratio"):
        insert(
            [("hasMeasuredProperty", "NPRatio")],
            {
                "@type": "NPRatio",
                "hasNumericalPart": {
                    "@type": "emmo:RealData",
                    "hasNumberValue": np,
                },
                "hasMeasurementUnit": "unit:FRACTION",
                "rdfs:comment": "Based on active material mass, accounting for the overlapping area.",
            },
        )

    return battinfo_jsonld


def has_type(jsonld: dict, target_type: str) -> bool:
    """Check a node's @type, which may be a string or a list of strings."""
    types = jsonld.get("@type")
    return types == target_type or (isinstance(types, list) and target_type in types)


def find_coin_cell(jsonld: dict) -> dict | None:
    """Search for the CoinCell in a dict."""
    if has_type(jsonld, "CoinCell"):
        return jsonld
    for value in jsonld.values():
        if isinstance(value, dict):
            result = find_coin_cell(value)
            if result is not None:
                return result
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    result = find_coin_cell(item)
                    if result is not None:
                        return result
    return None


def summarise_assembly(assembly: list[dict], sample_data: dict) -> str:
    """Summarise an assembly process."""
    mapping = {
        "Bottom": "CellCan",
        "Top": "CellLid",
        "Anode": "NegativeElectrode",
        "Separator": "Separator",
        "Cathode": "PositiveElectrode",
        "Spring": "Spring",
    }
    for step in assembly:
        if step["Step"] in mapping:
            step["Summary"] = mapping[step["Step"]]
        elif step["Step"] == "Spacer":
            if "bottom" in step["Description"].lower():
                thickness = sample_data.get("Bottom spacer thickness (mm)")
                step["Summary"] = f"{thickness} mm Spacer"
            elif "top" in step["Description"].lower():
                thickness = sample_data.get("Top spacer thickness (mm)")
                if not thickness:  # duct tape
                    thickness = sample_data.get("Bottom spacer thickness (mm)")
                step["Summary"] = f"{thickness} mm Spacer"
        elif step["Step"] == "Electrolyte":
            if "before" in step["Description"].lower():
                amount = sample_data.get("Electrolyte amount before separator (uL)")
                step["Summary"] = f"{amount} uL Electrolyte"
            elif "after" in step["Description"].lower():
                amount = sample_data.get("Electrolyte amount after separator (uL)")
                step["Summary"] = f"{amount} uL Electrolyte"
    summaries = [step.get("Summary") for step in assembly]
    return "Cell assembly sequence: " + ", ".join([s for s in summaries if s is not None])


def make_type_parent(data: dict, target_type: str) -> dict:
    """Promote object with target @type to the top level.

    Anything referencing this type is put in @reverse.
    """
    if isinstance(data, dict) and has_type(data, target_type):
        return data
    if isinstance(data, dict) and data.get("@reverse") is not None:
        msg = "Cannot rearrange object if @reverse in json-ld"
        raise ValueError(msg)

    data = deepcopy(data)
    ctx = data.pop("@context") if "@context" in data else None

    def find_target_and_parent(
        obj: dict | list | str | float | None = None,
        parent: dict | list | str | float | None = None,
        key: str | None = None,
        index: int | None = None,
    ) -> tuple:
        if isinstance(obj, dict):
            if has_type(obj, target_type):
                return obj, parent, key, index
            for k, v in obj.items():
                result = find_target_and_parent(v, obj, k, None)
                if result[0] is not None:
                    return result
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                result = find_target_and_parent(item, parent, key, i)
                if result[0] is not None:
                    return result
        return None, None, None, None

    target, parent, key, index = find_target_and_parent(data)
    if target is None:
        msg = f"No node with @type '{target_type}' found."
        raise ValueError(msg)

    # Remove the CoinCell from its parent and replace with @id or remove
    placeholder = {"@id": "_:promoted"}
    assert isinstance(parent, (list, dict))  # noqa: S101  from intial check
    if isinstance(parent[key], list):
        # Replace only the matched item in the list
        parent[key][index] = placeholder
    else:
        parent[key] = placeholder

    # Build reversed object
    reversed_obj = deepcopy(parent)

    # Replace placeholder with original value in reversed object for display
    if isinstance(reversed_obj[key], list):
        # Filter out placeholder from list and keep other elements
        reversed_obj[key] = [item for item in reversed_obj[key] if item != placeholder]
        if len(reversed_obj[key]) == 1:
            reversed_obj[key] = reversed_obj[key][0]
    else:
        del reversed_obj[key]  # It was a direct object reference

    result = deepcopy(target)
    result["@reverse"] = {key: reversed_obj}
    if ctx:
        result["@context"] = ctx
    return result


def merge_contexts_strict(
    ctx1: str | list | dict,
    ctx2: str | list | dict,
    on_conflict: Literal["raise", "keep_left", "keep_right"] = "raise",
) -> list:
    """Merge the top level @context blocks of two JSON-LD."""
    ctx1_list = ctx1 if isinstance(ctx1, list) else [ctx1]
    ctx2_list = ctx2 if isinstance(ctx2, list) else [ctx2]

    def process_context_list(ctx_list: list) -> tuple[list, dict]:
        """Process a context list, split remotes and terms."""
        remotes = []
        terms: dict[str, str] = {}
        for ctx in ctx_list:
            if isinstance(ctx, str):
                remotes.append(ctx)
            elif isinstance(ctx, dict):
                terms.update(ctx.items())
        return remotes, terms

    remotes1, terms1 = process_context_list(ctx1_list)
    remotes2, terms2 = process_context_list(ctx2_list)

    # Merge remote contexts (keep unique)
    merged_remotes = list(dict.fromkeys(remotes1 + remotes2))  # preserves order, unique

    # Check for conflicts in namespace definitions
    for term, url in terms2.items():
        if term in terms1 and terms1[term] != url:
            if on_conflict == "raise":
                msg = f"Merging context: conflict for term '{term}': '{terms1[term]}' != '{url}'"
                raise ValueError(msg)
            logger.warning(
                "Merging context: conflict for term '%s': '%s' != '%s', keeping %s",
                term,
                terms1[term],
                url,
                "left" if on_conflict == "keep_left" else "right",
            )

    # Merge term definitions
    merged_terms = {**terms2, **terms1} if on_conflict == "keep_left" else {**terms1, **terms2}

    # Construct merged context list
    merged_context = [*merged_remotes]
    if merged_terms:
        merged_context.append(merged_terms)

    return merged_context


def recursive_merge(left: dict, right: dict, *, default_right: bool = True) -> dict:
    """Recursively merge dicts."""
    for k, rv in right.items():
        if k not in left:
            left[k] = rv
        else:
            lv = left[k]
            if isinstance(lv, dict) and isinstance(rv, dict):
                if lv.get("@id") and rv.get("@id") and lv["@id"] != rv["@id"]:
                    left[k] = dedupe_jsonld_list([lv, rv])
                else:
                    left[k] = recursive_merge(lv, rv, default_right=default_right)
            elif isinstance(lv, list) and isinstance(rv, list):
                left[k] = dedupe_jsonld_list([*lv, *rv])
            elif isinstance(lv, list):
                left[k] = dedupe_jsonld_list([*lv, rv])
            elif isinstance(rv, list):
                left[k] = dedupe_jsonld_list([lv, *rv])
            elif lv != rv:
                if default_right:
                    left[k] = rv
                    logger.warning(
                        "JSON-LD merge conflict at %s: left - %s, right - %s, defaulting to right", k, lv, rv
                    )
                else:
                    logger.warning("JSON-LD merge conflict at %s: left - %s, right - %s, defaulting to left", k, lv, rv)
            else:
                left[k] = lv  # unchanged
    return left


def dedupe_jsonld_list(lst: list) -> list:
    """Remove duplicates from a list of JSON-LD values (dict-aware)."""
    seen = set()
    deduped = []
    for item in lst:
        # Convert dicts to a frozen string for comparison
        key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


# Keys whose values from every source are combined instead of the higher priority one winning
UNION_KEYS = {"@type", "dcat:distribution", "schema:associatedMedia"}


def _short(value: object, limit: int = 150) -> str:
    """Compact one-line JSON for log messages."""
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def _plain(value: object) -> object:
    """Reduce a node that is only an @id to the @id, so a link and its IRI string compare equal."""
    if isinstance(value, dict) and set(value) == {"@id"}:
        return value["@id"]
    return value


def _union(high: object, low: object, names: tuple[str, str] | None, path: str, conflicts: list[str] | None) -> object:
    """Combine two values or lists, merging nodes that share an @id."""
    merged = deepcopy(high) if isinstance(high, list) else [deepcopy(high)]
    for item in low if isinstance(low, list) else [low]:
        match = next(
            (
                i
                for i, existing in enumerate(merged)
                if isinstance(item, dict)
                and isinstance(existing, dict)
                and item.get("@id") is not None
                and existing.get("@id") == item.get("@id")
            ),
            None,
        )
        if match is not None:
            merged[match] = priority_merge(merged[match], item, names=names, conflicts=conflicts, _path=path)
        elif item not in merged:
            merged.append(deepcopy(item))
    return merged[0] if len(merged) == 1 else merged


def priority_merge(
    high: dict,
    low: dict,
    *,
    names: tuple[str, str] | None = None,
    conflicts: list[str] | None = None,
    _path: str = "",
) -> dict:
    """Merge two JSON-LD nodes, keeping the values of `high` where both set a key.

    Keys in UNION_KEYS are combined instead, and @context blocks are merged. `names`
    labels the two sources in a warning for each disagreement, also appended to `conflicts`.
    """
    merged = deepcopy(high)
    for key, low_value in low.items():
        path = f"{_path}/{key}"
        if key not in merged:
            merged[key] = deepcopy(low_value)
            continue
        high_value = merged[key]
        if key == "@context":
            merged[key] = merge_contexts_strict(high_value, low_value, on_conflict="keep_left")
        elif key in UNION_KEYS:
            merged[key] = _union(high_value, low_value, names, path, conflicts)
        elif (
            isinstance(high_value, dict)
            and isinstance(low_value, dict)
            and high_value.get("@id") == low_value.get("@id")
        ):
            merged[key] = priority_merge(high_value, low_value, names=names, conflicts=conflicts, _path=path)
        elif _plain(high_value) != _plain(low_value) and names:
            msg = (
                f"{names[0]} and {names[1]} disagree at {path.lstrip('/')}, using {names[0]}: "
                f"{_short(high_value)}, ignoring: {_short(low_value)}"
            )
            logger.warning("%s", msg)
            if conflicts is not None:
                conflicts.append(msg)
    return merged


def _sheet_file_path(node: dict) -> str | None:
    """Get the path of a file node written by the BattINFO converter.

    Unpublished files carry the path as dcterms:identifier, published ones in their
    download URL, after the "#" for a file inside a zip.
    """
    if node.get("dcterms:identifier"):
        return node["dcterms:identifier"]
    file_id = node.get("@id")
    if not file_id:
        return None
    for separator in ("#", "/files/"):
        if separator in file_id:
            return unquote(file_id.split(separator, 1)[1])
    return unquote(file_id)


class SheetFile(NamedTuple):
    """A file listed in the BattINFO sheet: its section (hasInput/hasOutput) and description."""

    section: str
    comment: str | None


def pop_sheet_files(battinfo_jsonld: dict) -> dict[str, SheetFile]:
    """Remove file nodes from the converter output, return each path's section and description."""
    files: dict[str, SheetFile] = {}
    for section in ("hasOutput", "hasInput"):
        node = battinfo_jsonld.get(section)
        if not isinstance(node, dict) or "dcat:distribution" not in node:
            continue
        distribution = node.pop("dcat:distribution")
        for item in distribution if isinstance(distribution, list) else [distribution]:
            path = _sheet_file_path(item) if isinstance(item, dict) else None
            if path:
                files[path] = SheetFile(section, item.get("rdfs:comment"))
        # A section that only described files would leave its @type on aurora's files
        if set(node) <= {"@type"}:
            del battinfo_jsonld[section]
    return files


def unwrap_graph(jsonld: dict) -> dict:
    """Replace a single-node @graph wrapper with the node itself."""
    if "@graph" not in jsonld:
        return jsonld
    graph = jsonld["@graph"]
    if not isinstance(graph, list) or len(graph) != 1:
        msg = "Multi-node @graph not supported."
        raise ValueError(msg)
    node = deepcopy(graph[0])
    if ctx := jsonld.get("@context"):
        node["@context"] = (
            merge_contexts_strict(ctx, node["@context"], on_conflict="keep_right") if node.get("@context") else ctx
        )
    return node


def make_test_object(battinfo_jsonld: dict) -> dict:
    """Put BattINFO coin cell description inside a BatteryTest object."""
    battinfo_jsonld = unwrap_graph(battinfo_jsonld)
    if has_type(battinfo_jsonld, "CoinCell"):
        ctx = battinfo_jsonld.pop("@context", None)
        return {
            **({"@context": ctx} if ctx else {}),
            "@type": "BatteryTest",
            "hasTestObject": battinfo_jsonld,
        }
    if has_type(battinfo_jsonld, "BatteryTest"):
        return battinfo_jsonld
    msg = "BattINFO JSON-LD must have CoinCell or BatteryTest as top level @type"
    raise ValueError(msg)


def merge_jsonld(json1: dict, json2: dict) -> dict:
    """Merge two JSON-LD structures assuming they reference the SAME NODE."""
    if not isinstance(json1, dict) or not isinstance(json2, dict):
        raise TypeError
    if "@type" not in json1 or json1.get("@type") != json2.get("@type"):
        msg = "Two JSON-LDs do not have the same parent node"
        raise ValueError(msg)
    return recursive_merge(json1, json2)


def merge_jsonld_on_type(jsons: list[dict], target_type: str = "BatteryTest") -> dict:
    """Transform list of json-ld dicts, make target_type parent and merge."""
    jsons = deepcopy(jsons)  # Don't modify originals
    contexts = [j.pop("@context") for j in jsons if "@context" in j]
    jsons = [make_type_parent(j, target_type) for j in jsons]

    def repeated_context_merge(contexts: list[str | list | dict]) -> str | list | dict | None:
        if len(contexts) == 0:
            return None
        if len(contexts) == 1:
            return contexts[0]
        merged = merge_contexts_strict(contexts[0], contexts[1], on_conflict="keep_right")
        if len(contexts) == 2:
            return merged
        return repeated_context_merge([merged, *contexts[2:]])

    def repeated_merge(jsonld_list: list[dict]) -> dict:
        if len(jsonld_list) == 1:
            return jsonld_list[0]
        return recursive_merge(jsonld_list[0], repeated_merge(jsonld_list[1:]))

    merged_context = repeated_context_merge(contexts)
    merged_content = repeated_merge(jsons)

    return {
        **({"@context": merged_context} if merged_context else {}),
        **merged_content,
    }


def generate_battery_test(ontologized_protocols: dict | list[dict]) -> dict:
    """Generate test json-ld based on protocols."""
    if isinstance(ontologized_protocols, dict):
        ontologized_protocols = [ontologized_protocols]

    def recursive_protocol(ontologized_protocols: list[dict]) -> dict:
        test = {
            "@type": ["ConstantCurrentConstantVoltageCycling", "IterativeWorkflow"],
            "rdfs:label": "GeneratedBatteryTestProcedure",
            "hasTask": ontologized_protocols[0],
        }
        if len(ontologized_protocols) > 1:
            test["hasNext"] = recursive_protocol(ontologized_protocols[1:])
        return test

    return {
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
        "@type": "BatteryTest",
        "hasLaboratory": {
            "@type": "Laboratory",
            "@id": "https://www.wikidata.org/wiki/Q683116",
            "rdfs:label": "Empa",
        },
        "hasMeasurementParameter": recursive_protocol(ontologized_protocols),
    }


def generate_basic_output_jsonld() -> dict:
    """Generate default 'hasOutput' values for BatteryTest, used where nothing else sets them."""
    today = datetime.now().date().isoformat()  # noqa: DTZ005
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "@type": ["BatteryTestResult", "dcat:Dataset"],
            "dcat:keyword": ["battery", "R2032", "coin cell"],
            "dcterms:license": {"@id": "https://creativecommons.org/licenses/by/4.0/"},
            "dcterms:issued": today,
            "schema:datePublished": today,
        },
    }


def add_ccid_output(
    ccid: str,
) -> dict:
    """Add CCID to output section of json-ld output."""
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "dcterms:title": f"Cycling data coin cell {ccid}",
            "dcterms:description": f"Cycling data coin cell {ccid}",
        },
    }


def zenodo_record_id(zenodo_url: str) -> str:
    """Extract the numeric Zenodo record ID from a Zenodo DOI or record page URL.

    e.g. "https://doi.org/10.5281/zenodo.20338409" or "https://zenodo.org/records/20338409"
    -> "20338409". Raises ValueError for a URL with neither form.
    """
    match = _ZENODO_RECORD_RE.search(zenodo_url)
    if match is None:
        msg = (
            f"Cannot find a Zenodo record number in {zenodo_url!r}, expected e.g. "
            "https://doi.org/10.5281/zenodo.12345678 or https://zenodo.org/records/12345678"
        )
        raise ValueError(msg)
    return match.group(1) or match.group(2)


def zenodo_download_url(
    rel_file_path: str,
    zenodo_doi_url: str | None,
    *,
    zip_filename: str | None = None,
) -> str | None:
    """Build the Zenodo download URL for a file inside the zip uploaded to Zenodo.

    Zenodo cannot link into a zip, so the path within it is a fragment after "#".
    Returns None without both a zenodo_doi_url and a zip_filename.
    """
    if not zenodo_doi_url or not zip_filename:
        return None
    record_id = zenodo_record_id(zenodo_doi_url)
    return f"https://zenodo.org/records/{record_id}/files/{quote(zip_filename, safe='')}#{quote(rel_file_path)}"


def add_data(
    rel_file_path: str,
    zenodo_doi_url: str | None,
    *,
    zip_filename: str | None = None,
) -> dict:
    """Add links to data files to output section of json-ld output.

    Args:
        rel_file_path: path to file in the zip, e.g. empa__ccid01345/empa__ccid01345.bdf.parquet,
            also used as its "@id" so it matches the RO-Crate
        zenodo_doi_url: path to zenodo archive e.g. https://doi.org/10.1234/zenodo.12345678
        zip_filename: name of the zip on Zenodo, with zenodo_doi_url gives a download URL

    Returns:
        dict with "BatteryTest" as top level type.

    """
    suffix = Path(rel_file_path).suffix
    if suffix not in MEDIA_TYPES:
        msg = f"Unknown file type: {rel_file_path}"
        raise ValueError(msg)
    dist: dict = {
        "@id": quote(rel_file_path),
        "@type": "dcat:Distribution",
        "dcat:mediaType": {"@id": IANA_MEDIA_TYPES + MEDIA_TYPES[suffix]},
    }
    if ".bdf." in Path(rel_file_path).name:
        dist["csvw:tableSchema"] = {"@id": BDF_SCHEMA_IRI}
    if suffix == ".csv":
        dist["csvw:dialect"] = {"@type": "csvw:Dialect", "csvw:delimiter": ",", "csvw:skipRows": 0}
    if download_url := zenodo_download_url(rel_file_path, zenodo_doi_url, zip_filename=zip_filename):
        dist["dcat:downloadURL"] = {"@id": download_url}
    return {
        "@type": "BatteryTest",
        "hasOutput": {"dcat:distribution": dist},
    }


def add_zenodo_url(
    zenodo_doi_url: str,
) -> dict:
    """Add Zenodo URL to output section of json-ld output.

    Also sets "@base" to the Zenodo record, so the files' relative "@id"s become unique IRIs.
    """
    record_id = zenodo_record_id(zenodo_doi_url)
    return {
        "@context": [{"@base": f"https://zenodo.org/records/{record_id}/"}],
        "@type": "BatteryTest",
        "hasOutput": {
            "dcat:accessURL": {"@id": zenodo_doi_url},
            "dcat:endpointURL": {"@id": f"https://zenodo.org/api/records/{record_id}"},
        },
    }


def add_associated_media(
    paper_doi_url: str | None,
    sample_to_fig: dict,
    ccid: str | None,
    sample_id: str,
) -> dict:
    """Add associated media to output section of json-ld output."""
    figs = [v for k, v in sample_to_fig.items() if k in (sample_id, ccid)]  # e.g. ["Fig. 3a", "Fig. 3b"]
    if not figs:
        return {"@type": "BatteryTest"}
    associated_media: dict[str, str | list] = {"@id": paper_doi_url} if paper_doi_url else {}
    associated_media["rdfs:label"] = figs[0] if len(figs) == 1 else figs
    associated_media["rdfs:comment"] = FIGURE_COMMENT
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "schema:associatedMedia": associated_media,
        },
    }


def add_citation(
    citation_string: str,
) -> dict:
    """Add publishing details to output section of json-ld output."""
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "schema:citation": citation_string,
        },
    }


def add_authors(
    authors: list[dict],
    institutions: dict[str, dict],
) -> dict:
    """Add authors to output section of json-ld.

    Args:
        authors: a list of dictionaries, each describing one author
            the dict must include the keys:
                "orcid": str e.g. https://orcid.org/1234-1234-1234-1234
                "name": str e.g. "John Battery"
                "affiliation": list[str] e.g. ["Empa", "ETH Zurich", "EPFL"]
        institutions: a dict of dictionaries, each describing one institution
            e.g.
                {
                    "Empa": {"wikidata_url": "https://www.wikidata.org/wiki/Q683116"}
                    "ETH Zurich": {"wikidata_url": "https://www.wikidata.org/wiki/Q11942"}
                    "EPFL": {"wikidata_url": "https://www.wikidata.org/wiki/Q262760"}
                }

    Returns:
        dict with "BatteryTest" as top level type.

    """

    def make_author(author: dict) -> dict:
        auth_dict: dict[str, str | list | dict] = {"@type": "schema:Person"}
        if author.get("orcid"):
            auth_dict["@id"] = author["orcid"]
        auth_dict["schema:name"] = author["name"]
        affiliations = [
            {
                "@type": "schema:ResearchOrganization",
                "@id": institutions[affil]["wikidata_url"],
                "schema:name": affil,
            }
            for affil in author["affiliation"]
        ]
        if affiliations:
            auth_dict["schema:affiliation"] = affiliations[0] if len(affiliations) == 1 else affiliations
        return auth_dict

    authors_jsonld = [make_author(author) for author in authors]
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "dcterms:creator": authors_jsonld[0] if len(authors_jsonld) == 1 else authors_jsonld,
        },
    }


def add_software(
    name: str = __title__,
    version: str = __version__,
    url: str = __url__,
) -> dict:
    """Record the software and time that generated the output dataset."""
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "prov:wasGeneratedBy": {
                "@type": ["prov:Activity", "schema:CreateAction"],
                "schema:name": f"{name} export",
                "schema:endTime": datetime.now().astimezone().isoformat(timespec="seconds"),
                "schema:instrument": {
                    "@id": url,
                    "@type": "schema:SoftwareApplication",
                    "schema:name": name,
                    "schema:softwareVersion": version,
                },
            },
        },
    }


def add_institution(
    name: str = "Empa",
    wikidata_url: str = "https://www.wikidata.org/wiki/Q683116",
) -> dict:
    """Add publishing details to output section of json-ld."""
    return {
        "@type": "BatteryTest",
        "hasOutput": {
            "dcterms:publisher": {
                "@type": "schema:ResearchOrganization",
                "@id": wikidata_url,
                "schema:name": name,
            },
        },
    }


# Cell values specific to one sample: BattINFO key, label in the BattINFO sheet, the sample's value
SAMPLE_SPECIFIC_FIELDS = (
    ("schema:productID", "Cell ID", lambda s: s.get("Barcode")),
    ("schema:name", "Cell name", lambda s: s.get("Sample ID")),
    ("schema:dateCreated", "Date of cell assembly", assembly_date),
)


def check_battinfo_for_samples(battinfo_jsonld: dict, samples: list[dict]) -> tuple[list[str], list[str]]:
    """Check a BattINFO JSON-LD can be applied to every sample, return errors and warnings.

    Values specific to one cell must match each sample, or be empty. Listed files are
    ignored, and values the database replaces are reported, both as warnings.
    """
    test = make_test_object(deepcopy(battinfo_jsonld))
    cell = find_coin_cell(test) or {}
    errors = []
    for key, label, sample_value in SAMPLE_SPECIFIC_FIELDS:
        if (value := cell.get(key)) in (None, ""):
            continue
        mismatched = []
        for sample in samples:
            expected = sample_value(sample)
            if expected is None and len(samples) > 1:
                mismatched.append(f"{sample['Sample ID']} (not in database)")
            elif expected is not None and str(value) != str(expected):
                mismatched.append(f"{sample['Sample ID']} ('{expected}')")
        if mismatched:
            errors.append(f"{label} '{value}' does not match: " + ", ".join(mismatched))

    output = test.get("hasOutput")
    media = as_list(output.get("schema:associatedMedia")) if isinstance(output, dict) else []
    if len(samples) > 1 and any(isinstance(m, dict) and m.get("rdfs:label") for m in media):
        errors.append(f"Associated publication figures are for one sample, but it is applied to {len(samples)}")

    warnings = []
    if files := pop_sheet_files(test):
        warnings.append("Ignoring listed files, aurora adds the files it exports: " + ", ".join(files))
    for sample in samples:
        conflicts: list[str] = []
        merge_battinfo_with_db_data(deepcopy(battinfo_jsonld), deepcopy(sample), conflicts=conflicts)
        warnings.extend(f"{sample['Sample ID']}: {c} from the database" for c in conflicts)
    return errors, warnings


def _zenodo_record_or_url(url: str) -> str:
    """Get the Zenodo record ID of a URL, or the URL itself if it is not a Zenodo record."""
    try:
        return zenodo_record_id(url)
    except ValueError:
        return url


def build_export_metadata(
    battinfo_jsonld: dict,
    file_snippets: list[dict],
    pub_info: dict,
    *,
    sample_id: str,
    ccid: str | None = None,
) -> tuple[dict, list[str]]:
    """Combine a sample's stored BattINFO with what the export knows, return it and any conflicts.

    The export's files and publication info win over the stored BattINFO, which wins over defaults.
    """
    conflicts: list[str] = []
    stored = make_test_object(deepcopy(battinfo_jsonld))
    if files := pop_sheet_files(stored):
        conflicts.append("Ignoring files listed in BattINFO metadata: " + ", ".join(files))

    export = [{"@type": "BatteryTest"}, *file_snippets, add_software()]
    if zenodo_url := pub_info.get("zenodo_doi_url"):
        try:
            export.append(add_zenodo_url(zenodo_url))
        except ValueError as e:
            conflicts.append(str(e))
        else:
            # The export's record replaces any in the stored metadata
            output = stored.get("hasOutput")
            stored_url = _plain(output.pop("dcat:accessURL", None)) if isinstance(output, dict) else None
            if isinstance(output, dict):
                output.pop("dcat:endpointURL", None)
            if (
                isinstance(output, dict)
                and isinstance(stored_url, str)
                and _zenodo_record_or_url(stored_url) != zenodo_record_id(zenodo_url)
            ):
                # The rest describes that other dataset, only its publications still apply to the sample
                ignored = [k for k in output if k not in {"@type", "schema:associatedMedia"}]
                for key in ignored:
                    del output[key]
                conflicts.append(
                    f"Ignoring Zenodo record {stored_url} in BattINFO metadata, using {zenodo_url}"
                    + (f", also ignoring its {', '.join(ignored)}" if ignored else "")
                )
    if pub_info.get("citation_string"):
        export.append(add_citation(pub_info["citation_string"]))
    if pub_info.get("authors") and pub_info.get("institutions"):
        export.append(add_authors(pub_info["authors"], pub_info["institutions"]))
    if pub_info.get("sample_to_fig"):
        export.append(
            add_associated_media(pub_info.get("publication_doi_url"), pub_info["sample_to_fig"], ccid, sample_id)
        )
    defaults = merge_jsonld_on_type([generate_basic_output_jsonld(), add_ccid_output(ccid or sample_id)])

    merged = priority_merge(
        merge_jsonld_on_type(export), stored, names=("export", "BattINFO metadata"), conflicts=conflicts
    )
    merged = priority_merge(merged, defaults)
    # Keep the stored document's key order, with @context first
    return {k: merged[k] for k in ["@context", *stored, *merged] if k in merged}, conflicts


def parse_zenodo_info_xlsx(
    xlsx_file: str | Path,
) -> dict:
    """Parse zenodo info from uploaded xlsx content string.

    Args:
        xlsx_file: Path to a file, a content string from upload button

    Returns:
        dict with zenodo info

    """
    if isinstance(xlsx_file, Path):
        excel_file = pd.ExcelFile(xlsx_file)
    else:
        _content_type, content_string = xlsx_file.split(",")
        decoded = base64.b64decode(content_string)
        excel_file = pd.ExcelFile(io.BytesIO(decoded))
    sheet_names = [str(s) for s in excel_file.sheet_names]
    expected_sheets = ["General", "Figures", "Authors", "Institutions"]
    if not all(sheet in expected_sheets for sheet in sheet_names):
        msg = "Uploaded xlsx does not have expected sheets for Zenodo info."
        raise ValueError(msg)

    # Read General sheet
    general_df = pd.read_excel(excel_file, sheet_name="General", engine="openpyxl", header=None)
    keys = general_df[0].to_numpy()
    keys = [k.lower().replace(" ", "_") for k in keys]
    values = general_df[1].to_numpy()
    general_dict = dict(zip(keys, values, strict=True))

    # Read Figures sheet
    figures_df = pd.read_excel(excel_file, sheet_name="Figures", engine="openpyxl")
    figures_dict = {}
    for _, row in figures_df.iterrows():
        key = row["Barcode"] if pd.notna(row["Barcode"]) else row["Sample ID"]
        figures = row.drop(labels=["Sample ID", "Barcode"]).dropna().astype(str).tolist()
        figures_dict[key] = figures

    # Read Institutions sheet
    institutions_df = pd.read_excel(excel_file, sheet_name="Institutions", engine="openpyxl")
    institutions_df.columns = [col.lower().replace(" ", "_") for col in institutions_df.columns]
    institutions_dict = institutions_df.set_index("name").to_dict(orient="index")

    # Read Authors sheet
    authors_df = pd.read_excel(excel_file, sheet_name="Authors", engine="openpyxl")
    authors_list = []
    for _, row in authors_df.iterrows():
        affil_list = row.drop(labels=["ORCID", "Name"]).dropna().astype(str).tolist()
        authors_list.append(
            {
                "name": row["Name"],
                "orcid": row["ORCID"],
                "affiliation": affil_list,
            }
        )

    return {
        **general_dict,
        "sample_to_fig": figures_dict,
        "institutions": institutions_dict,
        "authors": authors_list,
    }


def generate_zenodo_info_xlsx_template(
    sample_ids: list[str],
    ccids: list[str],
) -> io.BytesIO:
    """Generate a template excel file for zenodo info.

    Args:
        sample_ids: list of sample IDs
        ccids: list of CCIDs

    Returns:
        pd.DataFrame with template data

    """
    general_sheet = {
        "Citation string": "Enter citation here, like Author et al., Journal, Year",
        "Zenodo DOI URL": "Enter Zenodo link here, e.g. https://doi.org/10.1234/zenodo.12345678",
        "Publication DOI URL": "Enter Publication DOI link here, e.g. https://doi.org/10.1234/journal.12345678",
    }
    figures_sheet = {
        "Sample ID": sample_ids,
        "Barcode": ccids,
        "Figures": [None] * len(sample_ids),
    }
    authors_sheet = {
        "Name": ["Firstname Lastname"],
        "ORCID": ["https://orcid.org/0000-0000-0000-0000"],
        "Affiliations": ["Must be a name from the institutions sheet"],
    }
    institutions_sheet = [
        {"Name": "Empa", "Wikidata URL": "https://www.wikidata.org/wiki/Q683116"},
        {"Name": "ETH Zurich", "Wikidata URL": "https://www.wikidata.org/wiki/Q11942"},
        {"Name": "EPFL", "Wikidata URL": "https://www.wikidata.org/wiki/Q262760"},
        {"Name": "University of Bern", "Wikidata URL": "https://www.wikidata.org/wiki/Q659080"},
        {"Name": "UNIL", "Wikidata URL": "https://www.wikidata.org/wiki/Q658975"},
        {"Name": "University of Geneva", "Wikidata URL": "https://www.wikidata.org/wiki/Q503473"},
    ]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_settings = {"align": "left", "top": 1, "bottom": 1, "bold": True}
        title_format = workbook.add_format({**title_settings, "left": 1, "right": 1})
        left_title_format = workbook.add_format({**title_settings, "left": 1, "right": 0})
        middle_title_format = workbook.add_format({**title_settings, "left": 0, "right": 0})
        right_title_format = workbook.add_format({**title_settings, "left": 0, "right": 1})

        pd.DataFrame.from_dict(general_sheet, orient="index", columns=["Value"]).to_excel(
            writer, sheet_name="General", header=False
        )
        worksheet = writer.sheets["General"]
        worksheet.write("A1", "Citation string", title_format)
        worksheet.write("A2", "Zenodo DOI URL", title_format)
        worksheet.write("A3", "Publication DOI URL", title_format)
        worksheet.autofit()

        pd.DataFrame.from_dict(figures_sheet).to_excel(writer, sheet_name="Figures", index=False)
        worksheet = writer.sheets["Figures"]
        worksheet.autofit()
        worksheet.set_column("C:L", 10)
        worksheet.write("A1", "Sample ID", title_format)
        worksheet.write("B1", "Barcode", title_format)
        worksheet.write("C1", "Figures", left_title_format)
        worksheet.write_row("D1:K1", [""] * 8, middle_title_format)
        worksheet.write("L1", "", right_title_format)
        worksheet.write("C2", "Fig. 1a")
        worksheet.write("D2", "Fig. 2b")

        pd.DataFrame.from_dict(authors_sheet).to_excel(writer, sheet_name="Authors", index=False)
        worksheet = writer.sheets["Authors"]
        worksheet.autofit()
        worksheet.set_column("C:L", 16)
        worksheet.write("A1", "Name", title_format)
        worksheet.write("B1", "ORCID", title_format)
        worksheet.write("C1", "Affiliations", left_title_format)
        worksheet.write_row("D1:I1", [""] * 6, middle_title_format)
        worksheet.write("J1", "", right_title_format)
        worksheet.write("A3", "Corsin Battaglia")
        worksheet.write("B3", "https://orcid.org/0000-0002-5003-1134")
        worksheet.write("C3", "Empa")
        worksheet.write("D3", "ETH Zurich")
        worksheet.write("E3", "EPFL")

        pd.DataFrame(institutions_sheet).to_excel(writer, sheet_name="Institutions", index=False)
        worksheet = writer.sheets["Institutions"]
        worksheet.autofit()
        worksheet.write("A1", "Name", title_format)
        worksheet.write("B1", "Wikidata URL", title_format)
    buffer.seek(0)
    return buffer
