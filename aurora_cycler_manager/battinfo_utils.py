"""BattINFO ontology functions."""

import json
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


def _deep_merge_dicts(target: dict, source: dict) -> dict:
    """Recursively merge source into target."""
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge_dicts(target[k], v)
        elif k in target and isinstance(target[k], list) and isinstance(v, list):
            target[k].extend(v)
        else:
            target[k] = v
    return target


def insert_dict_in_jsonld(
    obj: dict | list,
    keys: list[tuple],
    new_dict: dict,
    *,
    merge: bool = True,
) -> None:
    """Insert a dict into a nested jsonld.

    Args:
        obj: The JSON-LD object to put the thing in.
        keys: list of tuples with the key and optional exptected type, e.g.
            [
                ("hasPositiveElectrode", "Electrode"),
                ("hasCoating", "Coating"),
                ("hasActiveMaterial", None),
                ("hasMeasuredProperty", "MassLoading"),
            ]
        new_dict: The dict to insert
        merge (optional): Whether to merge or replace, default True (merge)

    """
    if not keys:
        msg = "Keys list cannot be empty"
        raise ValueError(msg)

    key, expected_type = keys[0]

    # At the end of the chain - could by empty, a dict, or a list
    if len(keys) == 1:
        if key not in obj or obj[key] is None:
            obj[key] = new_dict
        elif isinstance(obj[key], list):
            for i, o in enumerate(obj[key]):
                if isinstance(o, dict) and o.get("@type") == new_dict.get("@type"):
                    if merge:
                        _deep_merge_dicts(o, new_dict)
                    else:
                        obj[key][i] = new_dict
                    break
            else:
                obj[key].append(new_dict)
        elif isinstance(obj[key], dict):
            if obj[key].get("@type") == new_dict.get("@type"):
                if merge:
                    _deep_merge_dicts(obj[key], new_dict)
                else:
                    obj[key] = new_dict
            else:
                obj[key] = [obj[key], new_dict]
        else:
            msg = f"Unexpected type at end of path: {type(obj[key])}"
            raise TypeError(msg)
        return

    # Otherwise, recursively descend
    if key not in obj or obj[key] is None:
        obj[key] = {}

    val = obj[key]

    # If dict, descend directly
    if isinstance(val, dict):
        insert_dict_in_jsonld(val, keys[1:], new_dict, merge=merge)

    # If list, select the dict with the expected type
    elif isinstance(val, list):
        target = None
        if expected_type:
            for el in val:
                if isinstance(el, dict) and el.get("@type") == expected_type:
                    target = el
                    break
        if not target:
            # If no match found, create a new dict with that type
            target = {"@type": expected_type} if expected_type else {}
            val.append(target)
        insert_dict_in_jsonld(target, keys[1:], new_dict, merge=merge)

    else:
        msg = f"Unexpected type in path: {type(val)}"
        raise TypeError(msg)


def merge_battinfo_with_db_data(battinfo_jsonld: dict, sample_data: dict) -> dict:
    """Merge info from the database with BattINFO ontology."""
    coin_cell = find_coin_cell(battinfo_jsonld)
    if coin_cell is None:
        msg = "Could not find CoinCell in JSON-LD"
        raise ValueError(msg)

    # Sample ID and CCID (barcode)
    if sample_data.get("Barcode"):
        coin_cell["schema:productID"] = [
            sample_data["Sample ID"],
            sample_data["Barcode"],
        ]
        logger.info("Added: productID (Sample ID and CCID)")
    else:
        coin_cell["schema:productID"] = sample_data["Sample ID"]
        logger.info("Skipped: productID (CCID)")

    # Date created
    if sample_data.get("Assembly history"):
        assembly_history: list = sample_data["Assembly history"]
        date = None
        for step in reversed(assembly_history):
            if step.get("Timestamp"):
                # It is YYYY-MM-DD hh-mm-ss %z
                # Needs to be YYYY-MM-DD ISO 8061
                date = step["Timestamp"][:10]
                break
        if date:
            coin_cell["schema:dateCreated"] = date
            logger.info("Added: dateCreated")
        else:
            logger.info("Skipped: dateCreated")

        # Cell assembly sequence
        if "rdfs:comment" not in coin_cell:
            coin_cell["rdfs:comment"] = summarise_assembly(assembly_history, sample_data)
            logger.info("Added: rdfs:comment")
        elif not any(s.startswith("Cell assembly sequence: ") for s in coin_cell["rdfs:comment"]):
            coin_cell["rdfs:comment"].append(summarise_assembly(assembly_history, sample_data))
        else:
            logger.info("Skipped: cell assembly - already present")
    else:
        logger.info("Skipped: dateCreated - no assembly history")
        logger.info("Skipped: cell assembly - no assembly history")

    # Electrode mass loading
    for xode in ("Anode", "Cathode"):
        key = "hasPositiveElectrode" if xode == "Cathode" else "hasNegativeElectrode"
        if sample_data.get(f"{xode} active material mass (mg)") and sample_data.get(f"{xode} diameter (mm)"):
            mass_g = sample_data[f"{xode} active material mass (mg)"] / 1000
            area_cm = 3.14159 * (sample_data[f"{xode} diameter (mm)"] / 10) ** 2
            mass_loading_g_per_cm2 = mass_g / area_cm
            mass_dict = {
                "@type": "MassLoading",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": mass_loading_g_per_cm2},
                "hasMeasurementUnit": "unit:MilliGM-PER-CentiM2",
            }
            insert_dict_in_jsonld(
                coin_cell,
                [
                    (key, "Electrode"),
                    ("hasCoating", "Coating"),
                    ("hasActiveMaterial", None),
                    ("hasMeasuredProperty", "MassLoading"),
                ],
                mass_dict,
            )
            logger.info("Added: %s mass loading", xode)
        else:
            logger.info("Skipped: %s mass loading", xode)

        if sample_data.get(f"{xode} diameter (mm)"):
            diameter_dict = {
                "@type": "Diameter",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": sample_data[f"{xode} diameter (mm)"]},
                "hasMeasurementUnit": "unit:MilliM",
            }
            insert_dict_in_jsonld(
                coin_cell,
                [(key, "Electrode"), ("hasMeasuredProperty", "Diameter")],
                diameter_dict,
            )
            logger.info("Added: %s diameter", xode)
        else:
            logger.info("Skipped: %s diameter", xode)

    return battinfo_jsonld


def find_coin_cell(jsonld: dict) -> dict | None:
    """Search for the CoinCell in a dict."""
    if "@type" in jsonld and jsonld["@type"] == "CoinCell":
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

    Anything referencing this type is put in @reversed.
    """
    if isinstance(data, dict) and data.get("@type") == target_type:
        return data
    if isinstance(data, dict) and data.get("@reversed") is not None:
        msg = "Cannot rearrange object if @reversed in json-ld"
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
            if obj.get("@type") == target_type:
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
    result["@reversed"] = {key: reversed_obj}
    if ctx:
        result["@context"] = ctx
    return result


def merge_contexts_strict(ctx1: str | list | dict, ctx2: str | list | dict) -> list:
    """Merge the top level @context blocks of two JSON-LD."""
    ctx1_list = ctx1 if isinstance(ctx1, list) else [ctx1]
    ctx2_list = ctx2 if isinstance(ctx2, list) else [ctx2]

    def process_context_list(ctx_list: list) -> tuple[list, dict]:
        """Process a context list, split terms and remotes."""
        remotes = []
        terms = {}
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

    # Check for conflicts in term definitions
    for term, iri in terms2.items():
        if term in terms1 and terms1[term] != iri:
            msg = f"Conflict for term '{term}': '{terms1[term]}' != '{iri}'"
            raise ValueError(msg)

    # Merge term definitions (terms2 overrides terms1, but no conflicts allowed)
    merged_terms = dict(terms1)
    merged_terms.update(terms2)

    # Construct merged context list
    merged_context = []
    if merged_remotes:
        merged_context.extend(merged_remotes)
    if merged_terms:
        merged_context.append(merged_terms)

    return merged_context


def recursive_merge(left: dict, right: dict) -> dict:
    """Recursively merge dicts."""
    for k, rv in right.items():
        if k not in left:
            left[k] = rv
        else:
            lv = left[k]
            if isinstance(lv, dict) and isinstance(rv, dict):
                left[k] = recursive_merge(lv, rv)
            elif isinstance(lv, list) and isinstance(rv, list):
                left[k] = dedupe_jsonld_list([*lv, *rv])
            elif isinstance(lv, list):
                left[k] = dedupe_jsonld_list([*lv, rv])
            elif isinstance(rv, list):
                left[k] = dedupe_jsonld_list([lv, *rv])
            elif lv != rv:
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


def merge_jsonld(json1: dict, json2: dict) -> dict:
    """Merge two JSON-LD structures assuming they reference the SAME NODE."""
    if not isinstance(json1, dict) or not isinstance(json2, dict):
        raise TypeError
    if "@type" not in json1 or json1.get("@type") != json2.get("@type"):
        msg = "Two JSON-LDs do not have the same parent node"
        raise ValueError(msg)
    return recursive_merge(json1, json2)


def merge_jsonld_on_type(json1: dict, json2: dict, target_type: str = "CoinCell") -> dict:
    """Transform two json-ld, make target_type parent and merge."""
    json1 = make_type_parent(json1, target_type)
    json2 = make_type_parent(json2, target_type)
    return merge_jsonld(json1, json2)


def generate_battery_test(ontologized_protocols: dict | list[dict]) -> dict:
    """Generate test json-ld based on protocols."""
    if isinstance(ontologized_protocols, list) and len(ontologized_protocols) == 1:
        ontologized_protocols = ontologized_protocols[0]
    return {
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
        "@type": "CoinCell",
        "@reversed": {
            "hasTestObject": {
                "@type": "BatteryTest",
                "hasMeasurementParameter": {
                    "@type": ["ConstantCurrentConstantVoltageCycling", "IterativeWorkflow"],
                    "rdfs:label": "GeneratedBatteryTestProcedure",
                    "hasLab": {
                        "@type": "Laboratory",
                        "@id": "https://www.wikidata.org/wiki/Q683116",
                        "rdfs:label": "Empa",
                    },
                    "hasTask": ontologized_protocols,
                },
            }
        },
    }
