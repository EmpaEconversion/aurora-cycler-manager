"""BattINFO ontology functions."""

import logging

logger = logging.getLogger(__name__)


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
        if key not in coin_cell and sample_data.get(f"{xode} diameter (mm)"):
            coin_cell[key] = {
                "@type": "Electrode",
                "hasCoating": {
                    "@type": "Coating",
                    "hasActiveMaterial": {
                        "hasMeasuredProperty": [],
                    },
                },
                "hasMeasuredProperty": [],
            }

        if sample_data.get(f"{xode} active material mass (mg)") and sample_data.get(f"{xode} diameter (mm)"):
            mass_g = sample_data[f"{xode} active material mass (mg)"] / 1000
            area_cm = 3.14159 * (sample_data[f"{xode} diameter (mm)"] / 10) ** 2
            mass_loading_g_per_cm2 = mass_g / area_cm
            mass_dict = {
                "@type": "MassLoading",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": mass_loading_g_per_cm2},
                "hasMeasurementUnit": "unit:MilliGM-PER-CentiM2",
            }
            props = coin_cell[key]["hasCoating"]["hasActiveMaterial"]["hasMeasuredProperty"]
            for p in props:
                # Replace existing value
                if p.get("@type") == "MassLoading":
                    p = mass_dict  # noqa: PLW2901
                    break
            else:
                # Add new value
                props.append(mass_dict)
            logger.info("Added: %s mass loading", xode)
        else:
            logger.info("Skipped: %s mass loading", xode)

        if sample_data.get(f"{xode} diameter (mm)"):
            props = coin_cell[key]["hasMeasuredProperty"]
            diameter_dict = {
                "@type": "Diameter",
                "hasNumericalPart": {"@type": "emmo:RealData", "hasNumberValue": sample_data[f"{xode} diameter (mm)"]},
                "hasMeasurementUnit": "unit:MilliM",
            }
            for p in props:
                # Replace existing value
                if p.get("@type") == "Diameter":
                    p = diameter_dict  # noqa: PLW2901
                    break
            else:
                # Add new value
                props.append(diameter_dict)
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
