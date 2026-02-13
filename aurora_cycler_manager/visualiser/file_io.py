"""Copyright © 2026, Empa.

Functions for file upload and download.
"""

import contextlib
import io
import json
import logging
import uuid
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
from aurora_unicycler import Protocol
from battinfoconverter_backend.json_convert import convert_excel_to_jsonld

import aurora_cycler_manager.battinfo_utils as bu
from aurora_cycler_manager.analysis import analyse_sample
from aurora_cycler_manager.bdf_converter import aurora_to_bdf, bdf_to_aurora
from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.data_bundle import get_cycles_summary, get_cycling, get_metadata, get_sample_folder
from aurora_cycler_manager.database_funcs import (
    add_data_to_db,
    add_protocol_to_job,
    add_samples_from_object,
    get_all_sampleids,
    get_sample_data,
    get_unicycler_protocols,
)
from aurora_cycler_manager.eclab_harvester import convert_mpr
from aurora_cycler_manager.server_manager import _Sample
from aurora_cycler_manager.stdlib_utils import run_from_sample
from aurora_cycler_manager.visualiser.notifications import (
    error_notification,
    success_notification,
)

logger = logging.getLogger(__name__)
CONFIG = get_config()


def is_samples_json(obj: list | str | dict) -> bool:
    """Check if object is a samples file."""
    return isinstance(obj, list) and all(isinstance(s, dict) for s in obj) and all(s.get("Sample ID") for s in obj)


def is_battinfo_jsonld(obj: list | str | dict) -> bool:
    """Check if object is battinfo JSON-LD."""
    if isinstance(obj, dict) and obj.get("@context"):
        coincell = bu.find_coin_cell(obj)
        if coincell:
            comments = coincell.get("rdfs:comment")
            return isinstance(comments, list) and len(comments) >= 1 and comments[0].startswith("BattINFO")
    return False


def is_aux_jsonld(obj: list | str | dict) -> bool:
    """Check if object is a generic JSON-LD."""
    return isinstance(obj, dict) and bool(obj.get("@context")) and (not is_battinfo_jsonld(obj))


def is_unicycler_protocol(obj: list | str | dict) -> bool:
    """Check if object is a unicycler protocol."""
    return isinstance(obj, dict) and bool(obj.get("unicycler"))


def determine_file(filepath: str | Path, selected_rows: list) -> tuple[str, str, bool, dict]:
    """Determine what the uploaded file should do."""
    logger.info("Determining upload")
    filepath = Path(filepath)
    if not filepath or not filepath.exists():
        return "Nothing uploaded", "grey", True, {"file": None, "data": None}

    if filepath.suffix in {".jsonld", ".json"}:
        # It could be ontology or samples
        with filepath.open("r") as f:
            data = json.load(f)
        if is_samples_json(data):
            samples = [d.get("Sample ID") for d in data]
            known_samples = set(get_all_sampleids())
            overwriting_samples = [s for s in samples if s in known_samples]
            if overwriting_samples:
                return (
                    f"Got a samples json\nContains{len(samples)} samples\n"
                    f"WARNING - it will overwrite {len(overwriting_samples)} samples:\n"
                    + "\n".join(overwriting_samples),
                    "orange",
                    False,
                    {"file": "samples-json", "data": data},
                )
            return (
                f"Got a samples json\nContains {len(samples)} samples",
                "green",
                False,
                {"file": "samples-json", "data": data},
            )

        samples = [s.get("Sample ID") for s in selected_rows if s.get("Sample ID")]
        if is_battinfo_jsonld(data):
            if not samples:
                return (
                    "Got a BattINFO json-ld, but you must select samples.",
                    "red",
                    True,
                    {"file": None, "data": None},
                )
            return (
                "Got a BattINFO json-ld\n"
                "The metadata will be merged with info from the database\n"
                f"It will be applied to {len(samples)} samples:\n" + "\n".join(samples),
                "green",
                False,
                {"file": "battinfo-jsonld", "data": data},
            )
        if is_aux_jsonld(data):
            if not samples:
                return (
                    "Got an auxiliary json-ld, but you must select samples.",
                    "red",
                    True,
                    {"file": None, "data": None},
                )
            return (
                "Got an auxiliary json-ld\n"
                "Each sample can have one auxiliary file that is merged when outputting\n"
                f"I will apply it to {len(samples)} samples:\n" + "\n".join(samples),
                "green",
                False,
                {"file": "aux-jsonld", "data": data},
            )
        if is_unicycler_protocol(data):
            jobs = [s.get("Job ID") for s in selected_rows if s.get("Job ID")]
            if not jobs:
                return (
                    "Got a unicycler protocol, but you must select jobs.",
                    "red",
                    True,
                    {"file": None, "data": None},
                )
            protocols = [s.get("Unicycler protocol") for s in selected_rows if s.get("Unicycler protocol")]
            if protocols:
                return (
                    "Got a unicycler protocol.\nWARNING - this will overwrite data",
                    "orange",
                    False,
                    {"file": "unicycler-json", "data": data},
                )
            return (
                "Got a unicycler protocol.",
                "green",
                False,
                {"file": "unicycler-json", "data": data},
            )

    elif filepath.suffix == ".xlsx":
        # It is probably a battinfo xlsx file
        excel_file = pd.ExcelFile(filepath)
        sheet_names = [str(s) for s in excel_file.sheet_names]
        expected_sheets = ["Schema", "@context-TopLevel", "@context-Connector", "Ontology - Unit", "Unique ID"]
        if not all(sheet in expected_sheets for sheet in sheet_names):
            return (
                "Excel file does not have the expected sheets"
                "Found: " + ", ".join(sheet_names) + "\n"
                "Expected: " + ", ".join(expected_sheets),
                "red",
                True,
                {"file": None, "data": None},
            )
        sample_ids = [s.get("Sample ID") for s in selected_rows]
        if not sample_ids:
            return "Got a BattINFO xlsx, but you must select samples.", "red", True, {"file": None, "data": None}
        return (
            "Got a BattINFO xlsx\n"
            "The metadata will be merged with info from the database\n"
            f"It will be applied to {len(sample_ids)} samples:\n" + "\n".join(sample_ids),
            "green",
            False,
            {"file": "battinfo-xlsx", "data": None},  # Don't copy the content_string
        )

    elif filepath.suffix == ".zip":
        # Open the zip archive
        with zipfile.ZipFile(filepath, "r") as zip_file:
            # List the contents
            valid_files = {}
            warning_files = {}
            invalid_files = {}
            new_samples = set()
            file_list = zip_file.namelist()
            known_samples = set(get_all_sampleids())
            for file in file_list:
                logger.info("Checking file %s", file)
                if file.endswith("/"):  # Just a folder
                    continue
                if file == "ro-crate-metadata.json":  # silently ignore
                    continue
                parts = file.split("/")
                if len(parts) < 2:
                    invalid_files[file] = "File must be inside a Sample ID folder"
                    continue
                sample_id = parts[-2]

                # Its a file, check if it is supported type
                filetype = file.split(".")[-1]
                if filetype == "mpl":  # silently ignore - sidecar file
                    continue
                if filetype in {"mpr", "xlsx", "parquet"}:
                    if sample_id in known_samples:
                        valid_files[file] = sample_id
                    else:
                        warning_files[file] = sample_id
                        new_samples.add(sample_id)
                    continue
                invalid_files[file] = f"Filetype {filetype} is not supported"

        if valid_files or warning_files:
            msg = "Got a zip with valid format\n"
            color = "green"
            if valid_files:
                msg = "Found data for the following EXISTING samples:\n" + "\n".join(sorted(set(valid_files.values())))
            if warning_files:
                color = "orange"
                msg += "Found data for the following NEW samples:\n" + "\n".join(sorted(set(warning_files.values())))
            if invalid_files:
                color = "orange"
                msg += "\n\nSKIPPING the following files:\n" + "\n".join(
                    file + "\n" + reason for file, reason in invalid_files.items()
                )
            return (
                msg,
                color,
                False,
                {"file": "zip", "data": {**valid_files, **warning_files}, "new_samples": list(new_samples)},
            )
        if invalid_files:
            msg = "No valid files found:\n" + "\n".join(file + "\n" + reason for file, reason in invalid_files.items())
            return msg, "red", True, {"file": None, "data": None, "new_samples": None}
        return "No files found in zip", "red", False, {"file": None, "data": None, "new_samples": None}

    return "File not understood", "red", True, {"file": None, "data": None, "new_samples": None}


def save_battinfo(data: dict, file: str | Path | io.BytesIO, sample_ids: list[str]) -> None:
    """Convert BattINFO to jsonld and save in sample folder."""
    battinfo_jsonld = (
        convert_excel_to_jsonld(file, debug_mode=False) if data["file"] == "battinfo-xlsx" else data["data"]
    )
    # Merge json with database info and save
    for s in sample_ids:
        sample_data = get_sample_data(s)
        merged_jsonld = bu.merge_battinfo_with_db_data(battinfo_jsonld, sample_data, allow_empty_battinfo=True)
        run_id = run_from_sample(s)
        save_path = CONFIG["Processed snapshots folder path"] / run_id / s / f"battinfo.{s}.jsonld"
        logger.info("Saving battinfo json-ld file to %s", save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(merged_jsonld, f, indent=4)


def save_parquet(file: str | Path | bytes, sample_id: str, file_stem: str | None = None) -> None:
    """Normalize a parquet and save as a snapshot."""
    df = pl.read_parquet(file)
    metadata = pl.read_parquet_metadata(file)
    metadata.pop("ARROW:schema", None)
    if "voltage_volt" in df.columns or "Voltage / V" in df.columns:  # bdf
        df = bdf_to_aurora(df)
    if not all(c in df.columns for c in ["uts", "V (V)", "I (A)", "Cycle"]):
        if "Discharge capacity (mAh)" in df.columns:
            return  # Silent - this is recalculated after
        logger.warning("Dataframe not time-series, not saving")
        return
    if file_stem is None:
        file_stem = Path(file).stem if isinstance(file, str | Path) else str(uuid.uuid4())
    job_id = add_data_to_db(sample_id, file_stem, df["uts"][0], df["uts"][-1])
    add_data_to_db(sample_id, file_stem, df["uts"][0], df["uts"][-1], job_id)
    folder = get_sample_folder(sample_id) / "snapshots"
    if not folder.exists():
        folder.mkdir(parents=True)
    if (aurora_metadata := metadata.get("AURORA:metadata")) and (
        sample_data := json.loads(aurora_metadata).get("sample_data")
    ):
        data_sample_id = sample_data.get("Sample ID")
        if data_sample_id == sample_id:
            add_samples_from_object([sample_data], overwrite=True)
        else:
            logger.warning("Sample ID in metadata does not match expected: %s vs %s", data_sample_id, sample_id)
    df.write_parquet(folder / f"snapshot.{file_stem}.parquet", metadata=metadata)


def process_file(data: dict, filepath: str | Path, selected_rows: list) -> int:
    """Process the uploaded file."""
    filepath = Path(filepath)
    if not data.get("file"):
        msg = "No file type provided"
        raise ValueError(msg)
    if not filepath or not filepath.exists():
        msg = "No file provided"
        raise ValueError(msg)
    match data.get("file"):
        case "samples-json":
            logger.info("Adding samples from file")
            samples = data["data"]
            try:
                logger.info("Adding samples %s", ", ".join(s.get("Sample ID") for s in samples))
                add_samples_from_object(samples, overwrite=True)
            except Exception as e:
                logger.exception("Error adding samples")
                error_notification(
                    "Error adding samples",
                    f"{e!s}",
                    queue=True,
                )
                return 0
            success_notification(
                "Samples added",
                f"{len(samples)} added to database",
                queue=True,
            )
            return 1

        case "battinfo-jsonld" | "battinfo-xlsx":
            try:
                sample_ids = [s.get("Sample ID") for s in selected_rows if s.get("Sample ID")]
                save_battinfo(data, filepath, sample_ids)
                success_notification(
                    "BattINFO json-ld uploaded",
                    f"JSON-LD merged with data from {len(sample_ids)} samples",
                    queue=True,
                )
            except Exception as e:
                logger.exception("Failed to convert, merge, save BattINFO json-ld")
                error_notification(
                    "Error saving BattINFO json-ld",
                    f"{e!s}",
                    queue=True,
                )
            return 1

        case "aux-jsonld":
            # No need to convert or add anything, just save the file
            try:
                aux_jsonld = data["data"]
                samples = [s.get("Sample ID") for s in selected_rows if s.get("Sample ID")]
                for s in samples:
                    run_id = run_from_sample(s)
                    save_path = CONFIG["Processed snapshots folder path"] / run_id / s / f"aux.{s}.jsonld"
                    logger.info("Saving auxiliary json-ld to %s", save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    with save_path.open("w", encoding="utf-8") as f:
                        json.dump(aux_jsonld, f, indent=4)
                success_notification(
                    "Aux json-ld uploaded",
                    f"Aux JSON-LD added to {len(samples)} samples",
                    queue=True,
                )
            except Exception as e:
                logger.exception("Failed to upload auxiliary json-ld")
                error_notification(
                    "Error saving aux json-ld",
                    f"{e!s}",
                    queue=True,
                )
            return 1

        case "unicycler-json":
            try:
                protocol = data["data"]
                Protocol.from_dict(protocol)
                jobs = [s.get("Job ID") for s in selected_rows if s.get("Job ID")]
                for job in jobs:
                    add_protocol_to_job(job, protocol)
                success_notification(
                    "Protocols added",
                    f"Protocols added to {len(jobs)} jobs",
                    queue=True,
                )
            except (ValueError, AttributeError, TypeError) as e:
                logger.exception("Error processing and uploading unicycler protocol")
                error_notification(
                    "Error adding protocol",
                    f"{e}",
                    queue=True,
                )
            return 1

        case "zip":
            # Add new samples if required
            if new_samples := data.get("new_samples"):
                add_samples_from_object([{"Sample ID": s} for s in new_samples])
                logger.info("Added new samples: %s", ",".join(new_samples))
            valid_files = data["data"]
            successful_samples = set()
            with zipfile.ZipFile(filepath, "r") as zip_file:
                for subfilepath, sample_id in valid_files.items():
                    filename = subfilepath.split("/")[-1]
                    try:
                        with zip_file.open(subfilepath) as file:
                            logger.info("Processing file: %s", filename)
                            match filename.split(".")[-1]:
                                case "mpr":
                                    # Check if there is an associated mpl file
                                    mpl_filename = subfilepath.replace(".mpr", ".mpl")
                                    if mpl_filename in zip_file.namelist():
                                        with zip_file.open(mpl_filename) as f:
                                            mpl_file = f.read()
                                    else:
                                        mpl_file = None
                                    # Convert mpr file and save
                                    convert_mpr(
                                        file.read(),
                                        mpl_file=mpl_file,
                                        update_database=True,
                                        sample_id=sample_id,
                                        file_name=filename,
                                    )
                                    successful_samples.add(sample_id)
                                case "xlsx":
                                    xlsx_bytes = io.BytesIO(file.read())
                                    save_battinfo({"file": "battinfo-xlsx"}, xlsx_bytes, [sample_id])
                                case "parquet":
                                    save_parquet(file.read(), sample_id, filename.rsplit(".", 1)[0])
                                    successful_samples.add(sample_id)
                    except Exception as e:
                        logger.exception("Error processing file: %s", filename)
                        error_notification(
                            "Error processing file",
                            f"{filename}: {e!s}",
                            queue=True,
                        )
                success_notification(
                    "Complete",
                    "All files processed",
                    queue=True,
                )

            for sample_id in successful_samples:
                logger.info("Analysing sample: %s", sample_id)
                analyse_sample(sample_id)
                success_notification("Sample analysed", f"{sample_id}", queue=True)
            success_notification(
                "All data processed and analysed",
                f"Files added for {len(successful_samples)} samples",
                queue=True,
            )
            return 1

        case _:
            error_notification("Oh no", f"Could not understand filetype {data}")
            return 0


def create_rocrate(
    sample_ids: list,
    filetypes: set,
    zip_path: str | Path,
    zenodo_info: str | None = None,
    set_progress: Callable | None = None,
) -> None:
    """Create ro-crate zip with multiple samples.

    Args:
        sample_ids: List of sample IDs to add to ro-crate.
        filetypes: set of file types, subset of:
            {"bdf-csv","bdf-parquet","cycles-json","metadata-jsonld"}
        zip_path: Path of where to save zip.
        zenodo_info (optional): Path to zenodo info xlsx file.
        set_progress (used by app): Function for progress bar.

    """
    zip_path = Path(zip_path)
    samples = [_Sample.from_id(s) for s in sample_ids]
    rocrate = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {
                "@type": "CreativeWork",
                "@id": "ro-crate-metadata.json",
                "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                "about": {"@id": "./"},
            },
            {
                "@id": "./",
                "@type": "Dataset",
                "name": "Aurora Battery Assembly & Cycling Experiments",
                "description": (
                    "A collection of battery assembly and cycling experiments. "
                    "Data processing, analysis, export, and ro-crate generation completed with "
                    "aurora-cycler-manager (https://github.com/empaeconversion/aurora-cycler-manager)"
                ),
                "dateCreated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hasPart": [],
            },
        ],
    }

    pub_info = bu.parse_zenodo_info_xlsx(zenodo_info) if zenodo_info else {}

    # Number of files
    n_files = len(samples) * len(filetypes)
    i = 0
    if set_progress:
        set_progress((100 * i / n_files, "Initializing...", "Grey"))
    messages = ""
    color = "green"
    # Create a new zip archive to populate
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sample in samples:
            sample_id: str = sample.get("Sample ID")
            ccid: str = sample.get("Barcode")
            run_id = run_from_sample(sample_id)
            data_folder = CONFIG["Processed snapshots folder path"]
            sample_folder = str(data_folder / run_id / sample_id)
            battinfo_files = []
            messages += f"{sample_id} - "
            warnings = []
            errors = []
            df = None
            metadata = None

            # If bdf is requested, convert the file
            if {"bdf-csv", "bdf-parquet"} & filetypes:
                with contextlib.suppress(FileNotFoundError):
                    df = aurora_to_bdf(get_cycling(sample_id))
                    metadata = get_metadata(sample_id)
            # Loop through requested files
            for filetype in filetypes:
                i += 1
                try:
                    if filetype == "cycles-csv":
                        cycles_df = get_cycles_summary(sample_id)
                        if cycles_df is None:
                            logger.warning("No cycles summary file found for %s", sample_id)
                            messages += "⚠️"
                            warnings.append(filetype)
                            color = "orange"
                            continue
                        buffer = io.BytesIO()
                        cycles_df.write_csv(buffer)
                        buffer.seek(0)
                        rel_file_path = sample_id + f"/cycles.{sample_id}.csv"
                        zf.writestr(rel_file_path, buffer.read())
                        messages += "✅"

                        rocrate["@graph"][1]["hasPart"].append({"@id": rel_file_path})
                        rocrate["@graph"].append(
                            {
                                "@id": rel_file_path,
                                "@type": "File",
                                "encodingFormat": "text/csv",
                                "about": {"@id": ccid or sample_id},
                                "description": (
                                    f"Summary data from battery cycling for sample: '{sample_id}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "File is in CSV format, and per-cycle summary stastics, e.g. discharge capacity."
                                ),
                            }
                        )
                        battinfo_files.append(bu.add_data(rel_file_path, pub_info.get("zenodo_doi_url")))

                    if filetype == "cycles-parquet":
                        cycles_df = get_cycles_summary(sample_id)
                        if cycles_df is None:
                            logger.warning("No cycles summary file found for %s", sample_id)
                            messages += "⚠️"
                            warnings.append(filetype)
                            color = "orange"
                            continue
                        buffer = io.BytesIO()
                        cycles_df.write_parquet(buffer)
                        buffer.seek(0)
                        rel_file_path = sample_id + f"/cycles.{sample_id}.parquet"
                        zf.writestr(rel_file_path, buffer.read())
                        messages += "✅"

                        rocrate["@graph"][1]["hasPart"].append({"@id": rel_file_path})
                        rocrate["@graph"].append(
                            {
                                "@id": rel_file_path,
                                "@type": "File",
                                "encodingFormat": "text/csv",
                                "about": {"@id": ccid or sample_id},
                                "description": (
                                    f"Summary data from battery cycling for sample: '{sample_id}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "File is in CSV format, and per-cycle summary stastics, e.g. discharge capacity."
                                ),
                            }
                        )
                        battinfo_files.append(bu.add_data(rel_file_path, pub_info.get("zenodo_doi_url")))

                    if filetype == "bdf-csv":
                        if df is None:
                            logger.warning("No time-series data for %s", sample_id)
                            messages += "⚠️"
                            warnings.append(filetype)
                            color = "orange"
                            continue
                        # convert to csv file and write to zip
                        buffer = io.BytesIO()
                        df.write_csv(buffer)
                        buffer.seek(0)
                        rel_file_path = sample_id + f"/full.{sample_id}.bdf.csv"
                        zf.writestr(rel_file_path, buffer.read())
                        messages += "✅"
                        rocrate["@graph"][1]["hasPart"].append({"@id": rel_file_path})
                        rocrate["@graph"].append(
                            {
                                "@id": rel_file_path,
                                "@type": "File",
                                "encodingFormat": "text/csv",
                                "about": {"@id": ccid or sample_id},
                                "description": (
                                    f"Time-series battery cycling data for sample: '{sample_id}'"
                                    + (f" with barcode: '{ccid}'. " if ccid else ". ")
                                    + "Data is csv format, columns are 'battery data format' (BDF) compliant."
                                ),
                            }
                        )
                        battinfo_files.append(bu.add_data(rel_file_path, pub_info.get("zenodo_doi_url")))

                    if filetype == "bdf-parquet":
                        if df is None:
                            logger.warning("No time-series data for %s", sample_id)
                            messages += "⚠️"
                            warnings.append(filetype)
                            color = "orange"
                            continue
                        # convert to parquet file and write to zip
                        buffer = io.BytesIO()
                        df.write_parquet(
                            buffer,
                            metadata={"AURORA:metadata": json.dumps(metadata)} if metadata else None,
                        )
                        buffer.seek(0)
                        parquet_name = f"full.{sample_id}.bdf.parquet"
                        rel_file_path = sample_id + "/" + parquet_name
                        zf.writestr(rel_file_path, buffer.read())
                        messages += "✅"
                        rocrate["@graph"][1]["hasPart"].append({"@id": rel_file_path})
                        rocrate["@graph"].append(
                            {
                                "@id": rel_file_path,
                                "@type": "File",
                                "encodingFormat": "application/vnd.apache.parquet",
                                "about": {"@id": ccid or sample_id},
                                "description": (
                                    f"Time-series battery cycling data for sample: '{sample_id}'"
                                    + (f" with barcode: '{ccid}'. " if ccid else ". ")
                                    + "Data is parquet format, columns are 'battery data format' (BDF) compliant."
                                ),
                            }
                        )
                        battinfo_files.append(bu.add_data(rel_file_path, pub_info.get("zenodo_doi_url")))

                    if filetype == "metadata-jsonld":
                        # Get the BattINFO file
                        battinfo_file = next(Path(sample_folder).glob("battinfo.*.jsonld"), None)
                        aux_file = next(Path(sample_folder).glob("aux.*.jsonld"), None)
                        if battinfo_file is None:
                            logger.warning("No BattINFO file for %s", sample_id)
                            sample_data = get_sample_data(sample_id)
                            battinfo_json = bu.merge_battinfo_with_db_data({}, sample_data, allow_empty_battinfo=True)
                        else:
                            with battinfo_file.open("r") as f:
                                battinfo_json = json.load(f)
                        battinfo_json = bu.make_test_object(battinfo_json)

                        # Check for auxiliary jsonld file
                        if aux_file:
                            with aux_file.open("r") as f:
                                aux_json = json.load(f)
                            try:
                                bu.merge_jsonld_on_type([battinfo_json, aux_json])
                            except ValueError:
                                bu.merge_jsonld_on_type(
                                    [battinfo_json["hasTestObject"], aux_json],
                                    target_type="CoinCell",
                                )

                        # Add unicycler protocols
                        db_jobs = get_unicycler_protocols(sample_id)
                        if db_jobs:
                            ontologized_protocols = []
                            for db_job in db_jobs:
                                protocol = Protocol.from_dict(json.loads(db_job["Unicycler protocol"]))
                                ontologized_protocols.append(
                                    protocol.to_battinfo_jsonld(capacity_mAh=db_job["Capacity (mAh)"])
                                )
                            test_jsonld = bu.generate_battery_test(ontologized_protocols)
                            battinfo_json = bu.merge_jsonld_on_type([battinfo_json, test_jsonld])

                        # Add data files
                        if battinfo_files:
                            battinfo_json = bu.merge_jsonld_on_type([battinfo_json, *battinfo_files])

                        # Add ccid to output
                        if ccid:
                            battinfo_json = bu.merge_jsonld_on_type([battinfo_json, bu.add_ccid_output(ccid)])

                        # Add citation string
                        if pub_info.get("citation_string"):
                            battinfo_json = bu.merge_jsonld_on_type(
                                [
                                    battinfo_json,
                                    bu.add_citation(pub_info["citation_string"]),
                                ]
                            )

                        # Add authors and institutions
                        if pub_info.get("authors") and pub_info.get("institutions"):
                            authors = bu.add_authors(pub_info["authors"], pub_info["institutions"])
                            battinfo_json = bu.merge_jsonld_on_type([battinfo_json, authors])

                        # Add publication info
                        if (
                            pub_info.get("publication_doi_url")
                            and pub_info.get("sample_to_fig")
                            and (ccid or sample_id)
                        ):
                            publication_extras = bu.add_associated_media(
                                pub_info["publication_doi_url"],
                                pub_info["sample_to_fig"],
                                ccid,
                                sample_id,
                            )
                            battinfo_json = bu.merge_jsonld_on_type([battinfo_json, publication_extras])

                        # Save the JSON-LD
                        jsonld_name = f"metadata.{sample_id}.jsonld"
                        rel_file_path = sample_id + "/" + jsonld_name
                        zf.writestr(rel_file_path, json.dumps(battinfo_json, indent=4))
                        messages += "✅"
                        rocrate["@graph"][1]["hasPart"].append({"@id": rel_file_path})
                        rocrate["@graph"].append(
                            {
                                "@id": rel_file_path,
                                "@type": "File",
                                "encodingFormat": "text/json",
                                "about": {"@id": ccid or sample_id},
                                "description": (
                                    f"Metadata for sample: '{sample_id}'"
                                    + (f" with CCID: '{ccid}'. " if ccid else ". ")
                                    + "File is a BattINFO JSON-LD, describing the sample and experiment."
                                ),
                            }
                        )
                except Exception:
                    logger.exception("Unexpected error processing %s %s", sample_id, filetype)
                    messages += "⁉️"
                    errors.append(filetype)
                    color = "orange"
                finally:
                    if set_progress:
                        set_progress((100 * i / n_files, messages, color))

            # After converting all files for a sample, give a summary message
            messages += "\n"
            if warnings:
                messages += "No data for " + ", ".join(warnings) + "\n"
            if errors:
                messages += "Error processing " + ", ".join(errors) + "\n"
            if set_progress:
                set_progress((100 * i / n_files, messages, color))

        # After converting all files for all samples, check if zipfile contains anything
        if zf.filelist:
            zf.writestr("ro-crate-metadata.json", json.dumps(rocrate, indent=4))
            logger.info("Saved zip file in server temp folder: %s", zip_path.name)
            messages += "\n✅ ZIP file ready to download"
            if set_progress:
                set_progress((100 * i / n_files, messages, color))
        else:
            messages += "\n❌ No ZIP file created"
            if set_progress:
                set_progress((100 * i / n_files, messages, "red"))
            msg = "Zip has no content"
            raise ValueError(msg)
