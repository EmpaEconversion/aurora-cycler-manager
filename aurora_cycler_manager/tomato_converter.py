"""Copyright © 2025, Empa.

Functions for converting raw tomato json files to aurora-compatible hdf5 files.

convert_tomato_json converts a tomato 0.2.3 json to a dataframe and metadata
dictionary, and optionally saves it as a hdf5 file. This file contains all
cycling data as well as metadata from the tomato json and sample information
from the database.

convert_all_tomato_jsons does this for all tomato files in the local snapshot
folder.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd
import pytz

from aurora_cycler_manager.config import get_config
from aurora_cycler_manager.database_funcs import get_job_data, get_sample_data
from aurora_cycler_manager.utils import run_from_sample
from aurora_cycler_manager.version import __url__, __version__

CONFIG = get_config()
tz = pytz.timezone(CONFIG.get("Time zone", "Europe/Zurich"))
logger = logging.getLogger(__name__)


def get_tomato_snapshot_folder() -> Path:
    """Get the path to the snapshot folder for tomato files."""
    snapshot_parent = CONFIG.get("Snapshots folder path")
    if not snapshot_parent:
        msg = (
            "No 'Snapshots folder path' in config file. "
            f"Please fill in the config file at {CONFIG.get('User config path')}.",
        )
        raise ValueError(msg)
    return Path(snapshot_parent) / "tomato_snapshots"


def puree_tomato(
    snapshot_file_path: Path | str,
) -> None:
    """Reduce raw tomato size by removing unneeded fields and indents.

    Args:
        snapshot_file_path (Path or str): path to the raw json file

    """
    snapshot_file_path = Path(snapshot_file_path)
    with snapshot_file_path.open("r") as f:
        input_dict = json.load(f)
    # Remove unneeded fields
    for step in input_dict.get("steps", []):
        for data in step.get("data", []):
            if "fn" in data:
                del data["fn"]
    # Write the reduced json back to the file
    with snapshot_file_path.open("w") as f:
        json.dump(input_dict, f, indent=None)


def puree_all_tomatos() -> None:
    """Reduce all raw tomato json files in the snapshot folder."""
    snapshot_folder = get_tomato_snapshot_folder()
    for batch_folder in snapshot_folder.iterdir():
        for sample_folder in batch_folder.iterdir():
            for snapshot_file in sample_folder.iterdir():
                snapshot_filename = snapshot_file.name
                if snapshot_filename.startswith("snapshot") and snapshot_filename.endswith(".json"):
                    try:
                        puree_tomato(snapshot_file)
                        logger.info("Pureed %s", snapshot_file)
                    except Exception:
                        logger.exception("Failed to puree %s", snapshot_file)


def convert_tomato_json(
    snapshot_file_path: Path | str,
    output_hdf_file: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Convert a raw json file from tomato to a pandas dataframe.

    Args:
        snapshot_file_path (Path or str): path to the raw json file
        output_hdf_file (str, optional): path to save the output hdf5 file

    Returns:
        pd.DataFrame: DataFrame containing the cycling data

    Columns in output DataFrame:
    - uts: Unix time stamp in seconds
    - V (V): Cell voltage in volts
    - I (A): Current in amps
    - loop_number: how many loops have been completed
    - cycle_number: used if there is a loop of loops
    - index: index of the method in the payload
    - technique: code of technique using Biologic convention
        100 = OCV, 101 = CA, 102 = CP, 103 = CV, 155 = CPLIMIT, 157 = CALIMIT,
        -1 = Unknown

    The dataframe is saved to 'data' key in the hdf5 file.
    Metadata is saved to the 'metadata' key in hdf5 file.
    The metadata includes json dumps of the job data and sample data extracted
    from the database.

    """
    # Extract data from the json file
    snapshot_file_path = Path(snapshot_file_path)
    with snapshot_file_path.open("r") as f:
        input_dict = json.load(f)
    n_steps = len(input_dict["steps"])
    dfs = []
    technique_code = {"NONE": 0, "OCV": 100, "CA": 101, "CP": 102, "CV": 103, "CPLIMIT": 155, "CALIMIT": 157}
    for i in range(n_steps):
        step_data = input_dict["steps"][i]["data"]
        step_dict = {
            "uts": [row["uts"] for row in step_data],
            "V (V)": [row["raw"]["Ewe"]["n"] for row in step_data],
            "I (A)": [row["raw"]["I"]["n"] if "I" in row["raw"] else 0 for row in step_data],
            "cycle_number": [row["raw"].get("cycle number", 0) for row in step_data],
            "loop_number": [row["raw"].get("loop number", 0) for row in step_data],
            "index": [row["raw"].get("index", -1) for row in step_data],
            "technique": [technique_code.get(row.get("raw", {}).get("technique"), -1) for row in step_data],
        }
        dfs.append(pd.DataFrame(step_dict))
    data = pd.concat(dfs, ignore_index=True)

    # Get metadata
    # Try to get the job number from the snapshot file and add to metadata
    json_filename = snapshot_file_path.name
    jobid = "".join(json_filename.split(".")[1:-1])
    # look up jobid in the database
    job_data = get_job_data(jobid)
    sampleid = job_data["Sample ID"]
    sample_data = get_sample_data(sampleid)
    job_data["job_type"] = "tomato_0_2_biologic"
    metadata = {
        "provenance": {
            "snapshot_file": str(snapshot_file_path),
            "tomato_metadata": input_dict["metadata"],
            "aurora_metadata": {
                "json_conversion": {
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "tomato_converter.py convert_tomato_json",
                    "datetime": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %z"),
                },
            },
        },
        "job_data": job_data,
        "sample_data": sample_data,
        "glossary": {
            "uts": "Unix time stamp in seconds",
            "V (V)": "Cell voltage in volts",
            "I (A)": "Current across cell in amps",
            "loop_number": "Number of loops completed from EC-lab loop technique",
            "cycle_number": "Number of cycles within one technique from EC-lab",
            "index": "index of the method in the payload, i.e. 0 for the first method, 1 for the second etc.",
            "technique": "code of technique using definitions from MPG2 developer package, see technique codes",
            "technique codes": {v: k for k, v in technique_code.items()},
        },
    }

    if output_hdf_file:  # Save and update database
        run_id = run_from_sample(sampleid)
        folder = Path(CONFIG["Processed snapshots folder path"]) / run_id / sampleid
        if not folder.exists():
            folder.mkdir(parents=True)
        hdf5_filepath = folder / json_filename.replace(".json", ".h5")
        data = data.astype({"V (V)": "float32", "I (A)": "float32"})
        data = data.astype(
            {
                "technique": "int16",
                "cycle_number": "int32",
                "loop_number": "int32",
                "index": "int16",
            },
        )
        data.to_hdf(
            hdf5_filepath,
            key="data",
            mode="w",
            complib="blosc",
            complevel=9,
        )
        # create a dataset called metadata and json dump the metadata
        with h5py.File(hdf5_filepath, "a") as f:
            f.create_dataset("metadata", data=json.dumps(metadata))
    return data, metadata


def convert_all_tomato_jsons(sampleid_contains: str = "") -> None:
    """Goes through all the raw json files in the snapshots folder and converts them to hdf5."""
    snapshot_folder = get_tomato_snapshot_folder()
    for batch_folder in snapshot_folder.iterdir():
        for sample_folder in batch_folder.iterdir():
            if sampleid_contains and sampleid_contains not in sample_folder.name:
                continue
            for snapshot_file in sample_folder.iterdir():
                snapshot_filename = snapshot_file.name
                if snapshot_filename.startswith("snapshot") and snapshot_filename.endswith(".json"):
                    try:
                        convert_tomato_json(
                            snapshot_file,
                            output_hdf_file=True,
                        )
                        logger.info("Converted %s", snapshot_file)
                    except Exception:
                        logger.exception("Failed to convert %s", snapshot_file)
