"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Functions for converting raw tomato json files to aurora-compatible .h5 or
.json.gz files.

convert_tomato_json converts a tomato 0.2.3 json to a dataframe and metadata
dictionary, and optionally saves it as a hdf5 or a gzipped json file. This file
contains all cycling data as well as metadata from the tomato json and sample
information from the database.

convert_all_tomato_jsons does this for all tomato files in the local snapshot
folder.
"""
import gzip
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from aurora_cycler_manager.version import __version__, __url__
from aurora_cycler_manager.analysis import _run_from_sample

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "..", "config.json")
with open(config_path, encoding = "utf-8") as f:
    config = json.load(f)
db_path = config["Database path"]

def convert_tomato_json(
        snapshot_file: str,
        output_hdf_file: bool = True,
        output_jsongz_file: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
    """Convert a raw json file from tomato to a pandas dataframe.

    Args:
        snapshot_file (str): path to the raw json file
        output_hdf_file (str, optional): path to save the output hdf5 file
        output_jsongz_file (str, optional): path to save the output json.gz file

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
    with open(snapshot_file, encoding="utf-8") as f:
        input_dict = json.load(f)
    n_steps = len(input_dict["steps"])
    data = []
    technique_code = {"NONE":0,"OCV":100,"CA":101,"CP":102,"CV":103,"CPLIMIT":155,"CALIMIT":157}
    for i in range(n_steps):
        step_data = input_dict["steps"][i]["data"]
        step_dict = {
            "uts" : [row["uts"] for row in step_data],
            "V (V)" : [row["raw"]["Ewe"]["n"] for row in step_data],
            "I (A)": [row["raw"]["I"]["n"] if "I" in row["raw"] else 0 for row in step_data],
            "cycle_number": [row["raw"].get("cycle number", 0) for row in step_data],
            "loop_number": [row["raw"].get("loop number", 0) for row in step_data],
            "index" : [row["raw"].get("index", -1) for row in step_data],
            "technique": [technique_code.get(row.get("raw", {}).get("technique"), -1) for row in step_data],
        }
        data.append(pd.DataFrame(step_dict))
    data = pd.concat(data, ignore_index=True)

    # Get metadata
    # Try to get the job number from the snapshot file and add to metadata
    json_filename = os.path.basename(snapshot_file)
    jobid = "".join(json_filename.split(".")[1:-1])
    # look up jobid in the database
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Get all data about this job
            cursor.execute("SELECT * FROM jobs WHERE `Job ID`=?", (jobid,))
            job_data = dict(cursor.fetchone())
            job_data["Payload"] = json.loads(job_data["Payload"])
            sampleid = job_data["Sample ID"]
            # Get all data about this sample
            cursor.execute("SELECT * FROM samples WHERE `Sample ID`=?", (sampleid,))
            sample_data = dict(cursor.fetchone())
    except Exception as e:
        print(f"Error getting job and sample data from database: {e}")
        return data, None
    job_data["job_type"] = "tomato_0_2_biologic"
    metadata = {
        "provenance": {
            "snapshot_file": str(snapshot_file),
            "tomato_metadata": input_dict["metadata"],
            "aurora_metadata": {
                "json_conversion": {
                    "repo_url": __url__,
                    "repo_version": __version__,
                    "method": "tomato_converter.py convert_tomato_json",
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "technique": "code of technique using definitions from MPG2 developer package",
        },
    }

    if output_hdf_file or output_jsongz_file:  # Save and update database
        run_id = _run_from_sample(sampleid)
        folder = Path(config["Processed snapshots folder path"]) / run_id / sampleid
        if not folder.exists():
            folder.mkdir(parents=True)

        snapshot_filename = Path(snapshot_file).name

        if output_jsongz_file:
            jsongz_filepath = folder / snapshot_filename.replace(".json", ".json.gz")
            full_data = {"data": data.to_dict(orient="list"), "metadata": metadata}
            with gzip.open(jsongz_filepath, "wt", encoding="utf-8") as f:
                json.dump(full_data, f)

        if output_hdf_file:
            hdf5_filepath = folder / snapshot_filename.replace(".json", ".h5")
            data = data.astype({"V (V)": "float32", "I (A)": "float32"})
            data = data.astype({
                "technique": "int16",
                "cycle_number": "int32",
                "loop_number": "int32",
                "index": "int16",
            })
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

def convert_all_tomato_jsons(
    sampleid_contains: str = "",
    ) -> None:
    """Goes through all the raw json files in the snapshots folder and converts them to hdf5."""
    raw_folder = Path(config["Snapshots folder path"])
    for batch_folder in raw_folder.iterdir():
        for sample_folder in batch_folder.iterdir():
            if sampleid_contains and sampleid_contains not in sample_folder.name:
                continue
            for snapshot_file in sample_folder.iterdir():
                snapshot_filename = snapshot_file.name
                if snapshot_filename.startswith("snapshot") and snapshot_filename.endswith(".json"):
                    convert_tomato_json(
                        snapshot_file,
                        output_hdf_file=True,
                        output_jsongz_file=False,
                    )
                    print(f"Converted {snapshot_file}")
