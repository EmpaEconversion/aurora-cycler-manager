
import json
import sqlite3

from aurora_cycler_manager.config import get_config

config = get_config()

def get_job_data(job_id: str) -> dict:
    """Get all data about a job from the database."""
    with sqlite3.connect(config["Database path"]) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE `Job ID`=?", (job_id,))
        result = cursor.fetchone()
        if not result:
            msg = f"Job ID '{job_id}' not found in the database"
            raise ValueError(msg)
        job_data = dict(result)
        # Convert json strings to python objects
        payload = job_data.get("Payload")
        if payload:
            job_data["Payload"] = json.loads(payload)
    return job_data

def get_sample_data(sample_id: str) -> dict:
    """Get all data about a sample from the database."""
    with sqlite3.connect(config["Database path"]) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE `Sample ID`=?", (sample_id,))
        result = cursor.fetchone()
        if not result:
            msg = f"Sample ID '{sample_id}' not found in the database"
            raise ValueError(msg)
        sample_data = dict(result)
        # Convert json strings to python objects
        history = sample_data.get("Assembly history")
        if history:
            sample_data["Assembly history"] = json.loads(history)
    return sample_data

if __name__ == "__main__":
    sample_data = get_job_data("tt1-23")
    print(sample_data)