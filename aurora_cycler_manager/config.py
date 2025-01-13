"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Functions for getting the configuration settings.
"""
import json
from pathlib import Path


def get_config() -> dict:
    """Get the configuration data from the user and shared config files.

    Returns:
        dict: dictionary containing the configuration data

    """
    current_dir = Path(__file__).resolve().parent
    user_config_path = current_dir / "config.json"

    # if there is no user config file, create one
    if not user_config_path.exists():
        with user_config_path.open("w", encoding = "utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "Shared config path": "",
                        "SSH private key path" : "",
                        "Snapshots folder path" : "",
                    },
                    indent = 4,
                ),
            )
            msg = f"""
                Please fill in the config file at {user_config_path}.

                REQUIRED:
                'Shared config path': Path to the shared config file on the network drive.

                OPTIONAL - if you want to interact directly with cyclers (e.g. load, eject, submit jobs):
                'SSH private key path': Path to the SSH private key file.
                'Snapshots folder path': Path to a (local) folder to store unprocessed snapshots e.g. 'C:/aurora-shapshots'.
            """
            raise FileNotFoundError(msg)

    with user_config_path.open(encoding = "utf-8") as f:
        config = json.load(f)

    # If there is a shared config file, update with settings from that file
    shared_config_path = config.get("Shared config path")
    if shared_config_path:
        with Path(shared_config_path).open(encoding = "utf-8") as f:
            shared_config = json.load(f)
        config.update(shared_config)

    if not config.get("Database path"):
        msg = f"""
                Please fill in the config file at {user_config_path}.

                REQUIRED:
                'Shared config path': Path to the shared config file on the network drive. If this does not exist, run database_setup.py.

                OPTIONAL - if you want to interact directly with cyclers (e.g. load, eject, submit jobs):
                'SSH private key path': Path to the SSH private key file.
                'Snapshots folder path': Path to a (local) folder to store unprocessed snapshots e.g. 'C:/aurora-shapshots'.
            """
        raise ValueError(msg)

    return config
