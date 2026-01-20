"""Copyright © 2025-2026, Empa.

Utility functions which depend on config and/or 3rd party imports.
"""

from contextlib import suppress
from datetime import datetime, timezone

import numpy as np
import paramiko

from aurora_cycler_manager.config import get_config

CONFIG = get_config()


def ssh_connect(ssh: paramiko.SSHClient, username: str, hostname: str) -> None:
    """Connect via SSH."""
    ssh.load_host_keys(CONFIG["SSH known hosts path"])
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    # Paramiko is case sensitive
    ssh.connect(hostname.lower(), username=username.lower(), key_filename=CONFIG.get("SSH private key path"))


def weighted_median(values: list[float] | np.ndarray, weights: list[float] | np.ndarray) -> float:
    """Calculate the weighted median of a list of values.

    Args:
        values: Array-like of values.
        weights: Array-like of weights.

    Returns:
        float: Weighted median of the values.

    """
    if len(values) != len(weights):
        msg = "Values and weights must have the same length."
        raise ValueError(msg)
    if len(values) == 0:
        msg = "Values and weights cannot be empty."
        raise ValueError(msg)
    values = np.array(values)
    weights = np.array(weights)

    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumulative_weights = sorted_weights.cumsum()
    cutoff = cumulative_weights[-1] / 2

    return sorted_values[np.where(cumulative_weights >= cutoff)[0][0]]


def parse_datetime(datetime_str: str | float) -> datetime:
    """Parse a datetime string.

    Could be ISO8601 format, timestamp, %Y-%m-%d %H:%M:%S, %Y-%m-%d %H:%M:%S %z, or %Y-%m-%d %H:%M:%S.%f
    """
    if isinstance(datetime_str, str):
        with suppress(ValueError):
            dt = datetime.fromisoformat(datetime_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CONFIG["tz"])
            return dt.astimezone(timezone.utc)
        with suppress(ValueError):
            return datetime.fromtimestamp(float(datetime_str), tz=timezone.utc)
        with suppress(ValueError):
            # Assume local timezone, convert to UTC
            return (
                datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=CONFIG["tz"])
                .astimezone(timezone.utc)
            )
        with suppress(ValueError):
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S %z")
        with suppress(ValueError):
            # Assume local timezone, convert to UTC
            return (
                datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
                .replace(tzinfo=CONFIG["tz"])
                .astimezone(timezone.utc)
            )
    if isinstance(datetime_str, float):
        return datetime.fromtimestamp(datetime_str, tz=timezone.utc)
    msg = f"Invalid datetime string: {datetime_str}"
    raise ValueError(msg)
