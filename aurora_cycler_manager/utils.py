"""Copyright © 2025, Empa.

Utility functions for the Aurora Cycler Manager.
"""

import json
import uuid
from fractions import Fraction
from io import TextIOWrapper

import numpy as np


def run_from_sample(sampleid: str) -> str:
    """Get the run_id from a sample_id."""
    if not isinstance(sampleid, str) or len(sampleid.split("_")) < 2 or not sampleid.split("_")[-1].isdigit():
        return "misc"
    return sampleid.rsplit("_", 1)[0]


def c_to_float(c_rate: str) -> float:
    """Convert a C-rate string to a float.

    Args:
        c_rate (str): C-rate string, e.g. 'C/2', '0.5C', '3D/5', '1/2 D'
    Returns:
        float: C-rate as a float

    """
    if "C" in c_rate:
        sign = 1
    elif "D" in c_rate:
        c_rate = c_rate.replace("D", "C")
        sign = -1
    else:
        msg = f"Invalid C-rate: {c_rate}"
        raise ValueError(msg)

    num, _, denom = c_rate.partition("C")
    number = f"{num}{denom}".strip()

    if "/" in number:
        num, denom = number.split("/")
        if not num:
            num = "1"
        if not denom:
            denom = "1"
        return sign * float(num) / float(denom)
    return sign * float(number)


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


def max_with_none(values: list[float | None]) -> float | None:
    """Get the maximum value from a list of values, allowing for None values."""
    if len(values) == 0:
        return None
    return max([v for v in values if v is not None], default=None)


def min_with_none(values: list[float | None]) -> float | None:
    """Get the minimum value from a list of values, allowing for None values."""
    if len(values) == 0:
        return None
    return min([v for v in values if v is not None], default=None)


def json_dumps_compress_lists(o: list | dict | str | float, indent: int = 4) -> str:
    """JSON dumps, but don't indent lists of numbers."""
    lists_dict = {}

    def replace_lists_with_uuid(o: list | dict | str | float) -> list | dict | str | float:
        """Recursively replace lists with UUIDs."""
        if isinstance(o, list):
            if all(isinstance(i, (int, float)) for i in o):
                key = str(uuid.uuid4())
                lists_dict[key] = o
                return key
            return [replace_lists_with_uuid(i) for i in o]
        if isinstance(o, dict):
            return {k: replace_lists_with_uuid(v) for k, v in o.items()}
        return o

    # replace numerical lists with UUIDs
    o = replace_lists_with_uuid(o)
    # JSON dumps to string
    o_str = json.dumps(o, indent=indent)
    # Replace UUIDs in the string with non-indented JSON dumps of the lists
    for k, v in lists_dict.items():
        o_str = o_str.replace(f'"{k}"', json.dumps(v, indent=None))
    return o_str


def json_dump_compress_lists(o: list | dict | str | float, f: TextIOWrapper, indent: int = 4) -> None:
    """JSON dump, but don't indent lists of numbers."""
    f.write(json_dumps_compress_lists(o, indent=indent))


def round_c_rate(x: float, round_to: int, max_denominator: int = 100, tolerance: float = 0.03) -> float:
    """Round float to nearest fraction if within tolerance."""
    frac = Fraction(x).limit_denominator(max_denominator)
    if abs(float(frac) - x) <= tolerance:
        return round(float(frac), round_to)
    return round(x, round_to)
