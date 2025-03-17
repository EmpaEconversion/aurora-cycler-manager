"""Some utility functions for the Aurora Cycler Manager."""


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
