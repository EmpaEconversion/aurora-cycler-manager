"""Copyright © 2025-2026, Empa.

Mappings for columns names.
"""

aurora_to_bdf_map: dict[str, str] = {
    "uts": "unix_time_second",
    "V (V)": "voltage_volt",
    "I (A)": "current_ampere",
    "Step": "step_count",
    "Cycle": "cycle_count",
    "f (Hz)": "frequency_hertz",
    "Re(Z) (ohm)": "real_impedance_ohm",
    "Im(Z) (ohm)": "imaginary_impedance_ohm",
}

bdf_to_aurora_map_extras: dict[str, str] = {
    "Unix Time / s": "uts",
    "Current / A": "I (A)",
    "Voltage / V": "V (V)",
    "Step Count / 1": "Step",
    "Cycle Count / 1": "Cycle",
    "Freqency / Hz": "f (Hz)",
    "Real Impedance / ohm": "Re(Z) (ohm)",
    "Imaginary Impedance / ohm": "Im(Z) (ohm)",
}

bdf_to_aurora_map: dict[str, str] = {
    **{v: k for k, v in aurora_to_bdf_map.items()},
    **bdf_to_aurora_map_extras,
}
