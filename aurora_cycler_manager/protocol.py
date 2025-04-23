"""A universal cycling Protocol model to convert to different formats.

Protocol is a Pydantic model that defines a cycling protocol which can be stored/read in JSON format.
The model only contains a subset of all possible techniques and parameters.
This can be converted into a Neware XML file, Tomato JSON file, or PyBaMM list of strings with
to_neware_xml(), to_tomato_mpg2() and to_pybamm_experiment() methods respectively.
"""

import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Literal, Union
from xml.dom import minidom

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

getcontext().prec = 10


class SampleParams(BaseModel):
    """Sample parameters."""

    name: str
    capacity_mAh: Decimal = Field(gt=0)


class MeasurementParams(BaseModel):
    """Measurement parameters, i.e. when to record."""

    current_mA: Decimal | None = None
    voltage_V: Decimal | None = None
    time_s: Decimal = Field(gt=0)


class SafetyParams(BaseModel):
    """Safety parameters, i.e. limits before cancelling measurement."""

    max_voltage_V: Decimal | None = None
    min_voltage_V: Decimal | None = None
    max_current_mA: Decimal | None = None
    min_current_mA: Decimal | None = None
    max_capacity_mAh: Decimal | None = None
    delay_s: Decimal = Field(ge=0)


class BaseTechnique(BaseModel):
    """Base class for all techniques."""


class OpenCircuitVoltage(BaseTechnique):
    """Open circuit voltage technique."""

    name: Literal["open_circuit_voltage"] = "open_circuit_voltage"
    until_time_s: Decimal | None = Field(gt=0)


class ConstantCurrent(BaseTechnique):
    """Constant current technique."""

    name: Literal["constant_current"] = "constant_current"
    rate_C: Decimal | None = None
    current_mA: Decimal | None = None
    until_time_s: Decimal | None = None
    until_voltage_V: Decimal | None = None

    @model_validator(mode="after")
    def ensure_rate_or_current(self) -> Self:
        """Ensure at least one of rate_C or current_mA is set."""
        has_rate_C = self.rate_C is not None and self.rate_C != 0
        has_current_mA = self.current_mA is not None and self.current_mA != 0
        if not (has_rate_C or has_current_mA):
            msg = "Either rate_C or current_mA must be set and non-zero."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def ensure_stop_condition(self) -> Self:
        """Ensure at least one stop condition is set."""
        has_time_s = self.until_time_s is not None and self.until_time_s != 0
        has_voltage_V = self.until_voltage_V is not None and self.until_voltage_V != 0
        if not (has_time_s or has_voltage_V):
            msg = "Either until_time_s or until_voltage_V must be set and non-zero."
            raise ValueError(msg)
        return self


class ConstantVoltage(BaseTechnique):
    """Constant voltage technique."""

    name: Literal["constant_voltage"] = "constant_voltage"
    voltage_V: Decimal
    until_time_s: Decimal | None = None
    until_rate_C: Decimal | None = None
    until_current_mA: Decimal | None = None

    @model_validator(mode="after")
    def check_stop_condition(self) -> Self:
        """Ensure at least one of until_rate_C or until_current_mA is set."""
        has_time_s = self.until_time_s is not None and self.until_time_s != 0
        has_rate_C = self.until_rate_C is not None and self.until_rate_C != 0
        has_current_mA = self.until_current_mA is not None and self.until_current_mA != 0
        print(has_time_s, has_rate_C, has_current_mA)
        if not (has_time_s or has_rate_C or has_current_mA):
            msg = "Either until_time_s, until_rate_C, or until_current_mA must be set and non-zero."
            raise ValueError(msg)
        return self


class Loop(BaseTechnique):
    """Loop technique."""

    name: Literal["loop"] = "loop"
    start_step: int = Field(gt=0)  # Steps are 1-indexed
    cycle_count: int = Field(gt=0)


AnyTechnique = Union[ConstantCurrent, ConstantVoltage, OpenCircuitVoltage, Loop]


# --- Main Protocol Model ---
class Protocol(BaseModel):
    """Protocol model which can be converted to various formats."""

    sample: SampleParams
    measurement: MeasurementParams
    safety: SafetyParams
    method: list[AnyTechnique] = Field(min_length=1)  # Ensure at least one step

    def to_neware_xml(self, save_path: Path | None = None) -> str:
        """Convert the protocol to Neware XML format."""
        # Neware takes capacity as milliamp seconds (mAs)
        capacity_mAs = self.sample.capacity_mAh * 3600

        root = ET.Element("root")
        config = ET.SubElement(
            root,
            "config",
            type="Step File",
            version="17",
            client_version="BTS Client 8.0.0.478(2024.06.24)(R3)",
            date=datetime.now().strftime("%Y%m%d%H%M%S"),
            Guid=str(uuid.uuid4()),
        )
        head_info = ET.SubElement(config, "Head_Info")
        ET.SubElement(head_info, "Operate", Value="66")
        ET.SubElement(head_info, "Scale", Value="1")
        ET.SubElement(head_info, "Start_Step", Value="1", Hide_Ctrl_Step="0")
        ET.SubElement(head_info, "Creator", Value="aurora_cycler_manager.protocol")
        ET.SubElement(head_info, "Remark", Value=self.sample.name)
        ET.SubElement(head_info, "RateType", Value="105")
        ET.SubElement(head_info, "MultCap", Value=str(capacity_mAs))

        whole_prt = ET.SubElement(config, "Whole_Prt")
        protect = ET.SubElement(whole_prt, "Protect")
        main_protect = ET.SubElement(protect, "Main")
        volt = ET.SubElement(main_protect, "Volt")
        if self.safety.max_voltage_V:
            ET.SubElement(volt, "Upper", Value=str(self.safety.max_voltage_V * 10000))
        if self.safety.min_voltage_V:
            ET.SubElement(volt, "Lower", Value=str(self.safety.min_voltage_V * 10000))
        curr = ET.SubElement(main_protect, "Curr")
        if self.safety.max_current_mA:
            ET.SubElement(curr, "Upper", Value=str(self.safety.max_current_mA))
        if self.safety.min_current_mA:
            ET.SubElement(curr, "Lower", Value=str(self.safety.min_current_mA))
        if self.safety.delay_s:
            ET.SubElement(main_protect, "Delay_Time", Value=str(self.safety.delay_s))
        cap = ET.SubElement(main_protect, "Cap")
        if self.safety.max_capacity_mAh:
            ET.SubElement(cap, "Upper", Value=str(self.safety.max_capacity_mAh * 3600))

        record = ET.SubElement(whole_prt, "Record")
        main_record = ET.SubElement(record, "Main")
        if self.measurement.time_s:
            ET.SubElement(main_record, "Time", Value=str(self.measurement.time_s * 1000))
        if self.measurement.voltage_V:
            ET.SubElement(main_record, "Volt", Value=str(self.measurement.voltage_V * 10000))
        if self.measurement.current_mA:
            ET.SubElement(main_record, "Curr", Value=str(self.measurement.current_mA))

        step_info = ET.SubElement(config, "Step_Info", Num=str(len(self.method)))

        def _step_to_element(step: AnyTechnique, step_num: int, parent: ET.Element) -> None:
            """Create XML subelement from protocol technique."""
            if step.name == "constant_current":
                if step.rate_C is not None and step.rate_C != 0:
                    step_type = "1" if step.rate_C > 0 else "2"
                elif step.current_mA is not None and step.current_mA != 0:
                    step_type = "1" if step.current_mA > 0 else "2"

                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type=step_type)
                limit = ET.SubElement(step_element, "Limit")
                main = ET.SubElement(limit, "Main")
                if step.rate_C is not None:
                    ET.SubElement(main, "Rate", Value=str(abs(step.rate_C)))
                    ET.SubElement(
                        main, "Curr", Value=str(abs(step.rate_C) * self.sample.capacity_mAh)
                    )  # TODO: double check this should actually be mA
                elif step.current_mA is not None:
                    ET.SubElement(
                        main, "Curr", Value=str(abs(step.current_mA))
                    )  # TODO: double check this should actually be mA
                if step.until_time_s is not None:
                    ET.SubElement(main, "Time", Value=str(step.until_time_s))
                if step.until_voltage_V is not None:
                    ET.SubElement(main, "Stop_Volt", Value=str(step.until_voltage_V * 10000))

            elif step.name == "constant_voltage":
                if step.until_rate_C is not None and step.until_rate_C != 0:
                    step_type = "3" if step.until_rate_C > 0 else "4"
                elif step.until_current_mA is not None and step.until_current_mA != 0:
                    step_type = "3" if step.until_current_mA > 0 else "4"
                else:
                    step_type = "3"  # If it can't be figured out, default to charge
                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type=step_type)
                limit = ET.SubElement(step_element, "Limit")
                main = ET.SubElement(limit, "Main")
                ET.SubElement(main, "Volt", Value=str(step.voltage_V * 10000))
                if step.until_time_s is not None:
                    ET.SubElement(main, "Time", Value=str(step.until_time_s))
                if step.until_rate_C is not None:
                    ET.SubElement(main, "Stop_Rate", Value=str(abs(step.until_rate_C)))
                    ET.SubElement(
                        main, "Stop_Curr", Value=str(abs(step.until_rate_C) * self.sample.capacity_mAh)
                    )  # TODO: double check this should actually be mA
                elif step.until_current_mA is not None:
                    ET.SubElement(main, "Stop_Curr", Value=str(abs(step.until_current_mA)))

            elif step.name == "open_circuit_voltage":
                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type="4")
                limit = ET.SubElement(step_element, "Limit")
                main = ET.SubElement(limit, "Main")
                ET.SubElement(main, "Time", Value=str(step.until_time_s))

            elif step.name == "loop":
                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type="5")
                limit = ET.SubElement(step_element, "Limit")
                other = ET.SubElement(limit, "Other")
                ET.SubElement(other, "Start_Step", Value=str(step.start_step))
                ET.SubElement(other, "Cycle_Count", Value=str(step.cycle_count))

        for i, technique in enumerate(self.method):
            step_num = i + 1
            _step_to_element(technique, step_num, step_info)

        # Add an end step
        ET.SubElement(step_info, f"Step{step_num}", Step_ID=str(step_num), Step_Type="6")

        smbus = ET.SubElement(config, "SMBUS")
        ET.SubElement(smbus, "SMBUS_Info", Num="0", AdjacentInterval="0")

        # Convert to string and prettify it
        pretty_xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")  # noqa: S318
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                f.write(pretty_xml_string)
        return pretty_xml_string

    def to_tomato_mpg2(self, save_path: Path | None = None, tomato_output: Path = Path("C:/tomato_data/")) -> str:
        """Convert protocol to tomato 0.2.3 + MPG2 compatible JSON format."""
        tomato_dict: dict = {
            "version": "0.1",
            "sample": {},
            "method": [],
            "tomato": {
                "unlock_when_done": True,
                "verbosity": "DEBUG",
                "output": {
                    "path": str(tomato_output),
                    "prefix": self.sample.name,
                },
            },
        }
        tomato_dict["sample"]["name"] = self.sample.name
        tomato_dict["sample"]["capacity_mAh"] = self.sample.capacity_mAh
        # TODO Check if tomato can handle safety parameters. Might be fixed in the device.
        for step in self.method:
            tomato_step: dict = {}
            tomato_step["device"] = "MPG2"
            tomato_step["technique"] = step.name
            if step.name in ["constant_current", "constant_voltage", "open_circuit_voltage"]:
                if self.measurement.time_s:
                    tomato_step["measure_every_dt"] = self.measurement.time_s
                if self.measurement.current_mA:
                    tomato_step["measure_every_dI"] = self.measurement.current_mA
                if self.measurement.voltage_V:
                    tomato_step["measure_every_dE"] = self.measurement.voltage_V
                tomato_step["I_range"] = "10 mA"
                tomato_step["E_range"] = "+-5.0 V"

            if step.name == "open_circuit_voltage":
                tomato_step["time"] = step.until_time_s

            elif step.name == "constant_current":
                if step.rate_C:
                    if step.rate_C > 0:
                        charging = True
                        tomato_step["current"] = str(step.rate_C) + "C"
                    else:
                        charging = False
                        tomato_step["current"] = str(abs(step.rate_C)) + "D"
                elif step.current_mA:
                    if step.current_mA > 0:
                        charging = True
                        tomato_step["current"] = step.current_mA / 1000
                    else:
                        charging = False
                        tomato_step["current"] = step.current_mA / 1000
                if step.until_time_s:
                    tomato_step["time"] = step.until_time_s
                if step.until_voltage_V:
                    if charging:
                        tomato_step["limit_voltage_max"] = step.until_voltage_V
                    else:
                        tomato_step["limit_voltage_min"] = step.until_voltage_V

            elif step.name == "constant_voltage":
                tomato_step["voltage"] = step.voltage_V
                if step.until_time_s:
                    tomato_step["time"] = step.until_time_s
                if step.until_rate_C:
                    if step.until_rate_C > 0:
                        tomato_step["limit_current_min"] = str(step.until_rate_C) + "C"
                    else:
                        tomato_step["limit_current_max"] = str(abs(step.until_rate_C)) + "D"

            elif step.name == "loop":
                tomato_step["goto"] = step.start_step - 1  # 0-indexed in mpr
                tomato_step["n_gotos"] = step.cycle_count - 1  # gotos is one less than cycles

            tomato_dict["method"].append(tomato_step)

        def _json_serialize(obj: object) -> float:
            """Serialize Decimal objects."""
            if isinstance(obj, Decimal):
                return float(obj)
            msg = f"Type {type(obj)} not serializable"
            raise TypeError(msg)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(tomato_dict, f, indent=4, default=_json_serialize)
        return json.dumps(tomato_dict, indent=4, default=_json_serialize)

    def to_pybamm_experiment(self) -> list[str]:
        """Convert protocol to PyBaMM experiment format."""
        pybamm_experiment: list[str] = []
        loops: dict[int, dict] = {}
        for i, step in enumerate(self.method):
            step_str = ""
            if step.name == "constant_current":
                if step.rate_C:
                    if step.rate_C > 0:
                        step_str += f"Charge at {step.rate_C}C"
                    else:
                        step_str += f"Discharge at {abs(step.rate_C)}C"
                elif step.current_mA:
                    if step.current_mA > 0:
                        step_str += f"Charge at {step.current_mA} mA"
                    else:
                        step_str += f"Discharge at {abs(step.current_mA)} mA"
                if step.until_time_s:
                    if step.until_time_s % 3600 == 0:
                        step_str += f" for {int(step.until_time_s / 3600)} hours"
                    elif step.until_time_s % 60 == 0:
                        step_str += f" for {int(step.until_time_s / 60)} minutes"
                    else:
                        step_str += f" for {step.until_time_s} seconds"
                if step.until_voltage_V:
                    step_str += f" until {step.until_voltage_V} V"

            elif step.name == "constant_voltage":
                step_str += f"Hold at {step.voltage_V} V"
                conditions = []
                if step.until_time_s:
                    if step.until_time_s % 3600 == 0:
                        step_str += f" for {int(step.until_time_s / 3600)} hours"
                    elif step.until_time_s % 60 == 0:
                        step_str += f" for {int(step.until_time_s / 60)} minutes"
                    else:
                        conditions.append(f"for {step.until_time_s} seconds")
                if step.until_rate_C:
                    conditions.append(f"until {step.until_rate_C}C")
                if step.until_current_mA:
                    conditions.append(f" until {step.until_current_mA} mA")
                if conditions:
                    step_str += " " + " or ".join(conditions)

            elif step.name == "open_circuit_voltage":
                step_str += f"Rest for {step.until_time_s} seconds"

            elif step.name == "loop":
                # The string from this will get dropped later
                loops[i] = {"goto": step.start_step - 1, "n": step.cycle_count, "n_done": 0}

            pybamm_experiment.append(step_str)

        exploded_steps = []
        i = 0
        total_itr = 0
        while i < len(pybamm_experiment):
            exploded_steps.append(i)
            if i in loops and loops[i]["n_done"] < loops[i]["n"]:
                # check if it passes over a different loop, if so reset its count
                for j in loops:  # noqa: PLC0206
                    if j < i and j >= loops[i]["goto"]:
                        loops[j]["n_done"] = 0
                loops[i]["n_done"] += 1
                i = loops[i]["goto"]
            else:
                i += 1
            total_itr += 1
            if total_itr > 10000:
                msg = "Over 10000 steps in protocol to_pybamm_experiment()"
                raise RuntimeError(msg)

        # remove all loop steps from the list
        cleaned_exploded_steps = [i for i in exploded_steps if i not in loops]
        # change from list of indices to list of strings
        return [pybamm_experiment[i] for i in cleaned_exploded_steps]


def from_dict(data: dict, sample_name: str | None = None, sample_capacity_mAh: float | None = None) -> Protocol:
    """Create a Protocol instance from a dictionary."""
    # If values given then overwrite
    data.setdefault("sample", {})
    if sample_name:
        data["sample"]["name"] = sample_name
    if sample_capacity_mAh:
        data["sample"]["capacity_mAh"] = sample_capacity_mAh
    if not isinstance(data["sample"].get("name"), str) or data["sample"].get("name") == "$NAME":
        msg = "If using blank sample name or $NAME placeholder, a sample name must be provided in this function."
        raise ValueError(msg)
    cap = data["sample"].get("capacity_mAh")
    if cap == 0 or not isinstance(cap, (int, float)) or data["sample"].get("capacity_mAh") == "$CAPACITY":
        msg = "If using blank, 0, or $CAPACITY placeholder, a sample capacity must be provided in this function."
        raise ValueError(msg)
    return Protocol(**data)


def from_json(json_file: Path, sample_name: str | None = None, sample_capacity_mAh: float | None = None) -> Protocol:
    """Create a Protocol instance from a JSON file."""
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data, sample_name, sample_capacity_mAh)
