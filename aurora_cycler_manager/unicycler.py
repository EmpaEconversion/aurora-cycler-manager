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
from typing import Annotated, Literal, Union
from xml.dom import minidom

from pydantic import BaseModel, BeforeValidator, Field, model_validator
from typing_extensions import Self

getcontext().prec = 10


def coerce_to_decimal(v: Decimal | float | str) -> Decimal:
    """Coerces input (int, float, str) to Decimal."""
    if v is None:
        return None
    if isinstance(v, float):
        return Decimal(str(v))  # Avoids float precision issues
    return Decimal(v)


PreciseDecimal = Annotated[Decimal, BeforeValidator(coerce_to_decimal)]


class SampleParams(BaseModel):
    """Sample parameters."""

    name: str = Field(default="$NAME")
    capacity_mAh: PreciseDecimal | None = Field(gt=0, default=None)


class MeasurementParams(BaseModel):
    """Measurement parameters, i.e. when to record."""

    current_mA: PreciseDecimal | None = None
    voltage_V: PreciseDecimal | None = None
    time_s: PreciseDecimal = Field(gt=0)


class SafetyParams(BaseModel):
    """Safety parameters, i.e. limits before cancelling measurement."""

    max_voltage_V: PreciseDecimal | None = None
    min_voltage_V: PreciseDecimal | None = None
    max_current_mA: PreciseDecimal | None = None
    min_current_mA: PreciseDecimal | None = None
    max_capacity_mAh: PreciseDecimal | None = None
    delay_s: PreciseDecimal = Field(ge=0, default=Decimal(0))


class BaseTechnique(BaseModel):
    """Base class for all techniques."""

    name: str


class OpenCircuitVoltage(BaseTechnique):
    """Open circuit voltage technique."""

    name: Literal["open_circuit_voltage"] = "open_circuit_voltage"
    until_time_s: PreciseDecimal = Field(gt=0)


class ConstantCurrent(BaseTechnique):
    """Constant current technique."""

    name: Literal["constant_current"] = "constant_current"
    rate_C: PreciseDecimal | None = None
    current_mA: PreciseDecimal | None = None
    until_time_s: PreciseDecimal | None = None
    until_voltage_V: PreciseDecimal | None = None

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
    voltage_V: PreciseDecimal
    until_time_s: PreciseDecimal | None = None
    until_rate_C: PreciseDecimal | None = None
    until_current_mA: PreciseDecimal | None = None

    @model_validator(mode="after")
    def check_stop_condition(self) -> Self:
        """Ensure at least one of until_rate_C or until_current_mA is set."""
        has_time_s = self.until_time_s is not None and self.until_time_s != 0
        has_rate_C = self.until_rate_C is not None and self.until_rate_C != 0
        has_current_mA = self.until_current_mA is not None and self.until_current_mA != 0
        if not (has_time_s or has_rate_C or has_current_mA):
            msg = "Either until_time_s, until_rate_C, or until_current_mA must be set and non-zero."
            raise ValueError(msg)
        return self


class Loop(BaseTechnique):
    """Loop technique."""

    name: Literal["loop"] = "loop"
    start_step: int = Field(gt=0)  # Steps are 1-indexed
    cycle_count: int = Field(gt=0)


AnyTechnique = Union[BaseTechnique, ConstantCurrent, ConstantVoltage, OpenCircuitVoltage, Loop]


# --- Main Protocol Model ---
class Protocol(BaseModel):
    """Protocol model which can be converted to various formats."""

    sample: SampleParams = Field(default_factory=SampleParams)
    measurement: MeasurementParams
    safety: SafetyParams
    method: list[AnyTechnique] = Field(min_length=1)  # Ensure at least one step

    def _validate_capacity_c_rates(self) -> None:
        """Ensure if using C-rate steps, a capacity is set."""
        if not self.sample.capacity_mAh and any(
            getattr(s, "rate_C", None) or getattr(s, "until_rate_C", None) for s in self.method
        ):
            msg = "Sample capacity must be set if using C-rate steps."
            raise ValueError(msg)

    def to_neware_xml(
        self,
        save_path: Path | None = None,
        sample_name: str | None = None,
        capacity_mAh: Decimal | float | None = None,
    ) -> str:
        """Convert the protocol to Neware XML format."""
        # Allow overwriting name and capacity
        if sample_name:
            self.sample.name = sample_name
        if capacity_mAh:
            self.sample.capacity_mAh = Decimal(capacity_mAh)

        # Make sure capacity is set if using C-rate steps
        self._validate_capacity_c_rates()

        # Create XML structure
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
        # 103, non C-rate mode, seems to give more precise values vs 105
        ET.SubElement(head_info, "RateType", Value="103")
        if self.sample.capacity_mAh:
            ET.SubElement(head_info, "MultCap", Value=f"{self.sample.capacity_mAh * 3600:f}")

        whole_prt = ET.SubElement(config, "Whole_Prt")
        protect = ET.SubElement(whole_prt, "Protect")
        main_protect = ET.SubElement(protect, "Main")
        volt = ET.SubElement(main_protect, "Volt")
        if self.safety.max_voltage_V:
            ET.SubElement(volt, "Upper", Value=f"{self.safety.max_voltage_V * 10000:f}")
        if self.safety.min_voltage_V:
            ET.SubElement(volt, "Lower", Value=f"{self.safety.min_voltage_V * 10000:f}")
        curr = ET.SubElement(main_protect, "Curr")
        if self.safety.max_current_mA:
            ET.SubElement(curr, "Upper", Value=f"{self.safety.max_current_mA:f}")
        if self.safety.min_current_mA:
            ET.SubElement(curr, "Lower", Value=f"{self.safety.min_current_mA:f}")
        if self.safety.delay_s:
            ET.SubElement(main_protect, "Delay_Time", Value=f"{self.safety.delay_s * 1000:f}")
        cap = ET.SubElement(main_protect, "Cap")
        if self.safety.max_capacity_mAh:
            ET.SubElement(cap, "Upper", Value=f"{self.safety.max_capacity_mAh * 3600:f}")

        record = ET.SubElement(whole_prt, "Record")
        main_record = ET.SubElement(record, "Main")
        if self.measurement.time_s:
            ET.SubElement(main_record, "Time", Value=f"{self.measurement.time_s * 1000:f}")
        if self.measurement.voltage_V:
            ET.SubElement(main_record, "Volt", Value=f"{self.measurement.voltage_V * 10000:f}")
        if self.measurement.current_mA:
            ET.SubElement(main_record, "Curr", Value=f"{self.measurement.current_mA:f}")

        step_info = ET.SubElement(config, "Step_Info", Num=str(len(self.method) + 1))  # +1 for end step

        def _step_to_element(step: AnyTechnique, step_num: int, parent: ET.Element) -> None:
            """Create XML subelement from protocol technique."""
            if isinstance(step, ConstantCurrent):
                if step.rate_C is not None and step.rate_C != 0:
                    step_type = "1" if step.rate_C > 0 else "2"
                elif step.current_mA is not None and step.current_mA != 0:
                    step_type = "1" if step.current_mA > 0 else "2"

                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type=step_type)
                limit = ET.SubElement(step_element, "Limit")
                main = ET.SubElement(limit, "Main")
                if step.rate_C is not None:
                    ET.SubElement(main, "Rate", Value=f"{abs(step.rate_C):f}")
                    ET.SubElement(main, "Curr", Value=f"{abs(step.rate_C) * self.sample.capacity_mAh:f}")
                elif step.current_mA is not None:
                    ET.SubElement(main, "Curr", Value=f"{abs(step.current_mA):f}")
                if step.until_time_s is not None:
                    ET.SubElement(main, "Time", Value=f"{step.until_time_s * 1000:f}")
                if step.until_voltage_V is not None:
                    ET.SubElement(main, "Stop_Volt", Value=f"{step.until_voltage_V * 10000:f}")

            elif isinstance(step, ConstantVoltage):
                if step.until_rate_C is not None and step.until_rate_C != 0:
                    step_type = "3" if step.until_rate_C > 0 else "4"
                elif step.until_current_mA is not None and step.until_current_mA != 0:
                    step_type = "3" if step.until_current_mA > 0 else "4"
                else:
                    step_type = "3"  # If it can't be figured out, default to charge
                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type=step_type)
                limit = ET.SubElement(step_element, "Limit")
                main = ET.SubElement(limit, "Main")
                ET.SubElement(main, "Volt", Value=f"{step.voltage_V * 10000:f}")
                if step.until_time_s is not None:
                    ET.SubElement(main, "Time", Value=f"{step.until_time_s * 1000:f}")
                if step.until_rate_C is not None:
                    ET.SubElement(main, "Stop_Rate", Value=f"{abs(step.until_rate_C):f}")
                    ET.SubElement(main, "Stop_Curr", Value=f"{abs(step.until_rate_C) * self.sample.capacity_mAh:f}")
                elif step.until_current_mA is not None:
                    ET.SubElement(main, "Stop_Curr", Value=f"{abs(step.until_current_mA):f}")

            elif isinstance(step, OpenCircuitVoltage):
                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type="4")
                limit = ET.SubElement(step_element, "Limit")
                main = ET.SubElement(limit, "Main")
                ET.SubElement(main, "Time", Value=f"{step.until_time_s * 1000:f}")

            elif isinstance(step, Loop):
                step_element = ET.SubElement(parent, f"Step{step_num}", Step_ID=str(step_num), Step_Type="5")
                limit = ET.SubElement(step_element, "Limit")
                other = ET.SubElement(limit, "Other")
                ET.SubElement(other, "Start_Step", Value=str(step.start_step))
                ET.SubElement(other, "Cycle_Count", Value=str(step.cycle_count))

            else:
                msg = f"to_neware_xml does not support step type: {step.name}"
                raise TypeError(msg)

        for i, technique in enumerate(self.method):
            step_num = i + 1
            _step_to_element(technique, step_num, step_info)

        # Add an end step
        step_num = len(self.method) + 1
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

    def to_tomato_mpg2(
        self,
        save_path: Path | None = None,
        tomato_output: Path = Path("C:/tomato_data/"),
        sample_name: str | None = None,
        capacity_mAh: Decimal | float | None = None,
    ) -> str:
        """Convert protocol to tomato 0.2.3 + MPG2 compatible JSON format."""
        # Allow overwriting name and capacity
        if sample_name:
            self.sample.name = sample_name
        if capacity_mAh:
            self.sample.capacity_mAh = Decimal(capacity_mAh)

        # Make sure capacity is set if using C-rate steps
        self._validate_capacity_c_rates()

        # Create JSON structure
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
        # tomato -> MPG2 does not support safety parameters, they are set in the instrument
        tomato_dict["sample"]["name"] = self.sample.name
        tomato_dict["sample"]["capacity_mAh"] = self.sample.capacity_mAh
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

            if isinstance(step, OpenCircuitVoltage):
                tomato_step["time"] = step.until_time_s

            elif isinstance(step, ConstantCurrent):
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

            elif isinstance(step, ConstantVoltage):
                tomato_step["voltage"] = step.voltage_V
                if step.until_time_s:
                    tomato_step["time"] = step.until_time_s
                if step.until_rate_C:
                    if step.until_rate_C > 0:
                        tomato_step["limit_current_min"] = str(step.until_rate_C) + "C"
                    else:
                        tomato_step["limit_current_max"] = str(abs(step.until_rate_C)) + "D"

            elif isinstance(step, Loop):
                tomato_step["goto"] = step.start_step - 1  # 0-indexed in mpr
                tomato_step["n_gotos"] = step.cycle_count - 1  # gotos is one less than cycles

            else:
                msg = f"to_tomato_mpg2 does not support step type: {step.name}"
                raise TypeError(msg)

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
        # A PyBaMM experiment doesn't need capacity or sample name
        # Don't need to validate capacity if using C-rate steps
        pybamm_experiment: list[str] = []
        loops: dict[int, dict] = {}
        for i, step in enumerate(self.method):
            step_str = ""
            if isinstance(step, ConstantCurrent):
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

            elif isinstance(step, ConstantVoltage):
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

            elif isinstance(step, OpenCircuitVoltage):
                step_str += f"Rest for {step.until_time_s} seconds"

            elif isinstance(step, Loop):
                # The string from this will get dropped later
                loops[i] = {"goto": step.start_step - 1, "n": step.cycle_count, "n_done": 0}

            else:
                msg = f"to_pybamm_experiment does not support step type: {step.name}"
                raise TypeError(msg)

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
                msg = "Over 10000 steps in protocol to_pybamm_experiment(), likely a loop definition error."
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
    cap = coerce_to_decimal(data["sample"].get("capacity_mAh"))
    if cap == 0 or not isinstance(cap, (int, float, Decimal)) or data["sample"].get("capacity_mAh") == "$CAPACITY":
        msg = "If using blank, 0, or $CAPACITY placeholder, a sample capacity must be provided in this function."
        raise ValueError(msg)
    return Protocol(**data)


def from_json(json_file: Path, sample_name: str | None = None, sample_capacity_mAh: float | None = None) -> Protocol:
    """Create a Protocol instance from a JSON file."""
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return from_dict(data, sample_name, sample_capacity_mAh)
