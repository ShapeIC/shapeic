from __future__ import annotations

import math
import os
import shutil
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_PARAMETERS = {
    "weff",
    "id",
    "vth",
    "vdsat",
    "vdssat",
    "vsat",
    "gm",
    "gmbs",
    "gds",
    "cgg",
    "cgd",
    "cgs",
    "cgb",
    "cdg",
    "cdd",
    "cds",
    "cdb",
    "csg",
    "csd",
    "css",
    "csb",
    "cbg",
    "cbd",
    "cbs",
    "cbb",
    "cgsol",
    "cgdol",
    "cjs",
    "cjd",
}

EXTRINSIC_CAPACITANCE_PARAMETERS = ("cgsol", "cgdol", "cjs", "cjd")
EXTRINSIC_CAPACITANCE_NF_SAMPLES = (1, 2, 3, 4)


def sampled_parameter_name(parameter: str, nf: int) -> str:
    return parameter if nf == 1 else f"{parameter}_nf{nf}"


@dataclass(frozen=True)
class LinearRange:
    start: float
    stop: float
    step: float
    stop_inclusive: bool = True

    def values(self) -> np.ndarray:
        if not all(math.isfinite(value) for value in (self.start, self.stop, self.step)):
            raise ValueError("range values must be finite")
        if self.step == 0 or (self.stop - self.start) * self.step <= 0:
            raise ValueError("range step must point from start toward stop")

        tolerance = 1.0e-12 * max(abs(self.start), abs(self.stop), abs(self.step), 1.0)
        values: list[float] = []
        index = 0
        while True:
            value = self.start + index * self.step
            before_stop = value < self.stop - tolerance if self.step > 0 else value > self.stop + tolerance
            at_stop = math.isclose(value, self.stop, rel_tol=1.0e-12, abs_tol=tolerance)
            if not before_stop and not (self.stop_inclusive and at_stop):
                break
            values.append(self.stop if at_stop else value)
            index += 1

        if not values:
            raise ValueError("range produces no values")
        if self.stop_inclusive and not math.isclose(
            values[-1], self.stop, rel_tol=1.0e-12, abs_tol=tolerance
        ):
            raise ValueError("inclusive range step does not reach stop exactly")
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class SweepConfig:
    length: np.ndarray
    vbs: LinearRange
    vgs: LinearRange
    vds: LinearRange
    finger_width: LinearRange

    def axes(self) -> dict[str, np.ndarray]:
        return {
            "length": self.length,
            "vbs": self.vbs.values(),
            "vgs": self.vgs.values(),
            "vds": self.vds.values(),
            "finger_width": self.finger_width.values(),
        }


@dataclass(frozen=True)
class SimulatorConfig:
    binary: str
    temperature_c: float
    workers: int
    parameters: tuple[str, ...]


class SpiceDirectiveKind(str, Enum):
    INCLUDE = "include"
    LIBRARY = "library"
    OSDI = "osdi"


@dataclass(frozen=True)
class SpiceDirective:
    kind: SpiceDirectiveKind
    path: Path
    section: str | None = None


@dataclass(frozen=True)
class SpiceConfig:
    directives: tuple[SpiceDirective, ...]


class SpiceInstanceKind(str, Enum):
    SUBCIRCUIT = "subcircuit"
    MODEL = "model"


class MosTerminal(str, Enum):
    DRAIN = "d"
    GATE = "g"
    SOURCE = "s"
    BULK = "b"


class WidthConvention(str, Enum):
    TOTAL = "total"
    PER_FINGER = "per_finger"


class CapacitanceNfMode(str, Enum):
    SIMULATE = "simulate"
    LINEAR = "linear"


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    instance: str
    instance_kind: SpiceInstanceKind
    terminals: tuple[MosTerminal, ...]
    length_parameter: str
    width_parameter: str
    finger_parameter: str
    width_convention: WidthConvention
    hierarchy: str
    parameter_map: dict[str, str]
    nf: int
    geometry_unit_m: float = 1.0
    capacitance_nf_mode: CapacitanceNfMode = CapacitanceNfMode.SIMULATE
    capacitance_nf_samples: tuple[int, ...] = ()

    def native_parameter(self, canonical_name: str) -> str:
        try:
            return self.parameter_map[canonical_name]
        except KeyError as error:
            raise ValueError(f"missing native mapping for parameter '{canonical_name}'") from error

    def spice_length(self, length: float) -> float:
        return length / self.geometry_unit_m

    def spice_width(self, finger_width: float, nf: int) -> float:
        if self.width_convention is WidthConvention.TOTAL:
            finger_width *= nf
        return finger_width / self.geometry_unit_m


@dataclass(frozen=True)
class PdkConfig:
    name: str
    revision: str | None
    corner: str
    nominal_voltage: float
    directory: Path


@dataclass(frozen=True)
class GenerationConfig:
    source_path: Path
    output_path: Path
    description: str
    simulator: SimulatorConfig
    spice: SpiceConfig
    device: DeviceConfig
    sweep: SweepConfig
    pdk: PdkConfig | None = None

    def output_parameters(self) -> tuple[str, ...]:
        parameters = list(self.simulator.parameters)
        for nf in self.device.capacitance_nf_samples:
            if nf == self.device.nf:
                continue
            parameters.extend(
                sampled_parameter_name(parameter, nf)
                for parameter in EXTRINSIC_CAPACITANCE_PARAMETERS
            )
        return tuple(parameters)


def load_config(path: Path) -> GenerationConfig:
    source_path = path.resolve()
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)

    output = _table(raw, "output")
    simulator = _table(raw, "simulator")
    device = _table(raw, "device")
    sweep = _table(raw, "sweep")
    pdk = _pdk_config(raw.get("pdk"))

    output_path = _resolve_path(str(output["path"]), source_path.parent)
    parameters = tuple(str(value).lower() for value in simulator["parameters"])
    unknown = sorted(set(parameters) - SUPPORTED_PARAMETERS)
    if unknown:
        raise ValueError(f"unsupported parameters: {', '.join(unknown)}")
    if not parameters or len(parameters) != len(set(parameters)):
        raise ValueError("parameters must be a non-empty list without duplicates")

    spice_value = raw.get("spice")
    if spice_value is None:
        spice = _legacy_spice_config(simulator, pdk, source_path.parent)
        device_config = _legacy_device_config(device, parameters)
    else:
        if pdk is None:
            raise ValueError("[pdk] is required when [spice] is configured")
        legacy_fields = sorted(
            {"model_library", "library_section", "osdi_paths"}.intersection(simulator)
        )
        if legacy_fields:
            raise ValueError(
                "[spice] cannot be combined with legacy simulator fields: "
                + ", ".join(legacy_fields)
            )
        if not isinstance(spice_value, dict):
            raise ValueError("[spice] must be a table")
        spice = _spice_config(spice_value, pdk.directory)
        device_config = _typed_device_config(device, parameters)

    lengths = np.asarray(sweep["length"], dtype=np.float64)
    _validate_axis(lengths, "length")
    config = GenerationConfig(
        source_path=source_path,
        output_path=output_path,
        description=str(output.get("description", "Shapeic five-dimensional MOS LUT")),
        simulator=SimulatorConfig(
            binary=str(simulator.get("binary", "ngspice")),
            temperature_c=float(simulator.get("temperature_c", 27.0)),
            workers=int(simulator.get("workers", 1)),
            parameters=parameters,
        ),
        spice=spice,
        device=device_config,
        sweep=SweepConfig(
            length=lengths,
            vbs=_linear_range(sweep, "vbs"),
            vgs=_linear_range(sweep, "vgs"),
            vds=_linear_range(sweep, "vds"),
            finger_width=_linear_range(sweep, "finger_width"),
        ),
        pdk=pdk,
    )
    _validate_config(config)
    for name, values in config.sweep.axes().items():
        _validate_axis(values, name)
    return config


def _table(values: dict[str, Any], name: str) -> dict[str, Any]:
    value = values.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _linear_range(sweep: dict[str, Any], name: str) -> LinearRange:
    values = _table(sweep, name)
    return LinearRange(
        start=float(values["start"]),
        stop=float(values["stop"]),
        step=float(values["step"]),
        stop_inclusive=bool(values.get("stop_inclusive", True)),
    )


def _capacitance_nf_samples(device: dict[str, Any]) -> tuple[int, ...]:
    values = device.get("capacitance_nf_samples", [])
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError("capacitance_nf_samples must be a list of integers")
    return tuple(values)


def _legacy_spice_config(
    simulator: dict[str, Any], pdk: PdkConfig | None, config_directory: Path
) -> SpiceConfig:
    if "model_library" not in simulator:
        raise ValueError("simulator.model_library is required by the legacy configuration")
    path_base = pdk.directory if pdk is not None else config_directory
    path_resolver = _resolve_pdk_path if pdk is not None else _resolve_path
    directives = [
        SpiceDirective(
            kind=SpiceDirectiveKind.LIBRARY,
            path=path_resolver(str(simulator["model_library"]), path_base),
            section=str(simulator.get("library_section", "mos_tt")),
        )
    ]
    directives.extend(
        SpiceDirective(
            kind=SpiceDirectiveKind.OSDI,
            path=path_resolver(str(value), path_base),
        )
        for value in simulator.get("osdi_paths", [])
    )
    return SpiceConfig(tuple(directives))


def _spice_config(values: dict[str, Any], pdk_directory: Path) -> SpiceConfig:
    unknown = sorted(set(values) - {"directives"})
    if unknown:
        raise ValueError("unsupported [spice] fields: " + ", ".join(unknown))
    raw_directives = values.get("directives")
    if not isinstance(raw_directives, list) or not raw_directives:
        raise ValueError("spice.directives must be a non-empty array of tables")

    directives: list[SpiceDirective] = []
    saw_osdi = False
    for index, raw in enumerate(raw_directives):
        context = f"spice.directives[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{context} must be a table")
        unknown = sorted(set(raw) - {"kind", "path", "section"})
        if unknown:
            raise ValueError(f"unsupported {context} fields: " + ", ".join(unknown))
        kind = _enum_value(
            SpiceDirectiveKind,
            _required_nonempty_string(raw, "kind", context),
            f"{context}.kind",
        )
        if saw_osdi and kind is not SpiceDirectiveKind.OSDI:
            raise ValueError("include and library directives must precede all OSDI directives")
        saw_osdi |= kind is SpiceDirectiveKind.OSDI
        path = _resolve_pdk_path(
            _required_nonempty_string(raw, "path", context), pdk_directory
        )
        section_value = raw.get("section")
        if kind is SpiceDirectiveKind.LIBRARY:
            section = _required_nonempty_string(raw, "section", context)
            _validate_spice_token(section, f"{context}.section")
        else:
            if section_value is not None:
                raise ValueError(f"{context}.section is only valid for library directives")
            section = None
        directives.append(SpiceDirective(kind=kind, path=path, section=section))
    return SpiceConfig(tuple(directives))


def _legacy_device_config(
    device: dict[str, Any], parameters: tuple[str, ...]
) -> DeviceConfig:
    typed_fields = {
        "instance_kind",
        "terminals",
        "length_parameter",
        "width_parameter",
        "finger_parameter",
        "width_convention",
        "parameter_map",
    }.intersection(device)
    if typed_fields:
        raise ValueError(
            "typed device fields require [spice]: " + ", ".join(sorted(typed_fields))
        )
    return DeviceConfig(
        name=str(device["name"]),
        instance=str(device.get("instance", "XM1")),
        instance_kind=SpiceInstanceKind.SUBCIRCUIT,
        terminals=tuple(MosTerminal),
        length_parameter="l",
        width_parameter="w",
        finger_parameter="ng",
        width_convention=WidthConvention.TOTAL,
        hierarchy=str(device["hierarchy"]),
        parameter_map={parameter: parameter for parameter in parameters if parameter != "id"},
        nf=int(device.get("nf", 1)),
        capacitance_nf_samples=_capacitance_nf_samples(device),
    )


def _typed_device_config(
    device: dict[str, Any], parameters: tuple[str, ...]
) -> DeviceConfig:
    supported_fields = {
        "name",
        "instance",
        "instance_kind",
        "terminals",
        "length_parameter",
        "width_parameter",
        "finger_parameter",
        "width_convention",
        "geometry_unit_m",
        "capacitance_nf_mode",
        "hierarchy",
        "parameter_map",
        "nf",
        "capacitance_nf_samples",
    }
    unknown = sorted(set(device) - supported_fields)
    if unknown:
        raise ValueError("unsupported [device] fields: " + ", ".join(unknown))
    instance_kind = _enum_value(
        SpiceInstanceKind,
        _required_nonempty_string(device, "instance_kind", "device"),
        "device.instance_kind",
    )
    raw_terminals = device.get("terminals")
    if not isinstance(raw_terminals, list):
        raise ValueError("device.terminals must be a list")
    terminals = tuple(
        _enum_value(MosTerminal, value, f"device.terminals[{index}]")
        for index, value in enumerate(raw_terminals)
    )
    width_convention = _enum_value(
        WidthConvention,
        _required_nonempty_string(device, "width_convention", "device"),
        "device.width_convention",
    )
    capacitance_nf_mode = _enum_value(
        CapacitanceNfMode,
        device.get("capacitance_nf_mode", CapacitanceNfMode.SIMULATE.value),
        "device.capacitance_nf_mode",
    )
    raw_map = device.get("parameter_map")
    if not isinstance(raw_map, dict):
        raise ValueError("missing [device.parameter_map] table")
    parameter_map: dict[str, str] = {}
    for canonical, native in raw_map.items():
        if not isinstance(canonical, str) or not isinstance(native, str) or not native:
            raise ValueError("device.parameter_map must map strings to non-empty strings")
        normalized = canonical.lower()
        if normalized in parameter_map:
            raise ValueError(f"duplicate device parameter mapping '{normalized}'")
        parameter_map[normalized] = native

    return DeviceConfig(
        name=_required_nonempty_string(device, "name", "device"),
        instance=_required_nonempty_string(device, "instance", "device"),
        instance_kind=instance_kind,
        terminals=terminals,
        length_parameter=_required_nonempty_string(device, "length_parameter", "device"),
        width_parameter=_required_nonempty_string(device, "width_parameter", "device"),
        finger_parameter=_required_nonempty_string(device, "finger_parameter", "device"),
        width_convention=width_convention,
        hierarchy=_required_nonempty_string(device, "hierarchy", "device"),
        parameter_map=parameter_map,
        nf=int(device.get("nf", 1)),
        geometry_unit_m=float(device.get("geometry_unit_m", 1.0)),
        capacitance_nf_mode=capacitance_nf_mode,
        capacitance_nf_samples=_capacitance_nf_samples(device),
    )


def _enum_value(enum_type: type[Enum], value: Any, context: str):
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        supported = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{context} must be one of: {supported}") from error


def _validate_spice_token(value: str, context: str) -> None:
    if any(character.isspace() for character in value) or any(
        character in value for character in "'[]"
    ):
        raise ValueError(f"{context} must be a single SPICE token")


def _pdk_config(value: Any) -> PdkConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("[pdk] must be a table")

    name = _required_nonempty_string(value, "name", "pdk")
    name_path = Path(name)
    if (
        name_path.is_absolute()
        or name in {".", ".."}
        or len(name_path.parts) != 1
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("pdk.name must be a single directory name")
    corner = _required_nonempty_string(value, "corner", "pdk")
    revision_value = value.get("revision")
    revision = None
    if revision_value is not None:
        if not isinstance(revision_value, str) or not revision_value.strip():
            raise ValueError("pdk.revision must be a non-empty string when provided")
        revision = revision_value

    if "nominal_voltage" not in value:
        raise ValueError("pdk.nominal_voltage is required")
    try:
        nominal_voltage = float(value["nominal_voltage"])
    except (TypeError, ValueError) as error:
        raise ValueError("pdk.nominal_voltage must be positive and finite") from error
    if not math.isfinite(nominal_voltage) or nominal_voltage <= 0.0:
        raise ValueError("pdk.nominal_voltage must be positive and finite")

    root_value = os.environ.get("PDK_ROOT")
    selected = os.environ.get("PDK")
    if not root_value:
        raise ValueError("PDK_ROOT must be set when [pdk] is configured")
    if not selected:
        raise ValueError("PDK must be set when [pdk] is configured")
    if selected != name:
        raise ValueError(f"PDK selects '{selected}', but pdk.name is '{name}'")

    root = _resolve_environment_path(root_value, "PDK_ROOT")
    if not root.is_dir():
        raise FileNotFoundError(f"PDK_ROOT directory not found: {root}")
    directory = (root / selected).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"PDK '{selected}' was not found at '{directory}'; install it separately "
            "or update PDK_ROOT/PDK"
        )

    return PdkConfig(
        name=name,
        revision=revision,
        corner=corner,
        nominal_voltage=nominal_voltage,
        directory=directory,
    )


def _required_nonempty_string(values: dict[str, Any], name: str, table: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{table}.{name} must be a non-empty string")
    return value


def _resolve_environment_path(value: str, name: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"{name} contains an undefined environment variable: {value}")
    return Path(expanded).resolve()


def _resolve_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"path contains an undefined environment variable: {value}")
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_pdk_path(value: str, pdk_directory: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"PDK path contains an undefined environment variable: {value}")
    path = Path(expanded)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"PDK-owned path must be relative to PDK_ROOT/PDK: {value}")
    return (pdk_directory / path).resolve()


def _validate_axis(values: np.ndarray, name: str) -> None:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if values.size > 1:
        differences = np.diff(values)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"{name} values must be strictly monotonic")


def _validate_config(config: GenerationConfig) -> None:
    simulator = config.simulator
    if simulator.workers < 1:
        raise ValueError("workers must be at least one")
    if not math.isfinite(simulator.temperature_c):
        raise ValueError("temperature_c must be finite")
    if shutil.which(simulator.binary) is None:
        raise ValueError(f"simulator binary '{simulator.binary}' is not accessible")
    if not config.spice.directives:
        raise ValueError("at least one SPICE directive is required")
    for directive in config.spice.directives:
        if not directive.path.is_file():
            raise FileNotFoundError(
                f"{directive.kind.value} file not found: {directive.path}"
            )
        if "'" in str(directive.path) or any(
            character in str(directive.path) for character in "\r\n"
        ):
            raise ValueError(f"unsafe SPICE path: {directive.path}")

    device = config.device
    if not device.name or not device.hierarchy or device.nf != 1:
        raise ValueError("device name and hierarchy are required, and nf must equal one")
    for value, context in [
        (device.name, "device.name"),
        (device.instance, "device.instance"),
        (device.length_parameter, "device.length_parameter"),
        (device.width_parameter, "device.width_parameter"),
        (device.finger_parameter, "device.finger_parameter"),
        (device.hierarchy, "device.hierarchy"),
    ]:
        _validate_spice_token(value, context)
    expected_prefix = "X" if device.instance_kind is SpiceInstanceKind.SUBCIRCUIT else "M"
    if not device.instance.upper().startswith(expected_prefix):
        raise ValueError(
            f"device.instance must start with '{expected_prefix}' for "
            f"instance_kind = '{device.instance_kind.value}'"
        )
    if len(device.terminals) != 4 or set(device.terminals) != set(MosTerminal):
        raise ValueError("device.terminals must contain d, g, s and b exactly once")
    geometry_parameters = {
        device.length_parameter,
        device.width_parameter,
        device.finger_parameter,
    }
    if len(geometry_parameters) != 3:
        raise ValueError("length, width and finger parameter names must be distinct")
    if not math.isfinite(device.geometry_unit_m) or device.geometry_unit_m <= 0.0:
        raise ValueError("device.geometry_unit_m must be positive and finite")

    expected_mappings = set(simulator.parameters) - {"id"}
    actual_mappings = set(device.parameter_map)
    if actual_mappings != expected_mappings:
        missing = sorted(expected_mappings - actual_mappings)
        extra = sorted(actual_mappings - expected_mappings)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise ValueError(
            "device.parameter_map does not match parameters (" + "; ".join(details) + ")"
        )
    for canonical, native in device.parameter_map.items():
        _validate_spice_token(canonical, f"device.parameter_map.{canonical}")
        _validate_spice_token(native, f"device.parameter_map.{canonical}")

    samples = device.capacitance_nf_samples
    if device.capacitance_nf_mode is not CapacitanceNfMode.SIMULATE and not samples:
        raise ValueError("device.capacitance_nf_mode requires capacitance_nf_samples")
    if samples:
        if samples != EXTRINSIC_CAPACITANCE_NF_SAMPLES:
            raise ValueError(
                "capacitance_nf_samples must equal [1, 2, 3, 4] in format version 2"
            )
        missing = sorted(
            set(EXTRINSIC_CAPACITANCE_PARAMETERS) - set(simulator.parameters)
        )
        if missing:
            raise ValueError(
                "capacitance_nf_samples requires parameters: " + ", ".join(missing)
            )
    if config.output_path.suffix != ".npz":
        raise ValueError("output path must end in .npz")
