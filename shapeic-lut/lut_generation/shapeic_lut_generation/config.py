from __future__ import annotations

import math
import os
import shutil
import tomllib
from dataclasses import dataclass
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
    model_library: Path
    library_section: str
    osdi_paths: tuple[Path, ...]
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    instance: str
    hierarchy: str
    nf: int
    capacitance_nf_samples: tuple[int, ...] = ()


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
    path_base = pdk.directory if pdk is not None else source_path.parent
    path_resolver = _resolve_pdk_path if pdk is not None else _resolve_path
    model_library = path_resolver(str(simulator["model_library"]), path_base)
    osdi_paths = tuple(
        path_resolver(str(value), path_base) for value in simulator.get("osdi_paths", [])
    )
    parameters = tuple(str(value).lower() for value in simulator["parameters"])
    unknown = sorted(set(parameters) - SUPPORTED_PARAMETERS)
    if unknown:
        raise ValueError(f"unsupported parameters: {', '.join(unknown)}")
    if not parameters or len(parameters) != len(set(parameters)):
        raise ValueError("parameters must be a non-empty list without duplicates")

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
            model_library=model_library,
            library_section=str(simulator.get("library_section", "mos_tt")),
            osdi_paths=osdi_paths,
            parameters=parameters,
        ),
        device=DeviceConfig(
            name=str(device["name"]),
            instance=str(device.get("instance", "XM1")),
            hierarchy=str(device["hierarchy"]),
            nf=int(device.get("nf", 1)),
            capacitance_nf_samples=_capacitance_nf_samples(device),
        ),
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
    if not simulator.model_library.is_file():
        raise FileNotFoundError(f"model library not found: {simulator.model_library}")
    for path in simulator.osdi_paths:
        if not path.is_file():
            raise FileNotFoundError(f"OSDI model not found: {path}")
    if not config.device.name or not config.device.hierarchy or config.device.nf != 1:
        raise ValueError("device name and hierarchy are required, and nf must equal one")
    samples = config.device.capacitance_nf_samples
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
