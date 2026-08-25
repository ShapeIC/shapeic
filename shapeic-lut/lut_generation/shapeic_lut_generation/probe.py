from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CapacitanceNfMode,
    EXTRINSIC_CAPACITANCE_PARAMETERS,
    GenerationConfig,
    MosTerminal,
    SpiceDirectiveKind,
)
from .ngspice import _parse_raw, normalize_capacitance_parameters


TERMINALS = ("G", "D", "S", "B")
INTRINSIC_PARAMETERS = ("cgg", "cgd", "cgs", "cdg", "cdd", "cds", "csg", "csd", "css")
SCALING_PARAMETERS = ("id", "gm", "gds")
PROBE_NF_VALUES = (1, 2, 3, 4)
PROBE_FREQUENCIES_HZ = (1.0e6, 1.0e7)
MATRIX_ERROR_LIMIT = 0.01
SCALING_ERROR_LIMIT = 0.01
CHARGE_RESIDUAL_LIMIT = 1.0e-7


@dataclass(frozen=True)
class ProbePoint:
    length: float
    finger_width: float
    vbs: float
    vgs: float
    vds: float


class ProbeError(RuntimeError):
    def __init__(self, message: str, artifacts: Path):
        super().__init__(message)
        self.artifacts = artifacts


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[3] / "target" / "shapeic-electrical-probe"


def run_probe(
    config: GenerationConfig,
    point: ProbePoint,
    output_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = (output_root or default_output_root()).resolve() / str(time.time_ns())
    root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "device": config.device.name,
        "capacitance_convention": config.device.capacitance_convention.value,
        "capacitance_nf_mode": config.device.capacitance_nf_mode.value,
        "frequencies_hz": list(PROBE_FREQUENCIES_HZ),
        "limits": {
            "matrix_relative_error": MATRIX_ERROR_LIMIT,
            "scaling_relative_error": SCALING_ERROR_LIMIT,
            "charge_residual": CHARGE_RESIDUAL_LIMIT,
        },
        "nf_results": [],
    }
    try:
        _validate_point(config, point)
        _write_json(root / "resolved_config.json", _resolved_config(config, point))
        base_raw, base = _run_operating_point(
            config, point, 1, root / "nf_1" / "op", native=True
        )
        for nf in PROBE_NF_VALUES:
            nf_root = root / f"nf_{nf}"
            if nf == 1:
                raw_values, values = base_raw, base
            elif config.device.capacitance_nf_mode is CapacitanceNfMode.SIMULATE:
                raw_values, values = _run_operating_point(
                    config, point, nf, nf_root / "op", native=True
                )
            else:
                raw_values, values = _run_operating_point(
                    config, point, nf, nf_root / "op", native=False
                )

            intrinsic = intrinsic_matrix({name: base[name] * nf for name in INTRINSIC_PARAMETERS})
            extrinsic_source = values
            predicted = combined_matrix(intrinsic, extrinsic_source)
            ac_matrices = _run_ac_multiport(config, point, nf, nf_root / "ac")
            matrix_errors = [matrix_relative_error(predicted, matrix) for matrix in ac_matrices]
            frequency_error = matrix_relative_error(ac_matrices[0], ac_matrices[1])
            scaling_errors = {
                name: relative_error(values[name], base[name] * nf)
                for name in SCALING_PARAMETERS
            }
            charge = {
                "predicted": charge_residual(predicted),
                "ac": [charge_residual(matrix) for matrix in ac_matrices],
            }
            passed = (
                max(matrix_errors) <= MATRIX_ERROR_LIMIT
                and max(scaling_errors.values()) <= SCALING_ERROR_LIMIT
                and max([charge["predicted"], *charge["ac"]]) <= CHARGE_RESIDUAL_LIMIT
            )
            nf_result = {
                "nf": nf,
                "status": "pass" if passed else "fail",
                "raw_parameters": raw_values,
                "canonical_parameters": values,
                "predicted_capacitance_f": predicted.tolist(),
                "ac_capacitance_f": [matrix.tolist() for matrix in ac_matrices],
                "matrix_relative_errors": matrix_errors,
                "frequency_relative_error": frequency_error,
                "scaling_relative_errors": scaling_errors,
                "charge_residuals": charge,
            }
            report["nf_results"].append(nf_result)
            _write_json(nf_root / "result.json", nf_result)

        failed = [result["nf"] for result in report["nf_results"] if result["status"] != "pass"]
        report["status"] = "pass" if not failed else "fail"
        if failed:
            raise RuntimeError(f"probe gates failed for nf={failed}")
    except Exception as error:
        report["status"] = "fail"
        report["error"] = str(error)
        _write_json(root / "report.json", report)
        raise ProbeError(str(error), root) from error

    _write_json(root / "report.json", report)
    return root, report


def intrinsic_matrix(values: dict[str, float]) -> np.ndarray:
    cgg, cgd, cgs, cdg, cdd, cds, csg, csd, css = (
        float(values[name]) for name in INTRINSIC_PARAMETERS
    )
    compact = np.asarray(
        [
            [cgg, cgd, cgs, cgg - cgd - cgs],
            [cdg, cdd, cds, cdd - cdg - cds],
            [csg, csd, css, css - csg - csd],
            [cgg - cdg - csg, cdd - cgd - csd, css - cgs - cds, 0.0],
        ],
        dtype=np.float64,
    )
    compact[3, 3] = compact[3, 0] + compact[3, 1] + compact[3, 2]
    matrix = compact.copy()
    for row in range(4):
        for column in range(4):
            if row != column:
                matrix[row, column] = -matrix[row, column]
    _require_finite(matrix, "intrinsic capacitance matrix")
    return matrix


def combined_matrix(intrinsic: np.ndarray, values: dict[str, float]) -> np.ndarray:
    matrix = np.asarray(intrinsic, dtype=np.float64).copy()
    for first, second, name in ((0, 2, "cgsol"), (0, 1, "cgdol"), (2, 3, "cjs"), (1, 3, "cjd")):
        value = float(values[name])
        matrix[first, first] += value
        matrix[second, second] += value
        matrix[first, second] -= value
        matrix[second, first] -= value
    _require_finite(matrix, "combined capacitance matrix")
    return matrix


def matrix_relative_error(reference: np.ndarray, observed: np.ndarray) -> float:
    return float(np.linalg.norm(reference - observed) / max(np.linalg.norm(observed), 1.0e-18))


def relative_error(reference: float, observed: float) -> float:
    return abs(reference - observed) / max(abs(observed), 1.0e-30)


def charge_residual(matrix: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(matrix)), 1.0e-30)
    return max(float(np.max(np.abs(matrix.sum(axis=0)))), float(np.max(np.abs(matrix.sum(axis=1))))) / scale


def _run_operating_point(
    config: GenerationConfig,
    point: ProbePoint,
    nf: int,
    root: Path,
    native: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "op.raw"
    deck = _operating_point_netlist(config, point, nf, raw_path, native)
    _run_ngspice(config, deck, root / "op.spice", root / "op.log", raw_path)
    data = _parse_raw(raw_path)
    if data.size != 1:
        raise RuntimeError(f"operating-point raw contains {data.size} points, expected one")
    values: dict[str, float] = {}
    for parameter in (*SCALING_PARAMETERS, *INTRINSIC_PARAMETERS, *EXTRINSIC_CAPACITANCE_PARAMETERS):
        column = f"i(shapeic_{parameter})" if parameter == "id" else f"shapeic_{parameter}"
        values[parameter] = _scalar_column(data, column)
    for optional in ("weff", "vth", "vdsat", "vdssat", "vsat"):
        column = f"shapeic_{optional}"
        if column in (data.dtype.names or ()):
            values[optional] = _scalar_column(data, column)
    raw_values = dict(values)
    arrays = {name: np.asarray(value) for name, value in values.items()}
    canonical = {
        name: float(np.asarray(value))
        for name, value in normalize_capacitance_parameters(
            arrays, config.device.capacitance_convention
        ).items()
    }
    _write_json(root / "parameter_bindings.json", {"raw": raw_values, "canonical": canonical})
    return raw_values, canonical


def _run_ac_multiport(
    config: GenerationConfig, point: ProbePoint, nf: int, root: Path
) -> list[np.ndarray]:
    admittance = np.zeros((len(PROBE_FREQUENCIES_HZ), 4, 4), dtype=np.complex128)
    for excited_index, excited in enumerate(TERMINALS):
        excitation_root = root / f"excite_{excited.lower()}"
        excitation_root.mkdir(parents=True, exist_ok=True)
        raw_path = excitation_root / "ac.raw"
        deck = _ac_netlist(config, point, nf, excited, raw_path)
        _run_ngspice(config, deck, excitation_root / "ac.spice", excitation_root / "ac.log", raw_path)
        data = _parse_raw(raw_path)
        if data.size != len(PROBE_FREQUENCIES_HZ):
            raise RuntimeError(f"AC raw contains {data.size} points, expected {len(PROBE_FREQUENCIES_HZ)}")
        for row, terminal in enumerate(TERMINALS):
            column = f"i(vport{terminal.lower()})"
            values = _complex_column(data, column)
            admittance[:, row, excited_index] = -values
    matrices = [
        admittance[index].imag / (2.0 * math.pi * frequency)
        for index, frequency in enumerate(PROBE_FREQUENCIES_HZ)
    ]
    for matrix in matrices:
        _require_finite(matrix, "AC capacitance matrix")
    _write_json(root / "matrices.json", {str(frequency): matrix.tolist() for frequency, matrix in zip(PROBE_FREQUENCIES_HZ, matrices)})
    return matrices


def _operating_point_netlist(
    config: GenerationConfig,
    point: ProbePoint,
    nf: int,
    raw_path: Path,
    native: bool,
) -> str:
    deck, osdi = _directives(config)
    parameters = tuple(dict.fromkeys((*SCALING_PARAMETERS[1:], *INTRINSIC_PARAMETERS, *EXTRINSIC_CAPACITANCE_PARAMETERS, "weff", "vth", "vdsat", "vdssat", "vsat")))
    available = set(config.simulator.parameters)
    instance_names = [config.device.instance] if native else [
        f"{config.device.instance[0]}PROBE{index}" for index in range(1, nf + 1)
    ]
    saved = ["save i(vportd)"]
    expressions = ["let shapeic_id = abs(i(vportd))"]
    outputs = ["shapeic_id"]
    for parameter in parameters:
        if parameter not in available:
            continue
        references = [
            f"@{_hierarchy_for_instance(config, instance)}[{config.device.native_parameter(parameter)}]"
            for instance in instance_names
        ]
        saved.extend(f"save {reference}" for reference in references)
        expressions.append(f"let shapeic_{parameter} = {' + '.join(references)}")
        outputs.append(f"shapeic_{parameter}")
    devices = [_device_line(config, point, nf, native=True)] if native else [
        _device_line(config, point, 1, native=False, index=index)
        for index in range(1, nf + 1)
    ]
    lines = ["* ShapeIC electrical operating-point probe", *deck, *_bias_sources(point), *devices, _options(config), ".control", *osdi, *saved, "op", *expressions, f"write '{raw_path}' {' '.join(outputs)}", ".endc", ".end", ""]
    return "\n".join(lines)


def _ac_netlist(config: GenerationConfig, point: ProbePoint, nf: int, excited: str, raw_path: Path) -> str:
    deck, osdi = _directives(config)
    sources = _bias_sources(point, excited)
    native = config.device.capacitance_nf_mode is CapacitanceNfMode.SIMULATE
    devices = [_device_line(config, point, nf, native=True)] if native else [
        _device_line(config, point, 1, native=False, index=index) for index in range(1, nf + 1)
    ]
    currents = " ".join(f"i(vport{terminal.lower()})" for terminal in TERMINALS)
    lines = ["* ShapeIC electrical multiport AC probe", *deck, *sources, *devices, _options(config), ".control", *osdi, f"save {currents}", f"ac dec 1 {PROBE_FREQUENCIES_HZ[0]:.17g} {PROBE_FREQUENCIES_HZ[1]:.17g}", f"write '{raw_path}' {currents}", ".endc", ".end", ""]
    return "\n".join(lines)


def _directives(config: GenerationConfig) -> tuple[list[str], list[str]]:
    deck: list[str] = []
    osdi: list[str] = []
    for directive in config.spice.directives:
        if directive.kind is SpiceDirectiveKind.INCLUDE:
            deck.append(f".include '{directive.path}'")
        elif directive.kind is SpiceDirectiveKind.LIBRARY:
            deck.append(f".lib '{directive.path}' {directive.section}")
        else:
            osdi.append(f"pre_osdi '{directive.path}'")
    return deck, osdi


def _bias_sources(point: ProbePoint, excited: str | None = None) -> list[str]:
    voltages = {"G": point.vgs, "D": point.vds, "S": 0.0, "B": point.vbs}
    return [
        f"VPORT{terminal} N{terminal} 0 DC={voltage:.17g}" + (f" AC={1 if terminal == excited else 0}" if excited else "")
        for terminal, voltage in voltages.items()
    ]


def _device_line(config: GenerationConfig, point: ProbePoint, nf: int, native: bool, index: int = 1) -> str:
    device = config.device
    nodes = {MosTerminal.DRAIN: "ND", MosTerminal.GATE: "NG", MosTerminal.SOURCE: "NS", MosTerminal.BULK: "NB"}
    instance_nodes = " ".join(nodes[terminal] for terminal in device.terminals)
    instance = device.instance if native else f"{device.instance[0]}PROBE{index}"
    return (
        f"{instance} {instance_nodes} {device.name} "
        f"{device.length_parameter}={device.spice_length(point.length):.17g} "
        f"{device.width_parameter}={device.spice_width(point.finger_width, nf):.17g} "
        f"{device.finger_parameter}={nf}"
    )


def _hierarchy_for_instance(config: GenerationConfig, instance: str) -> str:
    configured = config.device.instance.lower()
    components = config.device.hierarchy.split(".")
    try:
        index = [component.lower() for component in components].index(configured)
    except ValueError as error:
        raise RuntimeError(
            f"device hierarchy '{config.device.hierarchy}' does not contain instance "
            f"'{config.device.instance}'"
        ) from error
    components[index] = instance.lower()
    return ".".join(components)


def _options(config: GenerationConfig) -> str:
    temperature = config.simulator.temperature_c
    return f".options temp={temperature:.17g} tnom={temperature:.17g}"


def _run_ngspice(config: GenerationConfig, deck: str, input_path: Path, log_path: Path, raw_path: Path) -> None:
    input_path.write_text(deck, encoding="ascii")
    result = subprocess.run([config.simulator.binary, "-b", "-o", str(log_path), str(input_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode != 0 or not raw_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        raise RuntimeError(f"ngspice probe failed for {input_path}\n{log}")


def _scalar_column(data: np.ndarray, column: str) -> float:
    if column not in (data.dtype.names or ()):
        raise RuntimeError(f"ngspice output does not contain '{column}'; available columns: {data.dtype.names}")
    value = np.asarray(data[column]).reshape(-1)[0]
    if np.iscomplexobj(value):
        if not math.isclose(float(value.imag), 0.0, abs_tol=1.0e-24):
            raise RuntimeError(f"'{column}' contains a complex operating-point value")
        value = value.real
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"'{column}' is not finite")
    return result


def _complex_column(data: np.ndarray, column: str) -> np.ndarray:
    if column not in (data.dtype.names or ()):
        raise RuntimeError(f"ngspice output does not contain '{column}'; available columns: {data.dtype.names}")
    values = np.asarray(data[column], dtype=np.complex128)
    _require_finite(values, column)
    return values


def _require_finite(values: np.ndarray, context: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context} contains non-finite values")


def _validate_point(config: GenerationConfig, point: ProbePoint) -> None:
    values = (point.length, point.finger_width, point.vbs, point.vgs, point.vds)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("probe point values must be finite")
    if point.length <= 0.0 or point.finger_width <= 0.0:
        raise ValueError("length and finger width must be positive")
    axes = config.sweep.axes()
    for name, value in (("length", point.length), ("finger_width", point.finger_width), ("vbs", point.vbs), ("vgs", point.vgs), ("vds", point.vds)):
        lower, upper = sorted((float(axes[name][0]), float(axes[name][-1])))
        if value < lower or value > upper:
            raise ValueError(f"probe {name}={value} is outside configured range [{lower}, {upper}]")
    required = set((*SCALING_PARAMETERS, *INTRINSIC_PARAMETERS, *EXTRINSIC_CAPACITANCE_PARAMETERS))
    missing = sorted(required - set(config.simulator.parameters))
    if missing:
        raise ValueError("probe requires simulator parameters: " + ", ".join(missing))


def _resolved_config(config: GenerationConfig, point: ProbePoint) -> dict[str, Any]:
    return {
        "source": str(config.source_path),
        "simulator": config.simulator.binary,
        "temperature_c": config.simulator.temperature_c,
        "pdk": None if config.pdk is None else {"name": config.pdk.name, "revision": config.pdk.revision, "corner": config.pdk.corner, "directory": str(config.pdk.directory)},
        "device": {"name": config.device.name, "instance": config.device.instance, "hierarchy": config.device.hierarchy, "width_convention": config.device.width_convention.value, "capacitance_nf_mode": config.device.capacitance_nf_mode.value, "capacitance_convention": config.device.capacitance_convention.value},
        "point": {"length": point.length, "finger_width": point.finger_width, "vbs": point.vbs, "vgs": point.vgs, "vds": point.vds},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
