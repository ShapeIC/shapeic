from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


MATRIX_FLOOR = 1.0e-18


@dataclass(frozen=True)
class Geometry:
    length: float
    finger_width: float
    nf: int


@dataclass(frozen=True)
class Bias:
    vbs: float
    vgs: float
    vds: float


@dataclass(frozen=True)
class MosInstance:
    name: str
    drain: str
    gate: str
    source: str
    bulk: str


@dataclass(frozen=True)
class PrimitiveDeviceDefinition:
    name: str
    ports: tuple[str, ...]
    model: str
    instances: tuple[MosInstance, ...]
    bias_variables: tuple[str, ...]


@dataclass(frozen=True)
class SimulatorConfig:
    binary: str
    model_library: Path | None
    library_section: str
    osdi_paths: tuple[Path, ...]
    temperature_c: float
    frequencies_hz: tuple[float, float]
    model_statements: tuple[str, ...] = ()


@dataclass(frozen=True)
class Admittance:
    conductance: np.ndarray
    capacitance: np.ndarray
    frequency_consistency: float


def mos_only_pex(
    text: str,
    definition: PrimitiveDeviceDefinition,
    normalize_device: Callable[[list[str]], list[str]],
) -> str:
    """Keep only a primitive's MOS devices and bind their bulks to logical B."""
    lines = _logical_lines(text)
    headers = [
        line.split()
        for line in lines
        if line.split() and line.split()[0].casefold() == ".subckt"
    ]
    if len(headers) != 1 or len(headers[0]) < 2:
        raise ValueError("PEX must contain exactly one flattened subcircuit")
    expected_ports = tuple(port.casefold() for port in definition.ports)
    found_ports = tuple(
        port.casefold() for port in headers[0][2 : 2 + len(expected_ports)]
    )
    if found_ports != expected_ports or len(headers[0]) != 2 + len(expected_ports):
        raise ValueError(
            f"{definition.name} PEX ports must be {definition.ports}, "
            f"found {tuple(headers[0][2:])}"
        )

    output = [" ".join(headers[0])]
    device_count = 0
    for line in lines:
        fields = line.split()
        if len(fields) < 6 or not fields[0].casefold().startswith("x"):
            continue
        if fields[5].casefold() != definition.model.casefold():
            continue
        fields = normalize_device(fields)
        if len(fields) < 6:
            raise ValueError("normalized MOS device has fewer than six fields")
        output.append(" ".join(fields))
        device_count += 1
    if device_count == 0:
        raise ValueError(
            f"PEX contains no MOS devices using model '{definition.model}'"
        )
    output.append(f".ends {headers[0][1]}")
    return "\n".join(output) + "\n"


def aggregate_primitive_spice(
    definition: PrimitiveDeviceDefinition,
    geometry: Geometry,
    device_parameters: Callable[[PrimitiveDeviceDefinition, Geometry], str],
) -> str:
    """Build the compact-model reference with the PCell's total device width."""
    _validate_geometry(geometry)
    parameters = device_parameters(definition, geometry)
    if not parameters.strip():
        raise ValueError("aggregate MOS parameters must not be empty")
    devices = [
        " ".join(
            (
                instance.name,
                instance.drain,
                instance.gate,
                instance.source,
                instance.bulk,
                parameters,
            )
        )
        for instance in definition.instances
    ]
    return "\n".join(
        (
            f".subckt aggregate_{definition.name} {' '.join(definition.ports)}",
            *devices,
            f".ends aggregate_{definition.name}",
            "",
        )
    )


def bias_port_voltages(
    definition: PrimitiveDeviceDefinition,
    bias: Bias,
) -> dict[str, float]:
    """Map source-referenced MOS biases onto the primitive's logical ports."""
    values = {
        "zero": 0.0,
        "vbs": bias.vbs,
        "vgs": bias.vgs,
        "vds": bias.vds,
    }
    if len(definition.ports) != len(definition.bias_variables):
        raise ValueError(
            f"{definition.name} has inconsistent port and bias definitions"
        )
    try:
        return {
            port: values[variable]
            for port, variable in zip(definition.ports, definition.bias_variables)
        }
    except KeyError as error:
        raise ValueError(
            f"{definition.name} uses unknown bias variable '{error.args[0]}'"
        ) from error


def extract_port_admittance(
    simulator: SimulatorConfig,
    definition: PrimitiveDeviceDefinition,
    bias: Bias,
    subcircuit_path: Path,
    subcircuit_name: str,
    output_root: Path,
) -> Admittance:
    """Recover a complete port matrix with one NGSpice process per excitation."""
    _validate_simulator(simulator)
    output_root.mkdir(parents=True, exist_ok=True)
    voltages = bias_port_voltages(definition, bias)
    frequencies = np.asarray(simulator.frequencies_hz, dtype=np.float64)
    currents = np.empty(
        (frequencies.size, len(definition.ports), len(definition.ports)),
        dtype=np.complex128,
    )

    for column, excitation in enumerate(definition.ports):
        run_root = output_root / f"excite_{excitation.lower()}"
        run_root.mkdir(parents=True, exist_ok=True)
        raw_path = run_root / "admittance.raw"
        netlist_path = run_root / "netlist.spice"
        netlist_path.write_text(
            port_admittance_netlist(
                simulator,
                definition.ports,
                voltages,
                excitation,
                subcircuit_path,
                subcircuit_name,
                raw_path,
            ),
            encoding="ascii",
        )
        log_path = run_root / "ngspice.log"
        completed = subprocess.run(
            [simulator.binary, "-b", "-o", str(log_path), str(netlist_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0 or not raw_path.is_file():
            log = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.is_file()
                else "(no ngspice log)"
            )
            raise RuntimeError(
                f"NGSpice port extraction failed for {definition.name}, "
                f"bias={bias}, excitation={excitation}\n{log}"
            )

        data = _parse_raw(raw_path)
        found_frequencies = np.asarray(data["frequency"]).real
        if not np.allclose(found_frequencies, frequencies, rtol=1.0e-12, atol=0.0):
            raise RuntimeError(
                f"NGSpice returned frequencies {found_frequencies}, "
                f"expected {frequencies}"
            )
        for row, port in enumerate(definition.ports):
            column_name = f"i(vport_{port.lower()})"
            if column_name not in (data.dtype.names or ()):
                raise RuntimeError(
                    f"NGSpice output is missing '{column_name}'; "
                    f"found {data.dtype.names}"
                )
            currents[:, row, column] = np.asarray(data[column_name])

    return admittance_from_port_currents(frequencies, currents)


def port_admittance_netlist(
    simulator: SimulatorConfig,
    ports: tuple[str, ...],
    voltages: dict[str, float],
    excitation: str,
    subcircuit_path: Path,
    subcircuit_name: str,
    raw_path: Path,
) -> str:
    """Create one column-excitation netlist containing both test frequencies."""
    _validate_simulator(simulator)
    if excitation not in ports:
        raise ValueError(f"excitation '{excitation}' is not a port")
    missing = [port for port in ports if port not in voltages]
    if missing:
        raise ValueError(f"missing DC voltages for ports: {', '.join(missing)}")

    sources = [
        (
            f"VPORT_{port} {port} 0 DC={voltages[port]:.17e} "
            f"AC={1 if port == excitation else 0}"
        )
        for port in ports
    ]
    saved = " ".join(f"i(VPORT_{port})" for port in ports)
    osdi = [f"pre_osdi '{path}'" for path in simulator.osdi_paths]
    model_statements = simulator.model_statements or (
        f".lib '{simulator.model_library}' {simulator.library_section}",
    )
    return "\n".join(
        (
            "* Shapeic primitive multiport admittance extraction",
            *model_statements,
            f".include '{subcircuit_path}'",
            *sources,
            f"XPRIMITIVE {' '.join(ports)} {subcircuit_name}",
            (
                f".options temp={simulator.temperature_c:.17g} "
                f"tnom={simulator.temperature_c:.17g}"
            ),
            ".control",
            *osdi,
            f"save {saved}",
            "op",
            (
                f"ac dec 1 {simulator.frequencies_hz[0]:.17e} "
                f"{simulator.frequencies_hz[1]:.17e}"
            ),
            f"write '{raw_path}' frequency {saved}",
            ".endc",
            ".end",
            "",
        )
    )


def admittance_from_port_currents(
    frequencies_hz: np.ndarray,
    source_currents: np.ndarray,
) -> Admittance:
    """Convert NGSpice voltage-source currents into averaged G and C matrices."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    currents = np.asarray(source_currents, dtype=np.complex128)
    if frequencies.shape != (2,) or not np.isfinite(frequencies).all():
        raise ValueError("exactly two finite frequencies are required")
    if np.any(frequencies <= 0.0):
        raise ValueError("frequencies must be positive")
    if currents.ndim != 3 or currents.shape[0] != frequencies.size:
        raise ValueError("source currents must have shape (frequency, port, port)")
    if currents.shape[1] != currents.shape[2]:
        raise ValueError("source-current port matrices must be square")
    if not np.isfinite(currents).all():
        raise ValueError("source currents must be finite")

    admittance = -currents
    omega = 2.0 * np.pi * frequencies
    capacitances = admittance.imag / omega[:, None, None]
    return Admittance(
        conductance=admittance.real.mean(axis=0),
        capacitance=capacitances.mean(axis=0),
        frequency_consistency=relative_matrix_error(
            capacitances[0], capacitances[1]
        ),
    )


def relative_matrix_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    predicted_array = np.asarray(predicted)
    actual_array = np.asarray(actual)
    if predicted_array.shape != actual_array.shape:
        raise ValueError("matrix shapes do not match")
    denominator = max(float(np.linalg.norm(actual_array)), MATRIX_FLOOR)
    return float(np.linalg.norm(predicted_array - actual_array)) / denominator


def charge_conservation_error(matrix: np.ndarray) -> float:
    """Measure both KCL and reference-invariance error of a port matrix."""
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("port matrix must be square")
    denominator = max(float(np.linalg.norm(values)), MATRIX_FLOOR)
    row_error = float(np.linalg.norm(values.sum(axis=1))) / denominator
    column_error = float(np.linalg.norm(values.sum(axis=0))) / denominator
    return max(row_error, column_error)


def _validate_geometry(geometry: Geometry) -> None:
    if (
        not math.isfinite(geometry.length)
        or not math.isfinite(geometry.finger_width)
        or geometry.length <= 0.0
        or geometry.finger_width <= 0.0
        or isinstance(geometry.nf, bool)
        or not isinstance(geometry.nf, int)
        or geometry.nf < 1
    ):
        raise ValueError("geometry requires positive length, finger_width, and nf")


def _validate_simulator(simulator: SimulatorConfig) -> None:
    frequencies = simulator.frequencies_hz
    if (
        len(frequencies) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in frequencies)
        or not math.isclose(frequencies[1] / frequencies[0], 10.0, rel_tol=1.0e-12)
    ):
        raise ValueError("frequencies_hz must contain two positive decade endpoints")
    if not math.isfinite(simulator.temperature_c):
        raise ValueError("temperature must be finite")
    if not simulator.binary:
        raise ValueError("simulator binary is required")
    if simulator.model_statements:
        if simulator.model_library is not None:
            raise ValueError(
                "model_statements cannot be combined with the legacy model library"
            )
        if any(not statement.strip() for statement in simulator.model_statements):
            raise ValueError("model_statements must be non-empty")
    elif simulator.model_library is None or not simulator.library_section:
        raise ValueError("a model library and section are required")


def _parse_raw(path: Path) -> np.ndarray:
    metadata_keys = {
        b"title",
        b"date",
        b"plotname",
        b"flags",
        b"no. variables",
        b"no. points",
        b"dimensions",
        b"command",
        b"option",
    }
    with path.open("rb") as handle:
        plot: dict[bytes, object] = {}
        while True:
            fields = handle.readline(512).split(b":", maxsplit=1)
            if len(fields) != 2:
                break
            key = fields[0].lower()
            value = fields[1].strip()
            if key in metadata_keys:
                plot[key] = value
            elif key == b"variables":
                count = int(plot[b"no. variables"])  # type: ignore[arg-type]
                specs = [
                    handle.readline(512).strip().decode("ascii").split()
                    for _ in range(count)
                ]
                plot[b"varnames"] = tuple(spec[1].casefold() for spec in specs)
            elif key == b"binary":
                point_count = int(plot[b"no. points"])  # type: ignore[arg-type]
                variable_count = int(plot[b"no. variables"])  # type: ignore[arg-type]
                names = plot[b"varnames"]
                flags = plot.get(b"flags", b"")
                complex_values = b"complex" in flags  # type: ignore[operator]
                dtype = np.dtype(
                    {
                        "names": names,
                        "formats": [
                            np.complex128 if complex_values else np.float64
                        ]
                        * variable_count,
                    }
                )
                data = np.fromfile(handle, dtype=dtype, count=point_count)
                if data.size != point_count:
                    raise RuntimeError(
                        f"raw file contains {data.size} points, "
                        f"expected {point_count}"
                    )
                return data
    raise RuntimeError(f"no binary plot found in {path}")


def _logical_lines(text: str) -> list[str]:
    logical: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and logical:
            logical[-1] += " " + line[1:].strip()
        else:
            logical.append(line)
    return logical
