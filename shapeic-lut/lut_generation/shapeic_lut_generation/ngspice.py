from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import (
    CapacitanceNfMode,
    EXTRINSIC_CAPACITANCE_PARAMETERS,
    GenerationConfig,
    MosTerminal,
    SpiceDirectiveKind,
    sampled_parameter_name,
)


def simulate_block(
    config: GenerationConfig,
    length: float,
    vbs: float,
    finger_width: float,
) -> dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="shapeic-ngspice-") as temporary:
        root = Path(temporary)
        outputs = _simulate_nf_block(
            config,
            length,
            vbs,
            finger_width,
            config.device.nf,
            config.simulator.parameters,
            root,
        )
        for nf in config.device.capacitance_nf_samples:
            if nf == config.device.nf:
                continue
            if config.device.capacitance_nf_mode is CapacitanceNfMode.LINEAR:
                sampled = {
                    parameter: outputs[parameter] * np.float32(nf)
                    for parameter in EXTRINSIC_CAPACITANCE_PARAMETERS
                }
            else:
                sampled = _simulate_nf_block(
                    config,
                    length,
                    vbs,
                    finger_width,
                    nf,
                    EXTRINSIC_CAPACITANCE_PARAMETERS,
                    root,
                )
            outputs.update(
                (sampled_parameter_name(parameter, nf), values)
                for parameter, values in sampled.items()
            )
        return outputs


def _simulate_nf_block(
    config: GenerationConfig,
    length: float,
    vbs: float,
    finger_width: float,
    nf: int,
    parameters: tuple[str, ...],
    root: Path,
) -> dict[str, np.ndarray]:
    input_path = root / f"input_nf{nf}.spice"
    raw_path = root / f"output_nf{nf}.raw"
    log_path = root / f"ngspice_nf{nf}.log"
    input_path.write_text(
        _netlist(config, length, vbs, finger_width, nf, parameters, raw_path),
        encoding="ascii",
    )
    result = subprocess.run(
        [config.simulator.binary, "-b", "-o", str(log_path), str(input_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or not raw_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        raise RuntimeError(
            f"ngspice failed for L={length}, Vbs={vbs}, Wf={finger_width}, nf={nf}\n{log}"
        )

    data = _parse_raw(raw_path)
    expected_shape = (
        config.sweep.vgs.values().size,
        config.sweep.vds.values().size,
    )
    outputs: dict[str, np.ndarray] = {}
    for parameter in parameters:
        column = _raw_column(parameter)
        if column not in (data.dtype.names or ()):
            raise RuntimeError(
                f"ngspice output does not contain '{column}'; "
                f"available columns: {data.dtype.names}"
            )
        values = np.asarray(data[column])
        if np.iscomplexobj(values):
            if not np.allclose(values.imag, 0.0):
                raise RuntimeError(f"parameter '{parameter}' contains complex values")
            values = values.real
        values = values.reshape(expected_shape).astype(np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError(f"parameter '{parameter}' contains non-finite values")
        outputs[parameter] = values
    return outputs


def _raw_column(parameter: str) -> str:
    name = f"shapeic_{parameter}"
    if parameter == "id":
        return f"i({name})"
    if parameter in {"weff", "vth", "vdsat", "vdssat", "vsat"}:
        return f"v({name})"
    return name


def _netlist(
    config: GenerationConfig,
    length: float,
    vbs: float,
    finger_width: float,
    nf: int,
    parameters: tuple[str, ...],
    raw_path: Path,
) -> str:
    simulator = config.simulator
    device = config.device
    vgs = config.sweep.vgs
    vds = config.sweep.vds
    deck_directives: list[str] = []
    osdi_directives: list[str] = []
    for directive in config.spice.directives:
        if directive.kind is SpiceDirectiveKind.INCLUDE:
            deck_directives.append(f".include '{directive.path}'")
        elif directive.kind is SpiceDirectiveKind.LIBRARY:
            deck_directives.append(f".lib '{directive.path}' {directive.section}")
        else:
            osdi_directives.append(f"pre_osdi '{directive.path}'")
    saved: list[str] = []
    expressions: list[str] = []
    output_names: list[str] = []
    if "id" in parameters:
        saved.append("save i(vds)")
        expressions.append("let shapeic_id = abs(i(vds))")
        output_names.append("shapeic_id")
    for parameter in parameters:
        if parameter == "id":
            continue
        native_parameter = device.native_parameter(parameter)
        reference = f"@{device.hierarchy}[{native_parameter}]"
        saved.append(f"save {reference}")
        expressions.append(f"let shapeic_{parameter} = {reference}")
        output_names.append(f"shapeic_{parameter}")

    nodes = {
        MosTerminal.DRAIN: "ND",
        MosTerminal.GATE: "NG",
        MosTerminal.SOURCE: "0",
        MosTerminal.BULK: "NB",
    }
    instance_nodes = " ".join(nodes[terminal] for terminal in device.terminals)
    spice_length = device.spice_length(length)
    spice_width = device.spice_width(finger_width, nf)
    lines = [
        "* Shapeic five-dimensional LUT generation",
        *deck_directives,
        "VGS NG 0 DC=0",
        f"VBS NB 0 DC={vbs:.17g}",
        "VDS ND 0 DC=0",
        (
            f"{device.instance} {instance_nodes} {device.name} "
            f"{device.length_parameter}={spice_length:.17g} "
            f"{device.width_parameter}={spice_width:.17g} "
            f"{device.finger_parameter}={nf}"
        ),
        f".options temp={simulator.temperature_c:.17g} tnom={simulator.temperature_c:.17g}",
        ".control",
    ]
    lines.extend(osdi_directives)
    lines.extend(saved)
    lines.append(
        f"dc VDS {vds.start:.17g} {vds.stop:.17g} {vds.step:.17g} "
        f"VGS {vgs.start:.17g} {vgs.stop:.17g} {vgs.step:.17g}"
    )
    lines.extend(expressions)
    lines.append(f"write '{raw_path}' {' '.join(output_names)}")
    lines.extend([".endc", ".end", ""])
    return "\n".join(lines)


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
        plot: dict[bytes, bytes] = {}
        while True:
            fields = handle.readline(512).split(b":", maxsplit=1)
            if len(fields) != 2:
                break
            key = fields[0].lower()
            value = fields[1].strip()
            if key in metadata_keys:
                plot[key] = value
            elif key == b"variables":
                variable_count = int(plot[b"no. variables"])
                specs = [handle.readline(512).strip().decode("ascii").split() for _ in range(variable_count)]
                plot[b"varnames"] = tuple(spec[1] for spec in specs)  # type: ignore[assignment]
            elif key == b"binary":
                point_count = int(plot[b"no. points"])
                variable_count = int(plot[b"no. variables"])
                names = plot[b"varnames"]  # type: ignore[assignment]
                complex_values = b"complex" in plot.get(b"flags", b"")
                dtype = np.dtype(
                    {
                        "names": names,
                        "formats": [np.complex128 if complex_values else np.float64]
                        * variable_count,
                    }
                )
                data = np.fromfile(handle, dtype=dtype, count=point_count)
                if data.size != point_count:
                    raise RuntimeError(
                        f"raw file contains {data.size} points, expected {point_count}"
                    )
                return data
    raise RuntimeError(f"no binary plot found in {path}")
