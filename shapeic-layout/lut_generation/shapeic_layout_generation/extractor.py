from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .reducer import reduce_first_order


@dataclass(frozen=True)
class ExtractedNetwork:
    nodes: tuple[str, ...]
    conductance: np.ndarray
    capacitance: np.ndarray


@dataclass(frozen=True)
class MagicPexResult:
    spice_path: Path
    script_path: Path
    stdout_path: Path
    stderr_path: Path
    subcircuit_name: str


@dataclass(frozen=True)
class PrimitiveExtraction:
    conductance: np.ndarray
    capacitance: np.ndarray
    pex: MagicPexResult


def canonicalize_subcircuit_ports(text: str, ports: tuple[str, ...]) -> str:
    """Rewrite one flat subcircuit header to the manifest's canonical order."""

    lines = text.splitlines()
    headers = [
        index
        for index, raw in enumerate(lines)
        if raw.strip().casefold().startswith(".subckt ")
    ]
    if len(headers) != 1:
        raise ValueError("PEX must contain exactly one flattened subcircuit")
    index = headers[0]
    fields = lines[index].split()
    found = tuple(fields[2:])
    expected_keys = tuple(port.casefold() for port in ports)
    found_keys = tuple(port.casefold() for port in found)
    if len(found) != len(ports) or set(found_keys) != set(expected_keys):
        raise ValueError(f"PEX ports must be {ports}, found {found}")
    if len(set(found_keys)) != len(found_keys):
        raise ValueError(f"PEX exposes duplicate ports: {found}")
    lines[index] = " ".join((fields[0], fields[1], *ports))
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + suffix


def run_magic(
    gds_path: Path,
    cell_name: str,
    ports: tuple[str, ...],
    *,
    magic_binary: str,
    magic_rcfile: Path,
    work_directory: Path,
    primitive: str | None = None,
    magic_startup_commands: tuple[str, ...] = (),
    pex_normalizer: Callable[[str, str], str] | None = None,
    pex_validator: Callable[[str, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    extraction = extract_primitive(
        gds_path,
        cell_name,
        ports,
        magic_binary=magic_binary,
        magic_rcfile=magic_rcfile,
        work_directory=work_directory,
        primitive=primitive,
        magic_startup_commands=magic_startup_commands,
        pex_normalizer=pex_normalizer,
        pex_validator=pex_validator,
    )
    return extraction.conductance, extraction.capacitance


def extract_primitive(
    gds_path: Path,
    cell_name: str,
    ports: tuple[str, ...],
    *,
    magic_binary: str,
    magic_rcfile: Path,
    work_directory: Path,
    primitive: str | None = None,
    magic_startup_commands: tuple[str, ...] = (),
    pex_normalizer: Callable[[str, str], str] | None = None,
    pex_validator: Callable[[str, str], None] | None = None,
) -> PrimitiveExtraction:
    result = write_magic_pex(
        gds_path,
        cell_name,
        magic_binary=magic_binary,
        magic_rcfile=magic_rcfile,
        work_directory=work_directory,
        magic_startup_commands=magic_startup_commands,
    )
    extracted_text = result.spice_path.read_text(encoding="utf-8")
    if pex_normalizer is not None:
        if primitive is None:
            raise ValueError("PEX normalization requires a primitive name")
        extracted_text = pex_normalizer(extracted_text, primitive)
    extracted_text = canonicalize_subcircuit_ports(extracted_text, ports)
    if pex_validator is not None:
        try:
            pex_validator(extracted_text, result.subcircuit_name)
        except ValueError as error:
            raise ValueError(
                f"{error}; preserved GDS: {gds_path.resolve()}; "
                f"raw PEX: {result.spice_path}"
            ) from error
    network = parse_rc_spice(extracted_text)
    indices = [_find_port(network.nodes, port) for port in ports]
    conductance, capacitance = reduce_first_order(
        network.conductance, network.capacitance, indices
    )
    for index, port in enumerate(ports):
        if not np.any(conductance[index]) and not np.any(capacitance[index]):
            raise ValueError(
                f"extracted port '{port}' is disconnected; check primitive routing and labels"
            )
    return PrimitiveExtraction(conductance, capacitance, result)


def write_magic_pex(
    gds_path: Path,
    cell_name: str,
    *,
    magic_binary: str,
    magic_rcfile: Path,
    work_directory: Path,
    magic_startup_commands: tuple[str, ...] = (),
) -> MagicPexResult:
    """Run Magic and preserve the flattened transistor-level PEX artifacts."""
    work_directory.mkdir(parents=True, exist_ok=True)
    spice_path = work_directory / f"{cell_name}.pex.spice"
    script_path = work_directory / f"{cell_name}.tcl"
    stdout_path = work_directory / "magic.stdout.log"
    stderr_path = work_directory / "magic.stderr.log"
    flat_cell_name = f"{cell_name}_flat"
    spice_path.unlink(missing_ok=True)
    script_path.write_text(
        "\n".join(
            (
                "drc off",
                *magic_startup_commands,
                f"gds read {gds_path}",
                f"load {cell_name}",
                "select top cell",
                f"flatten -dotoplabels {flat_cell_name}",
                f"load {flat_cell_name}",
                "select top cell",
                "extract do local",
                "extract all",
                "ext2spice lvs",
                "ext2spice cthresh 0",
                "ext2spice rthresh 0",
                "ext2spice extresist on",
                f"ext2spice -o {spice_path}",
                "quit -noprompt",
                "",
            )
        ),
        encoding="ascii",
    )
    command = [
        magic_binary,
        "-dnull",
        "-noconsole",
        "-rcfile",
        str(magic_rcfile),
        str(script_path),
    ]
    completed = subprocess.run(
        command,
        cwd=work_directory,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or not spice_path.is_file():
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Magic extraction failed for {cell_name}: {details}")
    return MagicPexResult(
        spice_path=spice_path.resolve(),
        script_path=script_path.resolve(),
        stdout_path=stdout_path.resolve(),
        stderr_path=stderr_path.resolve(),
        subcircuit_name=flat_cell_name,
    )


def parse_rc_spice(text: str) -> ExtractedNetwork:
    elements: list[tuple[str, str, str, float]] = []
    nodes: dict[str, int] = {}
    logical_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and logical_lines:
            logical_lines[-1] += " " + line[1:].strip()
        else:
            logical_lines.append(line)

    subcircuits = [
        line.split()
        for line in logical_lines
        if line.casefold().startswith(".subckt ")
    ]
    if len(subcircuits) > 1:
        names = ", ".join(fields[1] for fields in subcircuits if len(fields) > 1)
        raise ValueError(
            f"extracted SPICE remains hierarchical ({names}); flatten it before parsing"
        )
    if subcircuits:
        for port in subcircuits[0][2:]:
            if "=" in port or port.casefold() == "params:":
                break
            if not _is_ground(port) and port not in nodes:
                nodes[port] = len(nodes)

    for line in logical_lines:
        fields = line.split()
        kind = fields[0][0].upper() if fields else ""
        if kind not in {"R", "C"} or len(fields) < 4:
            continue
        node_a, node_b = fields[1], fields[2]
        value = parse_spice_number(fields[3])
        if not math.isfinite(value) or (kind == "R" and value <= 0.0):
            raise ValueError(f"invalid extracted element value in: {line}")
        if kind == "C" and value == 0.0:
            continue
        for node in (node_a, node_b):
            if not _is_ground(node) and node not in nodes:
                nodes[node] = len(nodes)
        elements.append((kind, node_a, node_b, value))
    if not elements or len(nodes) < 2:
        raise ValueError("extracted SPICE contains no usable RC network")
    conductance = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    capacitance = np.zeros_like(conductance)
    for kind, node_a, node_b, value in elements:
        matrix = conductance if kind == "R" else capacitance
        branch = 1.0 / value if kind == "R" else value
        _stamp(matrix, nodes.get(node_a), nodes.get(node_b), branch)
    return ExtractedNetwork(tuple(nodes), conductance, capacitance)


def parse_spice_number(token: str) -> float:
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+)?",
        token,
    )
    if match is None:
        raise ValueError(f"unsupported SPICE number '{token}'")
    suffix = (match.group(2) or "").lower()
    multipliers = {
        "": 1.0,
        "t": 1.0e12,
        "g": 1.0e9,
        "meg": 1.0e6,
        "k": 1.0e3,
        "m": 1.0e-3,
        "u": 1.0e-6,
        "n": 1.0e-9,
        "p": 1.0e-12,
        "f": 1.0e-15,
    }
    if suffix.startswith("meg"):
        multiplier = multipliers["meg"]
    elif suffix:
        multiplier = multipliers.get(suffix[0])
        if multiplier is None:
            raise ValueError(f"unsupported SPICE suffix '{suffix}'")
    else:
        multiplier = 1.0
    return float(match.group(1)) * multiplier


def _stamp(matrix: np.ndarray, node_a: int | None, node_b: int | None, value: float) -> None:
    if node_a is not None:
        matrix[node_a, node_a] += value
    if node_b is not None:
        matrix[node_b, node_b] += value
    if node_a is not None and node_b is not None:
        matrix[node_a, node_b] -= value
        matrix[node_b, node_a] -= value


def _is_ground(node: str) -> bool:
    return node.casefold() in {"0", "gnd", "vss"}


def _find_port(nodes: tuple[str, ...], port: str) -> int:
    exact = [index for index, node in enumerate(nodes) if node.casefold() == port.casefold()]
    if len(exact) == 1:
        return exact[0]
    labelled = [
        index
        for index, node in enumerate(nodes)
        if re.split(r"[#:/.]", node, maxsplit=1)[0].casefold() == port.casefold()
    ]
    if len(labelled) != 1:
        raise ValueError(f"could not identify extracted port '{port}' in {nodes}")
    return labelled[0]
