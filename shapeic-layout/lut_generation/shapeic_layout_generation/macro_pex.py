"""PDK-independent rendering and Magic extraction of CellKit macros."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import GenerationConfig
from .extractor import canonicalize_subcircuit_ports, write_magic_pex


@dataclass(frozen=True)
class MacroPexTopology:
    subcircuit_name: str
    transistor_count: int
    resistor_count: int
    capacitor_count: int


def generate_macro_pex(
    config: GenerationConfig,
    macro_name: str,
    geometries: Mapping[str, tuple[float, float, int]],
    output_directory: Path,
    *,
    manifest_format: str = "shapeic-macro-pex",
) -> Path:
    """Render one CellKit macro and preserve its flattened PEX artifacts."""
    if config.cellkit is None:
        raise ValueError("macro PEX generation requires a CellKit configuration")
    if config.extractor.backend != "magic":
        raise ValueError("macro PEX generation requires physical.backend='magic'")
    if config.extractor.magic_rcfile is None:
        raise ValueError("macro PEX generation requires a Magic rcfile")

    macro = config.cellkit.catalog.macro_layout(macro_name)
    resolved_geometries = {
        instance: config.cellkit.geometry(length, finger_width, nf)
        for instance, (length, finger_width, nf) in geometries.items()
    }
    rendered = macro.render(resolved_geometries)
    root = output_directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    gds_path = root / f"{macro_name}.gds"
    rendered.component.write_gds(gds_path)

    extraction = write_magic_pex(
        gds_path,
        rendered.cell_name,
        magic_binary=config.extractor.magic_binary,
        magic_rcfile=config.extractor.magic_rcfile,
        work_directory=root / "magic",
        magic_startup_commands=config.extractor.magic_startup_commands,
    )
    normalized_path = root / f"{macro_name}.pex.spice"
    topology = prepare_macro_pex(
        extraction.spice_path,
        normalized_path,
        macro_name=macro_name,
        port_order=macro.port_order,
        technology=config.cellkit.technology,
        bulk_ports=_macro_bulk_ports(config.cellkit.catalog, macro),
        expected_subcircuit=extraction.subcircuit_name,
    )
    manifest = {
        "format": manifest_format,
        "version": 1,
        "macro": macro_name,
        "pdk": config.pdk,
        "pdk_revision": config.pdk_revision,
        "layout_policy": rendered.layout_policy,
        "implementation_sha256": rendered.implementation_digest,
        "technology_sha256": config.cellkit.catalog.technology_digest,
        "cell_name": rendered.cell_name,
        "subcircuit_name": topology.subcircuit_name,
        "port_order": list(macro.port_order),
        "instances": {
            name: {
                "primitive": primitive,
                "length_m": geometries[name][0],
                "finger_width_m": geometries[name][1],
                "nf": geometries[name][2],
            }
            for name, primitive in macro.instances
        },
        "gds_path": str(gds_path.resolve()),
        "pex_path": str(normalized_path.resolve()),
        "raw_pex_path": str(extraction.spice_path),
        "magic_script_path": str(extraction.script_path),
        "magic_stdout_path": str(extraction.stdout_path),
        "magic_stderr_path": str(extraction.stderr_path),
        "transistor_count": topology.transistor_count,
        "resistor_count": topology.resistor_count,
        "capacitor_count": topology.capacitor_count,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest_path


def prepare_macro_pex(
    source: Path,
    output: Path,
    *,
    macro_name: str,
    port_order: tuple[str, ...],
    technology,
    bulk_ports: Mapping[str, str],
    expected_subcircuit: str | None = None,
) -> MacroPexTopology:
    text = source.read_text(encoding="utf-8")
    normalized = technology.normalize_macro_pex(text, macro_name, bulk_ports)
    normalized = canonicalize_subcircuit_ports(normalized, port_order)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(normalized, encoding="utf-8")
    return validate_macro_pex(
        normalized,
        port_order,
        expected_subcircuit=expected_subcircuit,
    )


def validate_macro_pex(
    text: str,
    port_order: tuple[str, ...],
    *,
    expected_subcircuit: str | None = None,
) -> MacroPexTopology:
    lines = _logical_lines(text)
    headers = [
        fields
        for line in lines
        if (fields := line.split()) and fields[0].casefold() == ".subckt"
    ]
    if len(headers) != 1 or len(headers[0]) < 2:
        raise ValueError("macro PEX must contain exactly one flattened subcircuit")
    header = headers[0]
    found_ports = tuple(header[2:])
    if tuple(value.casefold() for value in found_ports) != tuple(
        value.casefold() for value in port_order
    ):
        raise ValueError(f"macro PEX ports must be {port_order}, found {found_ports}")
    subcircuit = header[1]
    if expected_subcircuit is not None and subcircuit.casefold() != expected_subcircuit.casefold():
        raise ValueError(
            f"expected macro PEX subcircuit '{expected_subcircuit}', found '{subcircuit}'"
        )
    counts = {"r": 0, "c": 0, "m": 0}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        kind = fields[0][0].casefold()
        if kind in {"r", "c"} and len(fields) >= 4:
            counts[kind] += 1
        elif kind in {"m", "x"} and len(fields) >= 5:
            counts["m"] += 1
    if counts["m"] == 0:
        raise ValueError("macro PEX contains no active devices")
    return MacroPexTopology(subcircuit, counts["m"], counts["r"], counts["c"])


def _logical_lines(text: str) -> list[str]:
    output: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and output:
            output[-1] += " " + line[1:].strip()
        else:
            output.append(line)
    return output


def _macro_bulk_ports(catalog, macro) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for instance, primitive in macro.instances:
        descriptor = catalog.primitive_descriptor(primitive)
        for terminal in {branch.bulk for branch in descriptor.branches}:
            net = next(
                value
                for value in macro.nets
                if (instance, terminal) in value.terminals
            )
            target = net.external_port or net.name
            polarity = descriptor.polarity.value
            previous = bindings.get(polarity)
            if previous is not None and previous != target:
                raise ValueError(
                    f"macro '{macro.name}' has multiple {polarity} bulk nets; "
                    "the selected technology requires one bulk net per polarity"
                )
            bindings[polarity] = target
    return bindings
