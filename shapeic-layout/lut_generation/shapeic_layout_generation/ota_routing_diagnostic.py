"""Historical IHP OTA diagnostic; not part of physical LUT generation."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GenerationConfig, load_config
from .device_capacitance import Geometry, relative_matrix_error
from .device_correction import create_device_correction_adapter
from .device_correction_diagnostic import (
    InterpolatedInterconnect,
    read_interpolated_interconnect,
)
from .extractor import parse_rc_spice
from .reducer import reduce_first_order


GLOBAL_NODES = ("VOUT", "N1", "VINP", "VINN", "IBIAS", "VDD")
VSS_ALIAS = "__SHAPEIC_VSS_PORT__"
LOCAL_CONNECTIONS = {
    "simplediffpair": {
        "DP": "VOUT",
        "DN": "N1",
        "GP": "VINP",
        "GN": "VINN",
        "S": "IBIAS",
    },
    "currentmirror": {
        "DOUT": "VOUT",
        "DREF": "N1",
        "S": "VDD",
        "B": "VDD",
    },
}


@dataclass(frozen=True)
class ReducedInterconnect:
    ports: tuple[str, ...]
    conductance: np.ndarray
    capacitance: np.ndarray


@dataclass(frozen=True)
class LocalRoutingResult:
    primitive: str
    geometry: Geometry
    lut: InterpolatedInterconnect
    exact_floating_bulk: ReducedInterconnect
    exact_bound_bulk: ReducedInterconnect


@dataclass(frozen=True)
class OtaRoutingDiagnostic:
    output_root: Path
    local_capacitance_interpolation_error: dict[str, float]
    local_bulk_binding_delta: dict[str, float]
    lut_local_vs_full_error: float
    exact_local_vs_full_error: float
    routing_maximum_absolute_capacitance: float


def diagnose_ota_routing(
    config_path: Path,
    archive_path: Path,
    ota_manifest_path: Path,
    diff_geometry: Geometry,
    mirror_geometry: Geometry,
    diff_bulk_node: str,
    output_root: Path,
) -> OtaRoutingDiagnostic:
    if output_root.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_root}")
    if diff_bulk_node not in {"IBIAS", "0"}:
        raise ValueError("diff bulk node must be 'IBIAS' or '0'")
    config = load_config(config_path)
    adapter = create_device_correction_adapter(config)
    adapter.validate_environment()
    ota_manifest = _load_ota_manifest(ota_manifest_path, config)
    output_root.mkdir(parents=True)

    local_results = {
        primitive: _characterize_local_interconnect(
            config,
            adapter,
            archive_path,
            primitive,
            geometry,
            output_root / "local" / primitive,
        )
        for primitive, geometry in (
            ("simplediffpair", diff_geometry),
            ("currentmirror", mirror_geometry),
        )
    }
    full = _reduce_full_ota(
        Path(ota_manifest["pex_path"]),
        ota_manifest["subcircuit_name"],
    )
    full_connections = {
        "VOUT": "VOUT",
        "VINP": "VINP",
        "VINN": "VINN",
        "IBIAS": "IBIAS",
        "VDD": "VDD",
        VSS_ALIAS: diff_bulk_node,
        full.ports[-1]: "N1",
    }
    full_g = stamp_port_matrix(full.conductance, full.ports, full_connections)
    full_c = stamp_port_matrix(full.capacitance, full.ports, full_connections)

    lut_g = np.zeros_like(full_g)
    lut_c = np.zeros_like(full_c)
    floating_g = np.zeros_like(full_g)
    floating_c = np.zeros_like(full_c)
    bound_g = np.zeros_like(full_g)
    bound_c = np.zeros_like(full_c)
    interpolation_errors = {}
    bulk_deltas = {}
    for primitive, local in local_results.items():
        connections = dict(LOCAL_CONNECTIONS[primitive])
        if primitive == "simplediffpair":
            connections["B"] = diff_bulk_node
        lut_g += stamp_port_matrix(local.lut.conductance, local.lut.ports, connections)
        lut_c += stamp_port_matrix(local.lut.capacitance, local.lut.ports, connections)
        floating_g += stamp_port_matrix(
            local.exact_floating_bulk.conductance,
            local.exact_floating_bulk.ports,
            connections,
        )
        floating_c += stamp_port_matrix(
            local.exact_floating_bulk.capacitance,
            local.exact_floating_bulk.ports,
            connections,
        )
        bound_g += stamp_port_matrix(
            local.exact_bound_bulk.conductance,
            local.exact_bound_bulk.ports,
            connections,
        )
        bound_c += stamp_port_matrix(
            local.exact_bound_bulk.capacitance,
            local.exact_bound_bulk.ports,
            connections,
        )
        interpolation_errors[primitive] = relative_matrix_error(
            local.lut.capacitance,
            local.exact_floating_bulk.capacitance,
        )
        bulk_deltas[primitive] = _relative_delta(
            local.exact_bound_bulk.capacitance,
            local.exact_floating_bulk.capacitance,
        )

    routing_c = full_c - bound_c
    result = OtaRoutingDiagnostic(
        output_root=output_root.resolve(),
        local_capacitance_interpolation_error=interpolation_errors,
        local_bulk_binding_delta=bulk_deltas,
        lut_local_vs_full_error=relative_matrix_error(lut_c, full_c),
        exact_local_vs_full_error=relative_matrix_error(bound_c, full_c),
        routing_maximum_absolute_capacitance=float(np.max(np.abs(routing_c))),
    )
    np.savez_compressed(
        output_root / "matrices.npz",
        nodes=np.asarray(GLOBAL_NODES),
        full_ota_g=full_g,
        full_ota_c=full_c,
        lut_local_g=lut_g,
        lut_local_c=lut_c,
        exact_floating_local_g=floating_g,
        exact_floating_local_c=floating_c,
        exact_bound_local_g=bound_g,
        exact_bound_local_c=bound_c,
        routing_residual_g=full_g - bound_g,
        routing_residual_c=routing_c,
    )
    _write_matrix_csv(
        output_root / "capacitance_comparison.csv",
        full_c,
        lut_c,
        floating_c,
        bound_c,
    )
    _write_summary(
        output_root / "summary.json",
        result,
        config_path.resolve(),
        archive_path.resolve(),
        ota_manifest_path.resolve(),
        diff_bulk_node,
    )
    return result


def stamp_port_matrix(
    matrix: np.ndarray,
    ports: tuple[str, ...],
    connections: dict[str, str],
) -> np.ndarray:
    if matrix.shape != (len(ports), len(ports)):
        raise ValueError("port matrix shape does not match its ports")
    missing = sorted(set(ports) - set(connections))
    if missing:
        raise ValueError(f"missing port connections: {', '.join(missing)}")
    indices = {node: index for index, node in enumerate(GLOBAL_NODES)}
    output = np.zeros((len(GLOBAL_NODES), len(GLOBAL_NODES)))
    for row, row_port in enumerate(ports):
        row_node = connections[row_port]
        if row_node == "0":
            continue
        if row_node not in indices:
            raise ValueError(f"unknown global node '{row_node}'")
        for column, column_port in enumerate(ports):
            column_node = connections[column_port]
            if column_node == "0":
                continue
            if column_node not in indices:
                raise ValueError(f"unknown global node '{column_node}'")
            output[indices[row_node], indices[column_node]] += matrix[row, column]
    return output


def normalize_primitive_bulk(text: str, model: str, bulk_port: str = "B") -> str:
    bulk_nodes = set()
    for line in _logical_lines(text):
        fields = line.split()
        if (
            len(fields) >= 6
            and fields[0].casefold().startswith("x")
            and fields[5].casefold() == model.casefold()
        ):
            bulk_nodes.add(fields[4].casefold())
    if len(bulk_nodes) != 1:
        raise ValueError(
            f"expected one extracted bulk node for model '{model}', found {bulk_nodes}"
        )
    extracted_bulk = next(iter(bulk_nodes))
    output = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("*"):
            output.append(raw)
            continue
        output.append(
            " ".join(
                bulk_port if field.casefold() == extracted_bulk else field
                for field in raw.split()
            )
        )
    return "\n".join(output) + "\n"


def reduce_explicit_interconnect(
    text: str,
    ports: tuple[str, ...],
) -> ReducedInterconnect:
    network = parse_rc_spice(text)
    retained = tuple(_find_node(network.nodes, port) for port in ports)
    conductance, capacitance = reduce_first_order(
        network.conductance,
        network.capacitance,
        list(retained),
    )
    return ReducedInterconnect(ports, conductance, capacitance)


def _characterize_local_interconnect(
    config: GenerationConfig,
    adapter,
    archive_path: Path,
    primitive: str,
    geometry: Geometry,
    output_root: Path,
) -> LocalRoutingResult:
    lut = read_interpolated_interconnect(
        archive_path,
        primitive,
        geometry,
    )
    if lut.pdk != config.pdk or lut.layout_policy != config.layout_policy:
        raise ValueError("physical LUT metadata does not match the generation config")
    adapter.prepare_geometry(primitive, geometry, output_root)
    raw_path = output_root / "primitive.pex.spice"
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"adapter did not preserve raw primitive PEX: {raw_path}"
        )
    raw = raw_path.read_text(encoding="utf-8")
    ports = config.port_order(primitive)
    floating = reduce_explicit_interconnect(raw, ports)
    bound = reduce_explicit_interconnect(
        normalize_primitive_bulk(raw, adapter.definition(primitive).model),
        ports,
    )
    np.savez_compressed(
        output_root / "interconnect_matrices.npz",
        ports=np.asarray(ports),
        lut_g=lut.conductance,
        lut_c=lut.capacitance,
        exact_floating_g=floating.conductance,
        exact_floating_c=floating.capacitance,
        exact_bound_g=bound.conductance,
        exact_bound_c=bound.capacitance,
    )
    return LocalRoutingResult(primitive, geometry, lut, floating, bound)


def _reduce_full_ota(pex_path: Path, expected_subcircuit: str) -> ReducedInterconnect:
    text = pex_path.read_text(encoding="utf-8")
    logical = _logical_lines(text)
    headers = [
        fields
        for line in logical
        if (fields := line.split()) and fields[0].casefold() == ".subckt"
    ]
    if len(headers) != 1 or headers[0][1] != expected_subcircuit:
        raise ValueError("full OTA PEX subcircuit does not match its manifest")
    n1 = _find_ota_internal_node(logical)
    symbolic = _replace_token(text, "VSS", VSS_ALIAS)
    ports = ("VOUT", "VINP", "VINN", "IBIAS", "VDD", VSS_ALIAS, n1)
    return reduce_explicit_interconnect(symbolic, ports)


def _find_ota_internal_node(lines: list[str]) -> str:
    candidates = set()
    for line in lines:
        fields = line.split()
        if (
            len(fields) < 6
            or not fields[0].casefold().startswith("x")
            or fields[5].casefold() != "sg13_lv_nmos"
            or fields[2].casefold() != "vinn"
        ):
            continue
        drain, source = fields[1], fields[3]
        if drain.casefold() == "ibias" and source.casefold() != "ibias":
            candidates.add(source)
        elif source.casefold() == "ibias" and drain.casefold() != "ibias":
            candidates.add(drain)
    if len({node.casefold() for node in candidates}) != 1:
        raise ValueError(f"could not identify one OTA internal node: {candidates}")
    return next(iter(candidates))


def _load_ota_manifest(path: Path, config: GenerationConfig) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "shapeic-ota-pex" or manifest.get("version") != 1:
        raise ValueError("unsupported OTA PEX manifest")
    if manifest.get("layout_policy") != config.layout_policy:
        raise ValueError("OTA PEX layout policy does not match the physical config")
    for name in ("pex_path", "subcircuit_name"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"OTA PEX manifest has no valid '{name}'")
    if not Path(manifest["pex_path"]).is_file():
        raise FileNotFoundError(manifest["pex_path"])
    return manifest


def _find_node(nodes: tuple[str, ...], port: str) -> int:
    exact = [
        index for index, node in enumerate(nodes) if node.casefold() == port.casefold()
    ]
    if len(exact) == 1:
        return exact[0]
    labelled = [
        index
        for index, node in enumerate(nodes)
        if re.split(r"[#:/.]", node, maxsplit=1)[0].casefold() == port.casefold()
    ]
    if len(labelled) != 1:
        raise ValueError(f"could not identify port '{port}' in {nodes}")
    return labelled[0]


def _logical_lines(text: str) -> list[str]:
    output = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and output:
            output[-1] += " " + line[1:].strip()
        else:
            output.append(line)
    return output


def _replace_token(text: str, source: str, replacement: str) -> str:
    return "\n".join(
        " ".join(
            replacement if field.casefold() == source.casefold() else field
            for field in raw.split()
        )
        if raw.strip() and not raw.lstrip().startswith("*")
        else raw
        for raw in text.splitlines()
    ) + "\n"


def _relative_delta(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first)), 1.0e-18)
    return float(np.linalg.norm(first - second)) / denominator


def _write_matrix_csv(
    path: Path,
    full: np.ndarray,
    lut: np.ndarray,
    floating: np.ndarray,
    bound: np.ndarray,
) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "row_node",
                "column_node",
                "full_ota_f",
                "lut_local_f",
                "exact_floating_local_f",
                "exact_bound_local_f",
                "routing_residual_f",
            )
        )
        for row, row_node in enumerate(GLOBAL_NODES):
            for column, column_node in enumerate(GLOBAL_NODES):
                writer.writerow(
                    (
                        row_node,
                        column_node,
                        f"{full[row, column]:.17e}",
                        f"{lut[row, column]:.17e}",
                        f"{floating[row, column]:.17e}",
                        f"{bound[row, column]:.17e}",
                        f"{full[row, column] - bound[row, column]:.17e}",
                    )
                )


def _write_summary(
    path: Path,
    result: OtaRoutingDiagnostic,
    config_path: Path,
    archive_path: Path,
    ota_manifest_path: Path,
    diff_bulk_node: str,
) -> None:
    payload = {
        "format": "shapeic-ota-routing-diagnostic",
        "version": 1,
        "physical_config": str(config_path),
        "physical_lut": str(archive_path),
        "ota_pex_manifest": str(ota_manifest_path),
        "diff_bulk_node": diff_bulk_node,
        "nodes": list(GLOBAL_NODES),
        "metrics": {
            "local_capacitance_interpolation_error": (
                result.local_capacitance_interpolation_error
            ),
            "local_bulk_binding_delta": result.local_bulk_binding_delta,
            "lut_local_vs_full_error": result.lut_local_vs_full_error,
            "exact_local_vs_full_error": result.exact_local_vs_full_error,
            "routing_maximum_absolute_capacitance_f": (
                result.routing_maximum_absolute_capacitance
            ),
        },
        "artifacts": {
            "matrices": str((result.output_root / "matrices.npz").resolve()),
            "capacitance_comparison": str(
                (result.output_root / "capacitance_comparison.csv").resolve()
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
