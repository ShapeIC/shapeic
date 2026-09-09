"""Minimal electrical MOS description consumed by physical characterization."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelDirective:
    kind: str
    path: Path
    section: str | None

    def deck_statement(self) -> str | None:
        if self.kind == "include":
            return f".include '{self.path}'"
        if self.kind == "library":
            return f".lib '{self.path}' {self.section}"
        return None


@dataclass(frozen=True)
class ElectricalMosModel:
    source_path: Path
    digest: str
    pdk: str
    revision: str | None
    corner: str
    simulator_binary: str
    temperature_c: float
    directives: tuple[ModelDirective, ...]
    name: str
    instance_kind: str
    terminals: tuple[str, ...]
    length_parameter: str
    width_parameter: str
    finger_parameter: str
    width_convention: str
    capacitance_nf_mode: str
    geometry_unit_m: float

    @property
    def model_statements(self) -> tuple[str, ...]:
        return tuple(
            statement
            for directive in self.directives
            if (statement := directive.deck_statement()) is not None
        )

    @property
    def osdi_paths(self) -> tuple[Path, ...]:
        return tuple(
            directive.path for directive in self.directives if directive.kind == "osdi"
        )

    def spice_geometry(self, length_m: float, finger_width_m: float, nf: int) -> str:
        width_m = finger_width_m * nf if self.width_convention == "total" else finger_width_m
        return " ".join(
            (
                self.name,
                f"{self.length_parameter}={length_m / self.geometry_unit_m:.17e}",
                f"{self.width_parameter}={width_m / self.geometry_unit_m:.17e}",
                f"{self.finger_parameter}={nf}",
            )
        )


def load_electrical_model(
    path: Path, *, pdk: str, pdk_directory: Path
) -> ElectricalMosModel:
    source = path.resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    pdk_raw = _table(raw, "pdk")
    simulator = _table(raw, "simulator")
    spice = _table(raw, "spice")
    device = _table(raw, "device")
    configured_pdk = _string(pdk_raw, "name", "pdk")
    if configured_pdk != pdk:
        raise ValueError(
            f"electrical model '{source}' targets PDK '{configured_pdk}', expected '{pdk}'"
        )
    directives = _directives(spice, pdk_directory)
    terminals = device.get("terminals")
    if terminals != ["d", "g", "s", "b"]:
        raise ValueError(
            f"electrical model '{source}' must declare terminals = ['d', 'g', 's', 'b']"
        )
    instance_kind = _string(device, "instance_kind", "device")
    if instance_kind not in {"subcircuit", "model"}:
        raise ValueError("device.instance_kind must be 'subcircuit' or 'model'")
    width_convention = _string(device, "width_convention", "device")
    if width_convention not in {"total", "per_finger"}:
        raise ValueError("device.width_convention must be 'total' or 'per_finger'")
    capacitance_nf_mode = str(device.get("capacitance_nf_mode", "simulate"))
    if capacitance_nf_mode not in {"simulate", "linear"}:
        raise ValueError(
            "device.capacitance_nf_mode must be 'simulate' or 'linear'"
        )
    geometry_unit_m = float(device.get("geometry_unit_m", 1.0))
    if not math.isfinite(geometry_unit_m) or geometry_unit_m <= 0.0:
        raise ValueError("device.geometry_unit_m must be positive and finite")
    temperature = float(simulator.get("temperature_c", 27.0))
    if not math.isfinite(temperature):
        raise ValueError("simulator.temperature_c must be finite")
    revision = pdk_raw.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("pdk.revision must be a non-empty string when present")
    return ElectricalMosModel(
        source_path=source,
        digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        pdk=configured_pdk,
        revision=revision,
        corner=_string(pdk_raw, "corner", "pdk"),
        simulator_binary=str(simulator.get("binary", "ngspice")),
        temperature_c=temperature,
        directives=directives,
        name=_string(device, "name", "device"),
        instance_kind=instance_kind,
        terminals=tuple(terminals),
        length_parameter=_string(device, "length_parameter", "device"),
        width_parameter=_string(device, "width_parameter", "device"),
        finger_parameter=_string(device, "finger_parameter", "device"),
        width_convention=width_convention,
        capacitance_nf_mode=capacitance_nf_mode,
        geometry_unit_m=geometry_unit_m,
    )


def _directives(raw: dict[str, Any], pdk_directory: Path) -> tuple[ModelDirective, ...]:
    values = raw.get("directives")
    if not isinstance(values, list) or not values:
        raise ValueError("spice.directives must be a non-empty array of tables")
    output = []
    saw_osdi = False
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"spice.directives[{index}] must be a table")
        kind = _string(value, "kind", f"spice.directives[{index}]")
        if kind not in {"include", "library", "osdi"}:
            raise ValueError(f"unsupported SPICE directive kind '{kind}'")
        if saw_osdi and kind != "osdi":
            raise ValueError("include and library directives must precede OSDI directives")
        saw_osdi |= kind == "osdi"
        relative = Path(_string(value, "path", f"spice.directives[{index}]"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("electrical model PDK paths must stay below PDK_ROOT/PDK")
        resolved = (pdk_directory / relative).resolve()
        if not resolved.is_relative_to(pdk_directory.resolve()):
            raise ValueError("electrical model PDK path escapes PDK_ROOT/PDK")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        section = value.get("section")
        if kind == "library":
            if not isinstance(section, str) or not section.strip():
                raise ValueError("library directives require a non-empty section")
        elif section is not None:
            raise ValueError("only library directives accept a section")
        output.append(ModelDirective(kind, resolved, section))
    return tuple(output)


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _string(raw: dict[str, Any], name: str, context: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{name} must be a non-empty string")
    return value
