from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PORTS, GenerationConfig, load_config
from .device_capacitance import (
    Bias,
    Geometry,
    charge_conservation_error,
    extract_port_admittance,
    relative_matrix_error,
)
from .device_capacitance_study import multilinear_interpolate
from .device_correction import create_device_correction_adapter
from .writer import CORRECTION_AXIS_ORDER


@dataclass(frozen=True)
class InterpolatedDeviceCorrection:
    pdk: str
    layout_policy: str
    ports: tuple[str, ...]
    capacitance: np.ndarray


@dataclass(frozen=True)
class InterpolatedInterconnect:
    pdk: str
    layout_policy: str
    ports: tuple[str, ...]
    conductance: np.ndarray
    capacitance: np.ndarray


@dataclass(frozen=True)
class DeviceCorrectionDiagnostic:
    output_root: Path
    primitive: str
    geometry: Geometry
    bias: Bias
    matrix_relative_error: float
    maximum_absolute_error: float
    exact_frequency_consistency: float
    interpolated_charge_residual: float
    exact_charge_residual: float


def read_interpolated_device_correction(
    archive_path: Path,
    primitive: str,
    geometry: Geometry,
    bias: Bias,
) -> InterpolatedDeviceCorrection:
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "shapeic-physical-lut" or manifest.get(
            "version"
        ) != 2:
            raise ValueError("device-correction diagnostics require a physical LUT v2")
        matches = [
            entry
            for entry in manifest.get("primitives", ())
            if entry.get("name") == primitive
        ]
        if len(matches) != 1:
            raise ValueError(
                f"physical LUT must contain exactly one '{primitive}' primitive"
            )
        entry = matches[0]
        ports = tuple(entry.get("port_order", ()))
        if ports != PORTS.get(primitive):
            raise ValueError(
                f"{primitive} ports are {ports}, expected {PORTS.get(primitive)}"
            )
        correction = entry.get("device_capacitance_correction")
        if not isinstance(correction, dict):
            raise ValueError(f"{primitive} has no device-capacitance correction")
        if tuple(correction.get("axis_order", ())) != CORRECTION_AXIS_ORDER:
            raise ValueError(
                f"{primitive} correction axes must be {CORRECTION_AXIS_ORDER}"
            )
        axis_entries = correction.get("axes", ())
        if tuple(axis.get("name") for axis in axis_entries) != CORRECTION_AXIS_ORDER:
            raise ValueError(
                f"{primitive} correction axis descriptors are out of order"
            )
        axes = tuple(
            _read_array(archive, axis["path"], f"{primitive} {axis['name']} axis")
            for axis in axis_entries
        )
        values = _read_array(
            archive,
            correction["capacitance"],
            f"{primitive} device-capacitance correction",
        )

    expected_shape = tuple(axis.size for axis in axes) + (len(ports), len(ports))
    if values.shape != expected_shape:
        raise ValueError(
            f"{primitive} correction has shape {values.shape}, "
            f"expected {expected_shape}"
        )
    point = (
        geometry.length,
        geometry.finger_width,
        float(geometry.nf),
        bias.vbs,
        bias.vgs,
        bias.vds,
    )
    interpolated = multilinear_interpolate(
        tuple(tuple(float(value) for value in axis) for axis in axes),
        values,
        point,
    )
    return InterpolatedDeviceCorrection(
        pdk=str(manifest.get("pdk", "")),
        layout_policy=str(manifest.get("layout_policy", "")),
        ports=ports,
        capacitance=interpolated,
    )


def read_interpolated_interconnect(
    archive_path: Path,
    primitive: str,
    geometry: Geometry,
) -> InterpolatedInterconnect:
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "shapeic-physical-lut" or manifest.get(
            "version"
        ) not in {1, 2}:
            raise ValueError("unsupported physical LUT archive")
        matches = [
            entry
            for entry in manifest.get("primitives", ())
            if entry.get("name") == primitive
        ]
        if len(matches) != 1:
            raise ValueError(
                f"physical LUT must contain exactly one '{primitive}' primitive"
            )
        entry = matches[0]
        ports = tuple(entry.get("port_order", ()))
        if ports != PORTS.get(primitive):
            raise ValueError(
                f"{primitive} ports are {ports}, expected {PORTS.get(primitive)}"
            )
        axis_order = ("length", "finger_width", "nf")
        if tuple(entry.get("axis_order", ())) != axis_order:
            raise ValueError(f"{primitive} interconnect axes must be {axis_order}")
        axis_entries = entry.get("axes", ())
        if tuple(axis.get("name") for axis in axis_entries) != axis_order:
            raise ValueError(
                f"{primitive} interconnect axis descriptors are out of order"
            )
        axes = tuple(
            _read_array(archive, axis["path"], f"{primitive} {axis['name']} axis")
            for axis in axis_entries
        )
        conductance = _read_array(
            archive,
            entry["conductance"],
            f"{primitive} interconnect conductance",
        )
        capacitance = _read_array(
            archive,
            entry["capacitance"],
            f"{primitive} interconnect capacitance",
        )

    expected_shape = tuple(axis.size for axis in axes) + (len(ports), len(ports))
    if conductance.shape != expected_shape or capacitance.shape != expected_shape:
        raise ValueError(
            f"{primitive} interconnect matrices must have shape {expected_shape}"
        )
    point = (geometry.length, geometry.finger_width, float(geometry.nf))
    interpolation_axes = tuple(
        tuple(float(value) for value in axis) for axis in axes
    )
    return InterpolatedInterconnect(
        pdk=str(manifest.get("pdk", "")),
        layout_policy=str(manifest.get("layout_policy", "")),
        ports=ports,
        conductance=multilinear_interpolate(
            interpolation_axes,
            conductance,
            point,
        ),
        capacitance=multilinear_interpolate(
            interpolation_axes,
            capacitance,
            point,
        ),
    )


def diagnose_device_correction(
    config_path: Path,
    archive_path: Path,
    primitive: str,
    geometry: Geometry,
    bias: Bias,
    output_root: Path,
) -> DeviceCorrectionDiagnostic:
    if output_root.exists():
        raise FileExistsError(f"diagnostic output already exists: {output_root}")
    interpolated = read_interpolated_device_correction(
        archive_path,
        primitive,
        geometry,
        bias,
    )
    config = load_config(config_path)
    _validate_sources(config, interpolated, primitive)
    adapter = create_device_correction_adapter(config)
    adapter.validate_environment()

    output_root.mkdir(parents=True)
    prepared = adapter.prepare_geometry(
        primitive,
        geometry,
        output_root / "exact",
    )
    definition = adapter.definition(primitive)
    pex = extract_port_admittance(
        adapter.simulator,
        definition,
        bias,
        prepared.pex_path,
        prepared.pex_subcircuit,
        output_root / "exact" / "bias" / "pex",
    )
    aggregate = extract_port_admittance(
        adapter.simulator,
        definition,
        bias,
        prepared.aggregate_path,
        prepared.aggregate_subcircuit,
        output_root / "exact" / "bias" / "aggregate",
    )
    exact = pex.capacitance - aggregate.capacitance
    result = DeviceCorrectionDiagnostic(
        output_root=output_root.resolve(),
        primitive=primitive,
        geometry=geometry,
        bias=bias,
        matrix_relative_error=relative_matrix_error(
            interpolated.capacitance,
            exact,
        ),
        maximum_absolute_error=float(np.max(np.abs(interpolated.capacitance - exact))),
        exact_frequency_consistency=max(
            pex.frequency_consistency,
            aggregate.frequency_consistency,
        ),
        interpolated_charge_residual=charge_conservation_error(
            interpolated.capacitance
        ),
        exact_charge_residual=charge_conservation_error(exact),
    )
    np.savez_compressed(
        output_root / "matrices.npz",
        ports=np.asarray(interpolated.ports),
        interpolated_delta_c=interpolated.capacitance,
        exact_delta_c=exact,
        difference_delta_c=interpolated.capacitance - exact,
        pex_c=pex.capacitance,
        aggregate_c=aggregate.capacitance,
    )
    _write_matrix_csv(
        output_root / "matrix_comparison.csv",
        interpolated.ports,
        interpolated.capacitance,
        exact,
    )
    _write_summary(
        output_root / "summary.json",
        result,
        config_path.resolve(),
        archive_path.resolve(),
    )
    return result


def _read_array(archive: zipfile.ZipFile, member: str, context: str) -> np.ndarray:
    try:
        data = archive.read(member)
    except KeyError as error:
        raise ValueError(f"missing {context} member '{member}'") from error
    values = np.load(io.BytesIO(data), allow_pickle=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} must contain only finite values")
    return values


def _validate_sources(
    config: GenerationConfig,
    interpolated: InterpolatedDeviceCorrection,
    primitive: str,
) -> None:
    if primitive not in config.primitives:
        raise ValueError(f"physical config does not contain primitive '{primitive}'")
    if (
        config.device_correction is None
        or config.device_correction.backend != "ngspice"
    ):
        raise ValueError("diagnostics require an NGSpice device-capacitance correction")
    if interpolated.pdk != config.pdk:
        raise ValueError(
            f"physical LUT PDK '{interpolated.pdk}' does not match "
            f"config '{config.pdk}'"
        )
    if interpolated.layout_policy != config.layout_policy:
        raise ValueError(
            "physical LUT layout policy does not match the generation config"
        )


def _write_matrix_csv(
    path: Path,
    ports: tuple[str, ...],
    interpolated: np.ndarray,
    exact: np.ndarray,
) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("row_port", "column_port", "interpolated_f", "exact_f", "difference_f")
        )
        for row, row_port in enumerate(ports):
            for column, column_port in enumerate(ports):
                writer.writerow(
                    (
                        row_port,
                        column_port,
                        f"{interpolated[row, column]:.17e}",
                        f"{exact[row, column]:.17e}",
                        f"{interpolated[row, column] - exact[row, column]:.17e}",
                    )
                )


def _write_summary(
    path: Path,
    result: DeviceCorrectionDiagnostic,
    config_path: Path,
    archive_path: Path,
) -> None:
    payload = {
        "format": "shapeic-device-correction-diagnostic",
        "version": 1,
        "physical_config": str(config_path),
        "physical_lut": str(archive_path),
        "primitive": result.primitive,
        "geometry": {
            "length_m": result.geometry.length,
            "finger_width_m": result.geometry.finger_width,
            "nf": result.geometry.nf,
        },
        "bias": {
            "vbs_v": result.bias.vbs,
            "vgs_v": result.bias.vgs,
            "vds_v": result.bias.vds,
        },
        "metrics": {
            "matrix_relative_error": result.matrix_relative_error,
            "maximum_absolute_error_f": result.maximum_absolute_error,
            "exact_frequency_consistency": result.exact_frequency_consistency,
            "interpolated_charge_residual": result.interpolated_charge_residual,
            "exact_charge_residual": result.exact_charge_residual,
        },
        "artifacts": {
            "matrices": str((result.output_root / "matrices.npz").resolve()),
            "matrix_comparison": str(
                (result.output_root / "matrix_comparison.csv").resolve()
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
