from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import PORTS, DeviceCorrectionConfig, GenerationConfig
from .device_capacitance import (
    Bias,
    Geometry,
    SimulatorConfig,
    extract_port_admittance,
)
from .device_capacitance_adapter import DeviceCapacitanceAdapter
from .ihp_device_capacitance import IhpSg13g2DeviceCapacitanceAdapter


def correction_shape(
    lengths: np.ndarray,
    finger_widths: np.ndarray,
    correction: DeviceCorrectionConfig,
    primitive: str,
) -> tuple[int, ...]:
    bias = correction.biases[primitive]
    port_count = len(PORTS[primitive])
    return (
        lengths.size,
        finger_widths.size,
        correction.finger_counts.size,
        bias.vbs.size,
        bias.vgs.size,
        bias.vds.size,
        port_count,
        port_count,
    )


def create_device_correction_adapter(
    config: GenerationConfig,
) -> DeviceCapacitanceAdapter:
    correction = config.device_correction
    if correction is None or correction.backend != "ngspice":
        raise ValueError("a real device correction requires the ngspice backend")
    if correction.model_library is None:
        raise ValueError("a real device correction requires a model library")
    simulator = SimulatorConfig(
        binary=correction.binary,
        model_library=correction.model_library,
        library_section=correction.library_section,
        osdi_paths=correction.osdi_paths,
        temperature_c=correction.temperature_c,
        frequencies_hz=correction.frequencies_hz,
    )
    if config.pdk == "ihp-sg13g2":
        return IhpSg13g2DeviceCapacitanceAdapter(
            config,
            simulator,
            correction.workers,
        )
    raise ValueError(f"no device-capacitance adapter is available for '{config.pdk}'")


def synthetic_device_correction(
    primitive: str,
    lengths: np.ndarray,
    finger_widths: np.ndarray,
    correction: DeviceCorrectionConfig,
) -> np.ndarray:
    output = np.empty(
        correction_shape(lengths, finger_widths, correction, primitive),
        dtype=np.float64,
    )
    bias = correction.biases[primitive]
    port_count = len(PORTS[primitive])
    base = np.full((port_count, port_count), -1.0)
    np.fill_diagonal(base, port_count - 1.0)
    for indices in itertools.product(
        range(lengths.size),
        range(finger_widths.size),
        range(correction.finger_counts.size),
        range(bias.vbs.size),
        range(bias.vgs.size),
        range(bias.vds.size),
    ):
        value = (
            1.0
            + lengths[indices[0]] * 1.0e6
            + finger_widths[indices[1]] * 1.0e6
            + correction.finger_counts[indices[2]]
            + abs(bias.vbs[indices[3]])
            + abs(bias.vgs[indices[4]])
            + abs(bias.vds[indices[5]])
        ) * 1.0e-18
        output[indices] = base * value
    validate_device_correction(primitive, output)
    return output


def characterize_geometry_correction(
    correction: DeviceCorrectionConfig,
    adapter: DeviceCapacitanceAdapter,
    primitive: str,
    geometry: Geometry,
    raw_pex_path: Path,
    pex_subcircuit: str,
    output_root: Path,
) -> tuple[np.ndarray, float]:
    if correction.backend != "ngspice":
        raise ValueError("real device correction requires the ngspice backend")
    prepared = adapter.prepare_extracted_geometry(
        primitive,
        geometry,
        raw_pex_path,
        pex_subcircuit,
        output_root,
    )
    definition = adapter.definition(primitive)
    bias_grid = correction.biases[primitive]
    biases = tuple(
        Bias(*values)
        for values in itertools.product(
            bias_grid.vbs,
            bias_grid.vgs,
            bias_grid.vds,
        )
    )

    def characterize(item: tuple[int, Bias]) -> tuple[np.ndarray, float]:
        index, bias = item
        point_root = output_root / f"bias_{index:04d}"
        pex = extract_port_admittance(
            adapter.simulator,
            definition,
            bias,
            prepared.pex_path,
            prepared.pex_subcircuit,
            point_root / "pex",
        )
        aggregate = extract_port_admittance(
            adapter.simulator,
            definition,
            bias,
            prepared.aggregate_path,
            prepared.aggregate_subcircuit,
            point_root / "aggregate",
        )
        delta = pex.capacitance - aggregate.capacitance
        np.savez_compressed(
            point_root / "matrices.npz",
            pex_conductance=pex.conductance,
            pex_capacitance=pex.capacitance,
            aggregate_conductance=aggregate.conductance,
            aggregate_capacitance=aggregate.capacitance,
            device_capacitance_correction=delta,
        )
        return (
            delta,
            max(pex.frequency_consistency, aggregate.frequency_consistency),
        )

    with ThreadPoolExecutor(max_workers=correction.workers) as executor:
        samples = list(executor.map(characterize, enumerate(biases)))
    maximum_consistency = max(value[1] for value in samples)
    if maximum_consistency > correction.frequency_consistency:
        raise ValueError(
            f"{primitive} {geometry} device correction frequency inconsistency "
            f"{maximum_consistency:.3%} exceeds "
            f"{correction.frequency_consistency:.3%}"
        )
    port_count = len(PORTS[primitive])
    matrices = np.stack([value[0] for value in samples]).reshape(
        bias_grid.vbs.size,
        bias_grid.vgs.size,
        bias_grid.vds.size,
        port_count,
        port_count,
    )
    validate_device_correction(primitive, matrices)
    return matrices, maximum_consistency


def validate_device_correction(primitive: str, values: np.ndarray) -> None:
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError(f"{primitive} device correction matrices must be square")
    if not np.isfinite(values).all():
        raise ValueError(f"{primitive} device correction must be finite")
    scale = np.maximum(np.max(np.abs(values), axis=(-2, -1)), 1.0e-30)
    row_error = np.max(np.abs(values.sum(axis=-1)), axis=-1)
    column_error = np.max(np.abs(values.sum(axis=-2)), axis=-1)
    if np.any(row_error > scale * 1.0e-7) or np.any(
        column_error > scale * 1.0e-7
    ):
        raise ValueError(f"{primitive} device correction must conserve charge")
