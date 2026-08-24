from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .config import GenerationConfig, PORTS
from .device_correction import validate_device_correction


AXIS_ORDER = ("length", "finger_width", "nf")
CORRECTION_AXIS_ORDER = (
    "length",
    "finger_width",
    "nf",
    "vbs",
    "vgs",
    "vds",
)


def write_archive(
    config: GenerationConfig,
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    device_corrections: dict[str, np.ndarray | None] | None = None,
    *,
    force: bool,
) -> Path:
    output = config.output_path
    if output.exists() and not force:
        raise FileExistsError(
            f"output already exists: {output}; use --force to replace it"
        )
    if set(matrices) != set(config.primitives):
        raise ValueError("matrix result does not match configured primitives")
    correction_config = config.device_correction
    if (correction_config is None) != (device_corrections is None):
        raise ValueError("device correction result does not match configuration")
    if device_corrections is not None and set(device_corrections) != set(
        config.primitives
    ):
        raise ValueError(
            "device correction result does not match configured primitives"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        units = {
            "length": "m",
            "finger_width": "m",
            "nf": "1",
            "conductance": "S",
            "capacitance": "F",
        }
        if correction_config is not None:
            units.update({"vbs": "V", "vgs": "V", "vds": "V"})
        manifest: dict[str, object] = {
            "format": "shapeic-physical-lut",
            "version": 2 if correction_config is not None else 1,
            "pdk": config.pdk,
            "layout_policy": config.layout_policy,
            "generator": "shapeic-layout/lut_generation",
            "units": units,
            "nf_query": "positive-integer-with-linear-interpolation",
            "primitives": [],
        }
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for index, primitive in enumerate(config.primitives):
                root = f"primitives/{index}"
                axis_paths = []
                for name, values in (
                    ("length", config.lengths),
                    ("finger_width", config.finger_widths),
                    ("nf", config.finger_counts),
                ):
                    member = f"{root}/axes/{name}.npy"
                    _write_npy(archive, member, values.astype(np.float64))
                    axis_paths.append({"name": name, "path": member})
                conductance, capacitance = matrices[primitive]
                expected = (
                    config.lengths.size,
                    config.finger_widths.size,
                    config.finger_counts.size,
                    len(PORTS[primitive]),
                    len(PORTS[primitive]),
                )
                if conductance.shape != expected or capacitance.shape != expected:
                    raise ValueError(f"{primitive} matrices must have shape {expected}")
                g_path = f"{root}/conductance.npy"
                c_path = f"{root}/capacitance.npy"
                _write_npy(archive, g_path, conductance.astype(np.float64))
                _write_npy(archive, c_path, capacitance.astype(np.float64))
                manifest["primitives"].append(
                    {
                        "name": primitive,
                        "port_order": list(PORTS[primitive]),
                        "axis_order": list(AXIS_ORDER),
                        "axes": axis_paths,
                        "conductance": g_path,
                        "capacitance": c_path,
                    }
                )
                if correction_config is not None and device_corrections is not None:
                    correction = device_corrections[primitive]
                    if correction is None:
                        raise ValueError(
                            f"{primitive} device correction result is missing"
                        )
                    bias = correction_config.biases[primitive]
                    correction_axes = (
                        config.lengths,
                        config.finger_widths,
                        correction_config.finger_counts,
                        bias.vbs,
                        bias.vgs,
                        bias.vds,
                    )
                    expected_correction = tuple(
                        axis.size for axis in correction_axes
                    ) + (len(PORTS[primitive]), len(PORTS[primitive]))
                    if correction.shape != expected_correction:
                        raise ValueError(
                            f"{primitive} device correction must have shape "
                            f"{expected_correction}"
                        )
                    validate_device_correction(primitive, correction)
                    correction_root = f"{root}/device_capacitance_correction"
                    correction_axis_paths = []
                    for name, values in zip(
                        CORRECTION_AXIS_ORDER,
                        correction_axes,
                    ):
                        member = f"{correction_root}/axes/{name}.npy"
                        _write_npy(archive, member, values.astype(np.float64))
                        correction_axis_paths.append({"name": name, "path": member})
                    correction_path = f"{correction_root}/capacitance.npy"
                    _write_npy(
                        archive,
                        correction_path,
                        correction.astype(np.float64),
                    )
                    manifest["primitives"][-1][
                        "device_capacitance_correction"
                    ] = {
                        "definition": (
                            "pex_mos_only_minus_aggregate_compact_model"
                        ),
                        "axis_order": list(CORRECTION_AXIS_ORDER),
                        "axes": correction_axis_paths,
                        "capacitance": correction_path,
                        "simulator": correction_config.binary,
                        "library_section": correction_config.library_section,
                        "temperature_c": correction_config.temperature_c,
                        "frequencies_hz": list(
                            correction_config.frequencies_hz
                        ),
                    }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_npy(archive: zipfile.ZipFile, member: str, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    archive.writestr(member, buffer.getvalue())
