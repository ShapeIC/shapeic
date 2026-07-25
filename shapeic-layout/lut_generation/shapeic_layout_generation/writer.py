from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .config import GenerationConfig, PORTS


AXIS_ORDER = ("length", "finger_width", "nf")


def write_archive(
    config: GenerationConfig,
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    force: bool,
) -> Path:
    output = config.output_path
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output}; use --force to replace it")
    if set(matrices) != set(config.primitives):
        raise ValueError("matrix result does not match configured primitives")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        manifest: dict[str, object] = {
            "format": "shapeic-physical-lut",
            "version": 1,
            "pdk": config.pdk,
            "layout_policy": config.layout_policy,
            "generator": "shapeic-layout/lut_generation",
            "units": {
                "length": "m",
                "finger_width": "m",
                "nf": "1",
                "conductance": "S",
                "capacitance": "F",
            },
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
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _write_npy(archive: zipfile.ZipFile, member: str, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    archive.writestr(member, buffer.getvalue())
