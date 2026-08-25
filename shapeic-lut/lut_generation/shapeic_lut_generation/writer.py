from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import GenerationConfig


AXIS_ORDER = ["length", "vbs", "vgs", "vds", "finger_width"]


class LutStorage:
    def __init__(self, config: GenerationConfig, root: Path):
        self.config = config
        self.root = root
        self.model_root = root / "models" / "0"
        self.axis_root = self.model_root / "axes"
        self.parameter_root = self.model_root / "parameters"
        self.axis_root.mkdir(parents=True)
        self.parameter_root.mkdir(parents=True)

        self.axes = config.sweep.axes()
        self.shape = tuple(self.axes[name].size for name in AXIS_ORDER)
        for name, values in self.axes.items():
            np.save(self.axis_root / f"{name}.npy", values.astype(np.float64))
        self.parameters: dict[str, np.memmap] = {}
        for name in config.output_parameters():
            array = np.lib.format.open_memmap(
                self.parameter_root / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=self.shape,
            )
            array[:] = np.nan
            self.parameters[name] = array

    def put(self, indices: tuple[int, int, int], values: Mapping[str, np.ndarray]) -> None:
        length_index, vbs_index, width_index = indices
        expected = self.shape[2:4]
        if set(values) != set(self.parameters):
            raise ValueError("simulation result parameter set does not match the configured set")
        for name, block in values.items():
            if block.shape != expected:
                raise ValueError(
                    f"parameter '{name}' has block shape {block.shape}, expected {expected}"
                )
            self.parameters[name][length_index, vbs_index, :, :, width_index] = block

    def validate_and_flush(self) -> None:
        for name, array in self.parameters.items():
            if not np.isfinite(array).all():
                raise ValueError(f"parameter '{name}' is incomplete or contains non-finite values")
            array.flush()

    def write_archive(self, force: bool) -> None:
        output = self.config.output_path
        if output.exists() and not force:
            raise FileExistsError(f"output already exists: {output}; use --force to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
                for name in AXIS_ORDER:
                    archive.write(
                        self.axis_root / f"{name}.npy",
                        arcname=f"models/0/axes/{name}.npy",
                    )
                for name in self.config.output_parameters():
                    archive.write(
                        self.parameter_root / f"{name}.npy",
                        arcname=f"models/0/parameters/{name}.npy",
                    )
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)

    def _manifest(self) -> dict[str, object]:
        device = self.config.device
        manifest: dict[str, object] = {
            "format": "shapeic-lut",
            "version": 2,
            "description": self.config.description,
            "simulator": "ngspice",
            "temperature_c": self.config.simulator.temperature_c,
            "models": [
                {
                    "name": device.name,
                    "axis_order": AXIS_ORDER,
                    "axes": [
                        {"name": name, "path": f"models/0/axes/{name}.npy"}
                        for name in AXIS_ORDER
                    ],
                    "parameters": [
                        {"name": name, "path": f"models/0/parameters/{name}.npy"}
                        for name in self.config.output_parameters()
                    ],
                    "device_parameters": {"nf": float(device.nf)},
                }
            ],
        }
        if self.config.pdk is not None:
            manifest.update(
                {
                    "pdk": self.config.pdk.name,
                    "corner": self.config.pdk.corner,
                    "nominal_voltage": self.config.pdk.nominal_voltage,
                }
            )
            if self.config.pdk.revision is not None:
                manifest["pdk_revision"] = self.config.pdk.revision
        return manifest
