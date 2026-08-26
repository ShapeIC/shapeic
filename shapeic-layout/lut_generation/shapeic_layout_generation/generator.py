from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .config import GenerationConfig, load_config
from .device_capacitance import Geometry
from .device_capacitance_adapter import DeviceCapacitanceAdapter
from .device_correction import (
    characterize_geometry_correction,
    correction_shape,
    create_device_correction_adapter,
    synthetic_device_correction,
)
from .extractor import extract_primitive
from .pcell import write_primitive_gds
from .synthetic import primitive_matrices
from .writer import write_archive


def generate(config_path: Path, *, force: bool = False) -> Path:
    config = load_config(config_path)
    if config.output_path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {config.output_path}; use --force to replace it"
        )
    adapter = (
        create_device_correction_adapter(config)
        if config.device_correction is not None
        and config.device_correction.backend == "ngspice"
        else None
    )
    generated = {
        primitive: _generate_primitive(config, primitive, adapter)
        for primitive in config.primitives
    }
    matrices = {primitive: value[0] for primitive, value in generated.items()}
    corrections = (
        {primitive: value[1] for primitive, value in generated.items()}
        if config.device_correction is not None
        else None
    )
    output = write_archive(
        config,
        matrices,
        corrections,
        force=force,
    )
    if not config.extractor.keep_work and config.extractor.work_directory.exists():
        shutil.rmtree(config.extractor.work_directory)
    return output


def _generate_primitive(
    config: GenerationConfig,
    primitive: str,
    adapter: DeviceCapacitanceAdapter | None,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray | None]:
    ports = config.port_order(primitive)
    port_count = len(ports)
    shape = (
        config.lengths.size,
        config.finger_widths.size,
        config.finger_counts.size,
        port_count,
        port_count,
    )
    conductance = np.empty(shape, dtype=np.float64)
    capacitance = np.empty(shape, dtype=np.float64)
    correction_config = config.device_correction
    correction = (
        synthetic_device_correction(
            primitive,
            config.lengths,
            config.finger_widths,
            correction_config,
        )
        if correction_config is not None
        and correction_config.backend == "synthetic"
        else (
            np.empty(
                correction_shape(
                    config.lengths,
                    config.finger_widths,
                    correction_config,
                    primitive,
                ),
                dtype=np.float64,
            )
            if correction_config is not None
            else None
        )
    )
    correction_nf_indices = (
        {
            int(value): index
            for index, value in enumerate(correction_config.finger_counts)
        }
        if correction_config is not None
        else {}
    )
    for li, length in enumerate(config.lengths):
        for wi, finger_width in enumerate(config.finger_widths):
            for ni, nf_value in enumerate(config.finger_counts):
                nf = int(nf_value)
                if config.extractor.backend == "synthetic":
                    g, c = primitive_matrices(
                        primitive, port_count, float(length), float(finger_width), nf
                    )
                else:
                    point_root = config.extractor.work_directory / (
                        f"{primitive}_l{li}_w{wi}_n{ni}"
                    )
                    gds_path = point_root / "layout.gds"
                    layout = config.primitive_layout(primitive)
                    if layout is None:
                        cell_name = write_primitive_gds(
                            primitive,
                            float(length),
                            float(finger_width),
                            nf,
                            gds_path,
                        )
                    else:
                        assert config.cellkit is not None
                        rendered = layout.render(
                            config.cellkit.geometry(
                                float(length), float(finger_width), nf
                            )
                        )
                        gds_path.parent.mkdir(parents=True, exist_ok=True)
                        rendered.component.write_gds(gds_path)
                        cell_name = rendered.cell_name
                    assert config.extractor.magic_rcfile is not None
                    extracted = extract_primitive(
                        gds_path,
                        cell_name,
                        ports,
                        magic_binary=config.extractor.magic_binary,
                        magic_rcfile=config.extractor.magic_rcfile,
                        magic_startup_commands=(
                            config.extractor.magic_startup_commands
                        ),
                        pex_normalizer=(
                            config.cellkit.technology.normalize_pex
                            if config.cellkit is not None
                            else None
                        ),
                        work_directory=point_root,
                        primitive=primitive,
                    )
                    g, c = extracted.conductance, extracted.capacitance
                    if (
                        correction_config is not None
                        and correction_config.backend == "ngspice"
                        and nf in correction_nf_indices
                    ):
                        if adapter is None or correction is None:
                            raise RuntimeError(
                                "real device correction adapter was not prepared"
                            )
                        values, _ = characterize_geometry_correction(
                            correction_config,
                            adapter,
                            primitive,
                            Geometry(float(length), float(finger_width), nf),
                            extracted.pex.spice_path,
                            extracted.pex.subcircuit_name,
                            point_root / "device_correction",
                        )
                        correction[
                            li,
                            wi,
                            correction_nf_indices[nf],
                        ] = values
                conductance[li, wi, ni] = g
                capacitance[li, wi, ni] = c
    return (conductance, capacitance), correction
