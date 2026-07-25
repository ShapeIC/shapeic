from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .config import PORTS, GenerationConfig, load_config
from .extractor import run_magic
from .pcell import write_primitive_gds
from .synthetic import primitive_matrices
from .writer import write_archive


def generate(config_path: Path, *, force: bool = False) -> Path:
    config = load_config(config_path)
    matrices = {
        primitive: _generate_primitive(config, primitive)
        for primitive in config.primitives
    }
    output = write_archive(config, matrices, force=force)
    if not config.extractor.keep_work and config.extractor.work_directory.exists():
        shutil.rmtree(config.extractor.work_directory)
    return output


def _generate_primitive(
    config: GenerationConfig, primitive: str
) -> tuple[np.ndarray, np.ndarray]:
    port_count = len(PORTS[primitive])
    shape = (
        config.lengths.size,
        config.finger_widths.size,
        config.finger_counts.size,
        port_count,
        port_count,
    )
    conductance = np.empty(shape, dtype=np.float64)
    capacitance = np.empty(shape, dtype=np.float64)
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
                    cell_name = write_primitive_gds(
                        primitive,
                        float(length),
                        float(finger_width),
                        nf,
                        gds_path,
                    )
                    assert config.extractor.magic_rcfile is not None
                    g, c = run_magic(
                        gds_path,
                        cell_name,
                        PORTS[primitive],
                        magic_binary=config.extractor.magic_binary,
                        magic_rcfile=config.extractor.magic_rcfile,
                        work_directory=point_root,
                        primitive=primitive,
                    )
                conductance[li, wi, ni] = g
                capacitance[li, wi, ni] = c
    return conductance, capacitance
