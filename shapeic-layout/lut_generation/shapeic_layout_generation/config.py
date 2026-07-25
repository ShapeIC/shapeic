from __future__ import annotations

import math
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PORTS = {
    "simplediffpair": ("DP", "DN", "GP", "GN", "S", "B"),
    "currentmirror": ("DOUT", "DREF", "S", "B"),
}


@dataclass(frozen=True)
class ExtractorConfig:
    backend: str
    magic_binary: str
    magic_rcfile: Path | None
    work_directory: Path
    keep_work: bool


@dataclass(frozen=True)
class GenerationConfig:
    source_path: Path
    output_path: Path
    pdk: str
    layout_policy: str
    lengths: np.ndarray
    finger_widths: np.ndarray
    finger_counts: np.ndarray
    primitives: tuple[str, ...]
    extractor: ExtractorConfig


def load_config(path: Path) -> GenerationConfig:
    source = path.resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    output = _table(raw, "output")
    sweep = _table(raw, "sweep")
    extraction = _table(raw, "extraction")
    backend = str(extraction.get("backend", "magic")).lower()
    if backend not in {"magic", "synthetic"}:
        raise ValueError("extraction.backend must be 'magic' or 'synthetic'")

    primitives = tuple(str(value).lower() for value in raw.get("primitives", PORTS))
    if not primitives or len(primitives) != len(set(primitives)):
        raise ValueError("primitives must be a non-empty list without duplicates")
    unknown = sorted(set(primitives) - set(PORTS))
    if unknown:
        raise ValueError(f"unsupported primitives: {', '.join(unknown)}")

    config = GenerationConfig(
        source_path=source,
        output_path=_path(str(output["path"]), source.parent),
        pdk=str(raw.get("pdk", "ihp-sg13g2")),
        layout_policy=str(
            raw.get("layout_policy", "symmetric-adjacent-with-edge-dummies-v3")
        ),
        lengths=np.asarray(sweep["length"], dtype=np.float64) * 1.0e-6,
        finger_widths=np.asarray(sweep["finger_width"], dtype=np.float64) * 1.0e-6,
        finger_counts=np.asarray(sweep["nf"], dtype=np.float64),
        primitives=primitives,
        extractor=ExtractorConfig(
            backend=backend,
            magic_binary=str(extraction.get("magic_binary", "magic")),
            magic_rcfile=(
                _path(str(extraction["magic_rcfile"]), source.parent)
                if extraction.get("magic_rcfile")
                else None
            ),
            work_directory=_path(
                str(extraction.get("work_directory", "../work")), source.parent
            ),
            keep_work=bool(extraction.get("keep_work", False)),
        ),
    )
    _validate(config)
    return config


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"path contains an undefined environment variable: {value}")
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _validate(config: GenerationConfig) -> None:
    if config.output_path.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    if not config.pdk or not config.layout_policy:
        raise ValueError("pdk and layout_policy must not be empty")
    for name, axis in (
        ("length", config.lengths),
        ("finger_width", config.finger_widths),
        ("nf", config.finger_counts),
    ):
        if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
            raise ValueError(f"{name} must be a non-empty finite vector")
        if axis.size > 1 and not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} must be strictly increasing")
    if np.any(config.lengths <= 0) or np.any(config.finger_widths <= 0):
        raise ValueError("length and finger_width must be positive")
    if np.any(config.finger_counts < 1) or np.any(config.finger_counts > 20):
        raise ValueError("nf samples must be in [1, 20]")
    if np.any(np.mod(config.finger_counts, 1.0) != 0.0):
        raise ValueError("nf samples must be integers")
    if not all(math.isfinite(float(value)) for value in config.finger_counts):
        raise ValueError("nf samples must be finite")
    if config.extractor.backend == "magic":
        if shutil.which(config.extractor.magic_binary) is None:
            raise FileNotFoundError(
                f"Magic binary not found: {config.extractor.magic_binary}"
            )
        if config.extractor.magic_rcfile is None:
            raise ValueError("magic_rcfile is required by the magic backend")
        if not config.extractor.magic_rcfile.is_file():
            raise FileNotFoundError(
                f"Magic rcfile not found: {config.extractor.magic_rcfile}"
            )
