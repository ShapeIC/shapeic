#!/usr/bin/env python3
"""Compare one interpolated device correction against exact characterization."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from shapeic_layout_generation.device_capacitance import Bias, Geometry
from shapeic_layout_generation.device_correction_diagnostic import (
    diagnose_device_correction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize one primitive geometry and bias exactly, then compare "
            "its device-capacitance correction against a physical LUT v2 query."
        )
    )
    parser.add_argument("config", type=Path, help="physical LUT generation TOML")
    parser.add_argument("physical_lut", type=Path, help="physical LUT v2 archive")
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--length", type=float, required=True, help="length in meters")
    parser.add_argument(
        "--finger-width", type=float, required=True, help="finger width in meters"
    )
    parser.add_argument("--nf", type=int, required=True)
    parser.add_argument("--vbs", type=float, required=True)
    parser.add_argument("--vgs", type=float, required=True)
    parser.add_argument("--vds", type=float, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="new artifact directory (defaults below target)",
    )
    arguments = parser.parse_args()
    output_root = arguments.output_root
    if output_root is None:
        repository = Path(__file__).resolve().parents[2]
        output_root = (
            repository
            / "target"
            / "shapeic-device-correction-diagnostic"
            / str(time.time_ns())
        )
    result = diagnose_device_correction(
        arguments.config,
        arguments.physical_lut,
        arguments.primitive,
        Geometry(arguments.length, arguments.finger_width, arguments.nf),
        Bias(arguments.vbs, arguments.vgs, arguments.vds),
        output_root,
    )
    print(f"Artifacts: {result.output_root}")
    print(f"Matrix relative error: {result.matrix_relative_error:.6%}")
    print(f"Maximum absolute error: {result.maximum_absolute_error:.6e} F")
    print(
        "Exact frequency consistency: "
        f"{result.exact_frequency_consistency:.6%}"
    )
    print(
        "Charge residual (interpolated/exact): "
        f"{result.interpolated_charge_residual:.6e} / "
        f"{result.exact_charge_residual:.6e}"
    )


if __name__ == "__main__":
    main()
