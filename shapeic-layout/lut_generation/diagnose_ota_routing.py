#!/usr/bin/env python3
"""Separate local primitive interpolation from complete-OTA routing parasitics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from shapeic_layout_generation.device_capacitance import Geometry
from shapeic_layout_generation.ota_routing_diagnostic import diagnose_ota_routing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare interpolated and exact local primitive interconnect matrices "
            "against the explicit RC matrix of a complete OTA PEX."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("physical_lut", type=Path)
    parser.add_argument("ota_pex_manifest", type=Path)
    parser.add_argument("--diff-length", type=float, required=True)
    parser.add_argument("--diff-finger-width", type=float, required=True)
    parser.add_argument("--diff-nf", type=int, required=True)
    parser.add_argument("--mirror-length", type=float, required=True)
    parser.add_argument("--mirror-finger-width", type=float, required=True)
    parser.add_argument("--mirror-nf", type=int, required=True)
    parser.add_argument(
        "--diff-bulk-node",
        choices=("IBIAS", "0"),
        default="IBIAS",
        help="node used by the verification testbench for the NMOS bulk",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    output_root = arguments.output_root
    if output_root is None:
        repository = Path(__file__).resolve().parents[2]
        output_root = (
            repository
            / "target"
            / "shapeic-ota-routing-diagnostic"
            / str(time.time_ns())
        )
    result = diagnose_ota_routing(
        arguments.config,
        arguments.physical_lut,
        arguments.ota_pex_manifest,
        Geometry(
            arguments.diff_length,
            arguments.diff_finger_width,
            arguments.diff_nf,
        ),
        Geometry(
            arguments.mirror_length,
            arguments.mirror_finger_width,
            arguments.mirror_nf,
        ),
        arguments.diff_bulk_node,
        output_root,
    )
    print(f"Artifacts: {result.output_root}")
    for primitive, error in result.local_capacitance_interpolation_error.items():
        print(f"Local C interpolation error ({primitive}): {error:.6%}")
    for primitive, delta in result.local_bulk_binding_delta.items():
        print(f"Bulk-binding matrix delta ({primitive}): {delta:.6%}")
    print(f"LUT-local vs. full OTA C error: {result.lut_local_vs_full_error:.6%}")
    print(
        "Exact bound-local vs. full OTA C error: "
        f"{result.exact_local_vs_full_error:.6%}"
    )
    print(
        "Maximum routing/context residual: "
        f"{result.routing_maximum_absolute_capacitance:.6e} F"
    )


if __name__ == "__main__":
    main()
