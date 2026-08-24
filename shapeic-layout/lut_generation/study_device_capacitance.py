#!/usr/bin/env python3
"""Validate the multifinger MOS capacitance correction independently."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shapeic_layout_generation.device_capacitance_study import (
    StudyExecutionError,
    run_study,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MOS-only PCell PEX against aggregate MOS devices and "
            "validate interpolation of their capacitance difference."
        )
    )
    parser.add_argument("config", type=Path, help="device-capacitance study TOML")
    parser.add_argument(
        "--stage",
        choices=("1", "2", "auto"),
        default="auto",
        help=(
            "run Stage 1 only, force both stages, or run Stage 2 only when "
            "Stage 1 passes"
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_study(arguments.config, stage=arguments.stage)
    except StudyExecutionError as error:
        print(f"Artifacts: {error.output_root}", file=sys.stderr)
        print(f"Error: {error.cause}", file=sys.stderr)
        return 1

    print(f"Artifacts: {result.output_root}")
    print(f"Stage 1: {'PASS' if result.stage_one_passed else 'FAIL'}")
    if result.stage_two_ran:
        print(f"Stage 2: {'PASS' if result.stage_two_passed else 'FAIL'}")
    else:
        print("Stage 2: not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
