#!/usr/bin/env python3
"""Render and extract one CellKit macro with the selected PDK."""

from __future__ import annotations

import argparse
from pathlib import Path

from shapeic_layout_generation.config import load_config
from shapeic_layout_generation.macro_pex import generate_macro_pex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("macro")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--geometry",
        action="append",
        required=True,
        metavar="INSTANCE,LENGTH_M,FINGER_WIDTH_M,NF",
    )
    arguments = parser.parse_args()
    geometries = dict(_geometry(value, parser) for value in arguments.geometry)
    manifest = generate_macro_pex(
        load_config(arguments.config),
        arguments.macro,
        geometries,
        arguments.output_dir,
    )
    print(manifest.resolve())


def _geometry(value: str, parser: argparse.ArgumentParser):
    fields = value.split(",")
    if len(fields) != 4 or not fields[0]:
        parser.error("--geometry must be INSTANCE,LENGTH_M,FINGER_WIDTH_M,NF")
    try:
        geometry = (float(fields[1]), float(fields[2]), int(fields[3]))
    except ValueError:
        parser.error("--geometry contains a non-numeric coordinate")
    return fields[0], geometry


if __name__ == "__main__":
    main()
