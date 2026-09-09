#!/usr/bin/env python3
"""Compatibility entrypoint for the CellKit four-transistor OTA macro."""

from __future__ import annotations

import argparse
from pathlib import Path

from shapeic_layout_generation.config import load_config
from shapeic_layout_generation.macro_pex import generate_macro_pex


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a routed CellKit four-transistor OTA and extract its PEX."
    )
    parser.add_argument("config", type=Path, help="physical LUT generation TOML")
    parser.add_argument("output_dir", type=Path, help="artifact directory")
    parser.add_argument("--diff-length", type=float, required=True)
    parser.add_argument("--diff-wf", type=float, required=True)
    parser.add_argument("--diff-nf", type=int, required=True)
    parser.add_argument("--mirror-length", type=float, required=True)
    parser.add_argument("--mirror-wf", type=float, required=True)
    parser.add_argument("--mirror-nf", type=int, required=True)
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    manifest_path = generate_macro_pex(
        config,
        "ota_4t",
        {
            "xdp": (
                arguments.diff_length,
                arguments.diff_wf,
                arguments.diff_nf,
            ),
            "xcm": (
                arguments.mirror_length,
                arguments.mirror_wf,
                arguments.mirror_nf,
            ),
        },
        arguments.output_dir,
        manifest_format="shapeic-ota-pex",
    )
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
