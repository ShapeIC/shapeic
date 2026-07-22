#!/usr/bin/env python3

import argparse
from pathlib import Path

from shapeic_lut_generation import generate, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a five-dimensional Shapeic MOS LUT")
    parser.add_argument("config", type=Path, help="TOML generation configuration")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the configured output file if it already exists",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    generate(config, force=args.force)


if __name__ == "__main__":
    main()
