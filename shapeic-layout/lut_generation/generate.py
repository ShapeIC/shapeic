#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from shapeic_layout_generation.generator import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Shapeic physical primitive LUT")
    parser.add_argument("config", type=Path)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    output = generate(arguments.config, force=arguments.force)
    print(output)


if __name__ == "__main__":
    main()

