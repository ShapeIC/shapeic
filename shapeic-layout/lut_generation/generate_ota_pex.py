#!/usr/bin/env python3
"""Generate and validate one full four-transistor OTA PEX reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapeic_layout_generation.config import load_config
from shapeic_layout_generation.extractor import write_magic_pex
from shapeic_layout_generation.ota_pex import OTA_PORTS, prepare_ota_pex
from shapeic_layout_generation.pcell import write_ota_gds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a routed IHP SG13G2 four-transistor OTA and extract its PEX."
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
    if config.extractor.backend != "magic":
        parser.error("the OTA PEX reference requires extraction.backend='magic'")
    assert config.extractor.magic_rcfile is not None

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    gds_path = output_dir / "ota_4t.gds"
    cell_name = write_ota_gds(
        arguments.diff_length,
        arguments.diff_wf,
        arguments.diff_nf,
        arguments.mirror_length,
        arguments.mirror_wf,
        arguments.mirror_nf,
        gds_path,
    )
    artifacts_path = output_dir / "artifacts.json"
    artifacts_path.write_text(
        json.dumps(
            {
                "format": "shapeic-ota-pex-artifacts",
                "version": 1,
                "layout_policy": config.layout_policy,
                "cell_name": cell_name,
                "gds_path": str(gds_path.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    magic = write_magic_pex(
        gds_path,
        cell_name,
        magic_binary=config.extractor.magic_binary,
        magic_rcfile=config.extractor.magic_rcfile,
        work_directory=output_dir / "magic",
    )
    normalized_pex = output_dir / "ota_4t.pex.spice"
    topology = prepare_ota_pex(
        magic.spice_path,
        normalized_pex,
        expected_subcircuit=magic.subcircuit_name,
    )
    manifest = {
        "format": "shapeic-ota-pex",
        "version": 1,
        "layout_policy": config.layout_policy,
        "cell_name": cell_name,
        "subcircuit_name": topology.subcircuit_name,
        "port_order": list(OTA_PORTS),
        "gds_path": str(gds_path.resolve()),
        "pex_path": str(normalized_pex.resolve()),
        "raw_pex_path": str(magic.spice_path),
        "magic_script_path": str(magic.script_path),
        "magic_stdout_path": str(magic.stdout_path),
        "magic_stderr_path": str(magic.stderr_path),
        "transistor_count": topology.transistor_count,
        "resistor_count": topology.resistor_count,
        "capacitor_count": topology.capacitor_count,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
