#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from shapeic_lut_generation import load_config
from shapeic_lut_generation.probe import (
    ProbeError,
    ProbePoint,
    run_probe,
    write_golden_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one electrical MOS point against an independent AC multiport extraction")
    parser.add_argument("config", type=Path, help="TOML generation configuration")
    parser.add_argument("--length", type=float, required=True, help="channel length in metres")
    parser.add_argument("--finger-width", type=float, required=True, help="width per finger in metres")
    parser.add_argument("--vbs", type=float, required=True, help="bulk-to-source voltage")
    parser.add_argument("--vgs", type=float, required=True, help="gate-to-source voltage")
    parser.add_argument("--vds", type=float, required=True, help="drain-to-source voltage")
    parser.add_argument("--output-root", type=Path, help="artifact parent directory")
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="write a deterministic golden reference from a passing probe",
    )
    parser.add_argument(
        "--force-reference",
        action="store_true",
        help="replace an existing --reference-output file",
    )
    args = parser.parse_args()
    if args.force_reference and args.reference_output is None:
        parser.error("--force-reference requires --reference-output")
    config = load_config(args.config)
    point = ProbePoint(
        args.length, args.finger_width, args.vbs, args.vgs, args.vds
    )
    try:
        artifacts, report = run_probe(
            config,
            point,
            args.output_root,
        )
    except ProbeError as error:
        print(f"Artifacts: {error.artifacts}")
        print(f"Probe: FAIL ({error})", file=sys.stderr)
        return 1
    if args.reference_output is not None:
        try:
            write_golden_reference(
                args.reference_output,
                config,
                point,
                report,
                force=args.force_reference,
            )
        except (FileExistsError, ValueError) as error:
            print(f"Reference: FAIL ({error})", file=sys.stderr)
            return 1
        print(f"Reference: {args.reference_output.resolve()}")
    print(f"Artifacts: {artifacts}")
    print(f"Probe: {report['status'].upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
