#!/usr/bin/env python3
"""Plot the Shapeic and NGSpice AC sweeps from one verification artifact."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sweep:
    frequency_hz: list[float]
    response_real: list[float]
    response_imag: list[float]
    gain_db: list[float]
    phase_deg: list[float]


def read_shapeic(path: Path) -> Sweep:
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        required = {
            "frequency_hz",
            "response_real",
            "response_imag",
            "gain_db",
            "phase_deg",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        columns = {name: [] for name in required}
        for row_number, row in enumerate(reader, start=2):
            for name in required:
                columns[name].append(_finite_float(row[name], path, row_number, name))
    return _validated_sweep(path, columns)


def read_ngspice(path: Path) -> Sweep:
    lines = [
        line.split()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"{path} contains no AC samples")

    aliases = {
        "frequency_hz": ("frequency_hz", "frequency"),
        "response_real": ("response_real",),
        "response_imag": ("response_imag",),
        "gain_db": ("gain_db",),
        "phase_deg": ("phase_deg",),
    }
    header = lines[0]
    indices = {}
    for name, candidates in aliases.items():
        index = next((header.index(item) for item in candidates if item in header), None)
        if index is None:
            raise ValueError(f"{path} is missing column '{name}'")
        indices[name] = index

    columns = {name: [] for name in aliases}
    for row_number, row in enumerate(lines[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{row_number} has {len(row)} values; expected {len(header)}"
            )
        for name, index in indices.items():
            columns[name].append(_finite_float(row[index], path, row_number, name))
    return _validated_sweep(path, columns)


def _finite_float(value: str, path: Path, row: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"{path}:{row} has invalid value {value!r} in '{column}'"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{row} has non-finite value in '{column}'")
    return parsed


def _validated_sweep(path: Path, columns: dict[str, list[float]]) -> Sweep:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError(f"{path} has inconsistent or empty columns")
    frequencies = columns["frequency_hz"]
    if frequencies[0] <= 0.0 or any(
        upper <= lower for lower, upper in zip(frequencies, frequencies[1:])
    ):
        raise ValueError(f"{path} frequencies must be positive and strictly increasing")
    return Sweep(
        frequency_hz=frequencies,
        response_real=columns["response_real"],
        response_imag=columns["response_imag"],
        gain_db=columns["gain_db"],
        phase_deg=columns["phase_deg"],
    )


def downward_crossing(
    frequencies: list[float], values: list[float], target: float
) -> float | None:
    for frequency_0, frequency_1, value_0, value_1 in zip(
        frequencies, frequencies[1:], values, values[1:]
    ):
        if value_0 >= target >= value_1 and value_0 != value_1:
            weight = (target - value_0) / (value_1 - value_0)
            log_frequency = math.log10(frequency_0) + weight * (
                math.log10(frequency_1) - math.log10(frequency_0)
            )
            return 10.0**log_frequency
    return None


def _frequency_label(value: float | None) -> str:
    if value is None:
        return "none"
    units = ((1.0e9, "GHz"), (1.0e6, "MHz"), (1.0e3, "kHz"))
    for scale, unit in units:
        if value >= scale:
            return f"{value / scale:.3g} {unit}"
    return f"{value:.3g} Hz"


def plot_comparison(
    shapeic: Sweep, ngspice: Sweep, output: Path, show: bool, title: str
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required; install it with 'python3 -m pip install matplotlib'"
        ) from error

    figure, (gain_axis, phase_axis) = plt.subplots(
        2, 1, figsize=(10.0, 7.2), sharex=True
    )
    curves = (
        ("Shapeic MNA", shapeic, "#1565c0", "-"),
        ("NGSpice", ngspice, "#c62828", "--"),
    )
    for name, sweep, color, linestyle in curves:
        bandwidth = downward_crossing(
            sweep.frequency_hz, sweep.gain_db, sweep.gain_db[0] - 3.0
        )
        unity = downward_crossing(sweep.frequency_hz, sweep.gain_db, 0.0)
        label = (
            f"{name}: f3dB={_frequency_label(bandwidth)}, "
            f"UGF={_frequency_label(unity)}, "
            f"end={sweep.gain_db[-1]:.2f} dB"
        )
        gain_axis.semilogx(
            sweep.frequency_hz,
            sweep.gain_db,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
        phase_axis.semilogx(
            sweep.frequency_hz,
            sweep.phase_deg,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=name,
        )
        if bandwidth is not None:
            gain_axis.axvline(
                bandwidth, color=color, linestyle=":", linewidth=1.0, alpha=0.8
            )
        if unity is not None:
            gain_axis.axvline(
                unity, color=color, linestyle="-.", linewidth=1.0, alpha=0.8
            )

    gain_axis.axhline(0.0, color="#333333", linewidth=0.9)
    gain_axis.set_ylabel("Gain [dB]")
    gain_axis.set_title(title)
    gain_axis.legend(loc="best", fontsize=9)
    gain_axis.grid(True, which="both", linewidth=0.5, alpha=0.35)

    phase_axis.set_xlabel("Frequency [Hz]")
    phase_axis.set_ylabel("Phase [deg]")
    phase_axis.legend(loc="best", fontsize=9)
    phase_axis.grid(True, which="both", linewidth=0.5, alpha=0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Shapeic and NGSpice AC sweeps from a verification run."
    )
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help="point artifact directory containing shapeic_ac.csv and ngspice_ac.tsv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output PNG path (default: ARTIFACT_DIR/ac_comparison.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the matplotlib window",
    )
    parser.add_argument(
        "--shapeic-file",
        default="shapeic_ac.csv",
        help="Shapeic CSV filename inside ARTIFACT_DIR",
    )
    parser.add_argument(
        "--title",
        default="4T OTA electrical AC comparison",
        help="plot title",
    )
    arguments = parser.parse_args()
    artifact_dir = arguments.artifact_dir
    output = arguments.output or artifact_dir / "ac_comparison.png"

    try:
        shapeic = read_shapeic(artifact_dir / arguments.shapeic_file)
        ngspice = read_ngspice(artifact_dir / "ngspice_ac.tsv")
        plot_comparison(shapeic, ngspice, output, arguments.show, arguments.title)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
