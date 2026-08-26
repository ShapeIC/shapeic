from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapeic_layout_generation.device_capacitance import Admittance, Bias, Geometry
from shapeic_layout_generation.device_correction_diagnostic import (
    diagnose_device_correction,
    read_interpolated_device_correction,
    read_interpolated_interconnect,
)
from shapeic_layout_generation.writer import CORRECTION_AXIS_ORDER


class DeviceCorrectionDiagnosticTest(unittest.TestCase):
    def test_reads_and_interpolates_a_version_two_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, values, axes = self._write_archive(Path(directory))
            result = read_interpolated_device_correction(
                archive,
                "currentmirror",
                Geometry(2.0e-6, 3.0e-6, 2),
                Bias(-0.5, 0.3, 0.5),
            )

        expected = values.mean(axis=(0, 1, 2, 3, 4, 5))
        self.assertEqual(tuple(axis.size for axis in axes), (2, 2, 2, 2, 2, 2))
        self.assertEqual(result.ports, ("DOUT", "DREF", "S", "B"))
        np.testing.assert_allclose(result.capacitance, expected, rtol=1.0e-15)

    def test_reads_and_interpolates_the_local_interconnect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive, _, _ = self._write_archive(Path(directory))
            result = read_interpolated_interconnect(
                archive,
                "currentmirror",
                Geometry(2.0e-6, 3.0e-6, 2),
            )

        base = np.full((4, 4), -1.0)
        np.fill_diagonal(base, 3.0)
        np.testing.assert_allclose(result.conductance, 4.0 * base)
        np.testing.assert_allclose(result.capacitance, 4.0e-15 * base)

    def test_compares_interpolated_and_exact_corrections_and_keeps_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, values, _ = self._write_archive(root)
            interpolated = values.mean(axis=(0, 1, 2, 3, 4, 5))
            exact = 1.1 * interpolated
            zero = np.zeros_like(exact)
            adapter = SimpleNamespace(
                simulator_for=lambda _primitive: object(),
                validate_environment=lambda: None,
                prepare_geometry=lambda *_args: SimpleNamespace(
                    pex_path=Path("pex.spice"),
                    pex_subcircuit="pex",
                    aggregate_path=Path("aggregate.spice"),
                    aggregate_subcircuit="aggregate",
                ),
                definition=lambda _primitive: object(),
            )
            config = SimpleNamespace(
                pdk="test-pdk",
                layout_policy="test-policy",
                primitives=("currentmirror",),
                device_correction=SimpleNamespace(backend="ngspice"),
            )
            admittances = (
                Admittance(zero, exact, 0.002),
                Admittance(zero, zero, 0.001),
            )
            output = root / "diagnostic"
            with (
                patch(
                    "shapeic_layout_generation.device_correction_diagnostic."
                    "load_config",
                    return_value=config,
                ),
                patch(
                    "shapeic_layout_generation.device_correction_diagnostic."
                    "create_device_correction_adapter",
                    return_value=adapter,
                ),
                patch(
                    "shapeic_layout_generation.device_correction_diagnostic."
                    "extract_port_admittance",
                    side_effect=admittances,
                ),
            ):
                result = diagnose_device_correction(
                    Path("physical.toml"),
                    archive,
                    "currentmirror",
                    Geometry(2.0e-6, 3.0e-6, 2),
                    Bias(-0.5, 0.3, 0.5),
                    output,
                )

            summary = json.loads((output / "summary.json").read_text())
            with np.load(output / "matrices.npz") as matrices:
                interpolated_matrix = matrices["interpolated_delta_c"]
                exact_matrix = matrices["exact_delta_c"]
            comparison_exists = (output / "matrix_comparison.csv").is_file()

        self.assertAlmostEqual(result.matrix_relative_error, 1.0 / 11.0)
        self.assertAlmostEqual(result.exact_frequency_consistency, 0.002)
        self.assertEqual(summary["primitive"], "currentmirror")
        self.assertAlmostEqual(
            summary["metrics"]["matrix_relative_error"],
            1.0 / 11.0,
        )
        np.testing.assert_allclose(interpolated_matrix, interpolated)
        np.testing.assert_allclose(exact_matrix, exact)
        self.assertTrue(comparison_exists)

    @staticmethod
    def _write_archive(
        root: Path,
    ) -> tuple[Path, np.ndarray, tuple[np.ndarray, ...]]:
        archive_path = root / "physical-v2.npz"
        axes = (
            np.asarray((1.0e-6, 3.0e-6)),
            np.asarray((2.0e-6, 4.0e-6)),
            np.asarray((1.0, 3.0)),
            np.asarray((-1.0, 0.0)),
            np.asarray((0.2, 0.4)),
            np.asarray((0.3, 0.7)),
        )
        base = np.full((4, 4), -1.0)
        np.fill_diagonal(base, 3.0)
        values = np.empty((2, 2, 2, 2, 2, 2, 4, 4))
        for indices in np.ndindex(2, 2, 2, 2, 2, 2):
            scale = (
                1.0
                + sum(
                    (axis + 1) * index
                    for axis, index in enumerate(indices)
                )
            ) * 1.0e-15
            values[indices] = scale * base
        axis_members = tuple(
            f"primitive/correction/axes/{name}.npy"
            for name in CORRECTION_AXIS_ORDER
        )
        correction_member = "primitive/correction/capacitance.npy"
        interconnect_axes = tuple(
            f"primitive/axes/{name}.npy"
            for name in ("length", "finger_width", "nf")
        )
        interconnect_g = np.empty((2, 2, 2, 4, 4))
        interconnect_c = np.empty_like(interconnect_g)
        for indices in np.ndindex(2, 2, 2):
            scale = 1.0 + sum(
                (axis + 1) * index for axis, index in enumerate(indices)
            )
            interconnect_g[indices] = scale * base
            interconnect_c[indices] = scale * 1.0e-15 * base
        manifest = {
            "format": "shapeic-physical-lut",
            "version": 2,
            "pdk": "test-pdk",
            "layout_policy": "test-policy",
            "primitives": [
                {
                    "name": "currentmirror",
                    "port_order": ["DOUT", "DREF", "S", "B"],
                    "axis_order": ["length", "finger_width", "nf"],
                    "axes": [
                        {"name": name, "path": member}
                        for name, member in zip(
                            ("length", "finger_width", "nf"),
                            interconnect_axes,
                        )
                    ],
                    "conductance": "primitive/conductance.npy",
                    "capacitance": "primitive/capacitance.npy",
                    "device_capacitance_correction": {
                        "axis_order": list(CORRECTION_AXIS_ORDER),
                        "axes": [
                            {"name": name, "path": member}
                            for name, member in zip(
                                CORRECTION_AXIS_ORDER,
                                axis_members,
                            )
                        ],
                        "capacitance": correction_member,
                    },
                }
            ],
        }
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for member, axis in zip(axis_members, axes):
                archive.writestr(member, DeviceCorrectionDiagnosticTest._npy(axis))
            for member, axis in zip(interconnect_axes, axes[:3]):
                archive.writestr(member, DeviceCorrectionDiagnosticTest._npy(axis))
            archive.writestr(
                "primitive/conductance.npy",
                DeviceCorrectionDiagnosticTest._npy(interconnect_g),
            )
            archive.writestr(
                "primitive/capacitance.npy",
                DeviceCorrectionDiagnosticTest._npy(interconnect_c),
            )
            archive.writestr(
                correction_member,
                DeviceCorrectionDiagnosticTest._npy(values),
            )
        return archive_path, values, axes

    @staticmethod
    def _npy(values: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
