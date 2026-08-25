import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shapeic_lut_generation.config import CapacitanceConvention
from shapeic_lut_generation.ngspice import normalize_capacitance_parameters
from shapeic_lut_generation.probe import (
    combined_matrix,
    charge_residual,
    intrinsic_matrix,
    matrix_relative_error,
    ProbePoint,
    write_golden_reference,
)


class CapacitanceConventionTests(unittest.TestCase):
    def test_signed_nodal_normalization_only_negates_mutual_coefficients(self):
        values = {
            name: np.asarray(float(index + 1))
            for index, name in enumerate(
                [
                    "cgg",
                    "cgd",
                    "cgs",
                    "cdg",
                    "cdd",
                    "cds",
                    "csg",
                    "csd",
                    "css",
                    "cgsol",
                    "cgdol",
                    "cjs",
                    "cjd",
                ]
            )
        }
        normalized = normalize_capacitance_parameters(
            values, CapacitanceConvention.SIGNED_NODAL
        )
        for name in ["cgd", "cgs", "cdg", "cds", "csg", "csd"]:
            self.assertEqual(normalized[name], -values[name])
        for name in ["cgg", "cdd", "css", "cgsol", "cgdol", "cjs", "cjd"]:
            self.assertEqual(normalized[name], values[name])

    def test_compact_mutual_normalization_preserves_all_values(self):
        values = {"cgd": np.asarray(2.0), "cgg": np.asarray(3.0)}
        normalized = normalize_capacitance_parameters(
            values, CapacitanceConvention.COMPACT_MUTUAL
        )
        np.testing.assert_array_equal(normalized["cgd"], values["cgd"])
        np.testing.assert_array_equal(normalized["cgg"], values["cgg"])


class ProbeMatrixTests(unittest.TestCase):
    def test_reconstructs_a_charge_conserving_matrix_and_adds_extrinsics(self):
        intrinsic = intrinsic_matrix(
            {
                "cgg": 10.0,
                "cgd": 2.0,
                "cgs": 3.0,
                "cdg": 1.0,
                "cdd": 8.0,
                "cds": 2.0,
                "csg": 2.0,
                "csd": 1.0,
                "css": 7.0,
            }
        )
        combined = combined_matrix(
            intrinsic, {"cgsol": 0.5, "cgdol": 0.4, "cjs": 0.3, "cjd": 0.2}
        )
        self.assertLess(charge_residual(intrinsic), 1.0e-15)
        self.assertLess(charge_residual(combined), 1.0e-15)
        self.assertAlmostEqual(combined[0, 0] - intrinsic[0, 0], 0.9)
        self.assertAlmostEqual(combined[0, 1] - intrinsic[0, 1], -0.4)
        self.assertAlmostEqual(combined[2, 3] - intrinsic[2, 3], -0.3)

    def test_matrix_error_is_normalized_by_the_measured_matrix(self):
        observed = np.eye(4)
        reference = observed * 1.01
        self.assertAlmostEqual(matrix_relative_error(reference, observed), 0.01)


class GoldenReferenceTests(unittest.TestCase):
    def test_writes_only_the_deterministic_portable_probe_subset(self):
        config = SimpleNamespace(
            pdk=SimpleNamespace(
                name="sky130A", revision="revision", corner="tt"
            ),
            simulator=SimpleNamespace(temperature_c=27.0),
            device=SimpleNamespace(
                name="test_nmos",
                capacitance_convention=CapacitanceConvention.SIGNED_NODAL,
            ),
        )
        point = ProbePoint(0.3e-6, 0.42e-6, 0.0, 0.6, 0.6)
        report = {
            "status": "pass",
            "frequencies_hz": [1.0e6, 1.0e7],
            "nf_results": [
                {
                    "nf": 1,
                    "canonical_parameters": {"id": 1.0e-6},
                    "ac_capacitance_f": [np.eye(4).tolist(), np.eye(4).tolist()],
                    "raw_parameters": {"native": 1.0},
                    "status": "pass",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            write_golden_reference(path, config, point, report)
            reference = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reference["format"], "shapeic-electrical-reference")
            self.assertEqual(reference["version"], 1)
            self.assertEqual(reference["pdk"], "sky130A")
            self.assertEqual(reference["point"]["finger_width"], 0.42e-6)
            self.assertNotIn("raw_parameters", reference["nf_results"][0])
            with self.assertRaises(FileExistsError):
                write_golden_reference(path, config, point, report)
            write_golden_reference(path, config, point, report, force=True)

    def test_rejects_a_failed_probe_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_golden_reference(
                    Path(temporary) / "reference.json",
                    SimpleNamespace(),
                    ProbePoint(1.0, 1.0, 0.0, 1.0, 1.0),
                    {"status": "fail"},
                )


if __name__ == "__main__":
    unittest.main()
