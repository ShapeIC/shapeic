import unittest

import numpy as np

from shapeic_lut_generation.config import CapacitanceConvention
from shapeic_lut_generation.ngspice import normalize_capacitance_parameters
from shapeic_lut_generation.probe import (
    combined_matrix,
    charge_residual,
    intrinsic_matrix,
    matrix_relative_error,
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


if __name__ == "__main__":
    unittest.main()
