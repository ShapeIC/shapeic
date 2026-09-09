from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines/ihp_sg13g2_v2.json"


class IhpBaselineTests(unittest.TestCase):
    def test_baseline_is_portable_and_charge_conserving(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

        self.assertEqual(baseline["format"], "shapeic-physical-baseline")
        self.assertEqual(baseline["version"], 1)
        self.assertEqual(baseline["pdk"], "ihp-sg13g2")
        for digest in baseline["source_artifacts"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)

        for primitive in ("simplediffpair", "currentmirror"):
            value = baseline["primitive_smoke"][primitive]
            ports = value["port_order"]
            _assert_charge_conserving(
                self,
                value["interconnect_capacitance_f"],
                len(ports),
                symmetric=True,
            )
            _assert_charge_conserving(
                self,
                value["device_capacitance_correction_f"],
                len(ports),
                symmetric=False,
            )

        self.assertFalse(_contains_absolute_path(baseline))

    def test_available_stable_source_archives_match_recorded_hashes(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        generated = ROOT / "generated"
        sources = {
            "magic_smoke_lut_sha256": generated
            / "ihp_sg13g2_ota_physical_magic_smoke.npz",
            "full_lut_v2_sha256": generated / "ihp_sg13g2_ota_physical_v2.npz",
        }
        for name, path in sources.items():
            if path.is_file():
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    baseline["source_artifacts"][name],
                )


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith("/")
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    return False


def _assert_charge_conserving(
    test: unittest.TestCase,
    matrix: list[list[float]],
    size: int,
    *,
    symmetric: bool,
) -> None:
    test.assertEqual(len(matrix), size)
    test.assertTrue(all(len(row) == size for row in matrix))
    for row in matrix:
        test.assertAlmostEqual(sum(row), 0.0, delta=1.0e-27)
    for index in range(size):
        test.assertAlmostEqual(
            sum(row[index] for row in matrix),
            0.0,
            delta=1.0e-27,
        )
        if symmetric:
            for other in range(size):
                test.assertAlmostEqual(
                    matrix[index][other],
                    matrix[other][index],
                    delta=1.0e-30,
                )


if __name__ == "__main__":
    unittest.main()
