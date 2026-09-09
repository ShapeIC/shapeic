import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from shapeic_lut_generation import load_config
from shapeic_lut_generation.probe import ProbePoint, run_probe, write_golden_reference


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
RUN_EDA = os.environ.get("SHAPEIC_RUN_EDA_TESTS") == "1"
SELECTED_PDK = os.environ.get("PDK")
PDK_CASES = {
    "sky130A": {
        "config_prefix": "sky130A_1v8",
        "fixture_prefix": "sky130a_1v8",
        "points": {
            "nmos": ProbePoint(0.3e-6, 0.42e-6, 0.0, 0.6, 0.6),
            "pmos": ProbePoint(0.3e-6, 0.42e-6, 0.0, -0.6, -0.6),
        },
    },
    "gf180mcuD": {
        "config_prefix": "gf180mcuD_3v3",
        "fixture_prefix": "gf180mcud_3v3",
        "points": {
            "nmos": ProbePoint(0.28e-6, 0.22e-6, 0.0, 0.8, 0.8),
            "pmos": ProbePoint(0.28e-6, 0.22e-6, 0.0, -0.8, -0.8),
        },
    },
}


def assert_reference_close(test: unittest.TestCase, actual, expected, context="root"):
    if isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected), context)
        for key, value in expected.items():
            assert_reference_close(test, actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected), context)
        for index, value in enumerate(expected):
            assert_reference_close(test, actual[index], value, f"{context}[{index}]")
    elif isinstance(expected, float):
        test.assertTrue(math.isfinite(actual), context)
        tolerance = 5.0e-6 * max(abs(expected), 1.0e-30)
        test.assertAlmostEqual(actual, expected, delta=tolerance, msg=context)
    else:
        test.assertEqual(actual, expected, context)


@unittest.skipUnless(
    RUN_EDA and SELECTED_PDK in PDK_CASES,
    "set SHAPEIC_RUN_EDA_TESTS=1 and select a supported PDK",
)
class PdkSmokeEdaTests(unittest.TestCase):
    def test_regenerates_the_committed_nmos_and_pmos_references(self):
        case = PDK_CASES[SELECTED_PDK]
        with tempfile.TemporaryDirectory(prefix="shapeic-pdk-smoke-eda-") as temporary:
            root = Path(temporary)
            for polarity, point in case["points"].items():
                with self.subTest(polarity=polarity):
                    config = load_config(
                        CONFIG_ROOT
                        / f"{case['config_prefix']}_{polarity}_smoke.toml"
                    )
                    artifacts, report = run_probe(config, point, root / "artifacts")
                    actual_path = root / f"{polarity}_reference.json"
                    write_golden_reference(actual_path, config, point, report)
                    expected_path = (
                        FIXTURE_ROOT
                        / f"{case['fixture_prefix']}_{polarity}_reference.json"
                    )
                    actual = json.loads(actual_path.read_text(encoding="utf-8"))
                    expected = json.loads(expected_path.read_text(encoding="utf-8"))
                    assert_reference_close(self, actual, expected)
                    self.assertTrue((artifacts / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
