import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from shapeic_lut_generation.config import (
    DeviceConfig,
    GenerationConfig,
    LinearRange,
    SimulatorConfig,
    SweepConfig,
    load_config,
)
from shapeic_lut_generation.ngspice import _raw_column
from shapeic_lut_generation.writer import LutStorage


class LinearRangeTests(unittest.TestCase):
    def test_builds_inclusive_ascending_and_descending_ranges(self):
        np.testing.assert_allclose(LinearRange(0.0, 0.2, 0.1).values(), [0.0, 0.1, 0.2])
        np.testing.assert_allclose(LinearRange(0.0, -0.2, -0.1).values(), [0.0, -0.1, -0.2])

    def test_builds_exclusive_finger_width_range(self):
        values = LinearRange(0.15e-6, 10.0e-6, 0.5e-6, False).values()
        self.assertEqual(values.size, 20)
        self.assertAlmostEqual(values[0], 0.15e-6)
        self.assertAlmostEqual(values[-1], 9.65e-6)

    def test_rejects_an_inclusive_range_that_misses_its_stop(self):
        with self.assertRaisesRegex(ValueError, "does not reach stop"):
            LinearRange(0.0, 1.0, 0.3).values()


class ConfigTests(unittest.TestCase):
    def test_loads_paths_relative_to_config_and_expands_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "pdk" / "model.lib"
            osdi = root / "pdk" / "model.osdi"
            model.parent.mkdir()
            model.touch()
            osdi.touch()
            config_path = root / "config.toml"
            config_path.write_text(
                """
[output]
path = "output/lut.npz"

[simulator]
binary = "true"
model_library = "${SHAPEIC_TEST_PDK}/model.lib"
osdi_paths = ["${SHAPEIC_TEST_PDK}/model.osdi"]
parameters = ["id", "gm"]

[device]
name = "test_nmos"
hierarchy = "m1"
nf = 1

[sweep]
length = [1.0]
[sweep.vbs]
start = 0.0
stop = 1.0
step = 1.0
[sweep.vgs]
start = 0.0
stop = 1.0
step = 1.0
[sweep.vds]
start = 0.0
stop = 1.0
step = 1.0
[sweep.finger_width]
start = 1.0
stop = 2.0
step = 1.0
""",
                encoding="ascii",
            )
            previous = os.environ.get("SHAPEIC_TEST_PDK")
            os.environ["SHAPEIC_TEST_PDK"] = str(model.parent)
            try:
                config = load_config(config_path)
            finally:
                if previous is None:
                    os.environ.pop("SHAPEIC_TEST_PDK", None)
                else:
                    os.environ["SHAPEIC_TEST_PDK"] = previous
            self.assertEqual(config.simulator.model_library, model)
            self.assertEqual(config.output_path, root / "output" / "lut.npz")
            self.assertEqual(config.simulator.parameters, ("id", "gm"))


class WriterTests(unittest.TestCase):
    def test_writes_manifest_and_five_dimensional_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.npz"
            two_values = LinearRange(0.0, 1.0, 1.0)
            config = GenerationConfig(
                source_path=root / "test.toml",
                output_path=output,
                description="writer test",
                simulator=SimulatorConfig(
                    binary="true",
                    temperature_c=27.0,
                    workers=1,
                    model_library=root / "model.lib",
                    library_section="tt",
                    osdi_paths=(),
                    parameters=("id", "gm"),
                ),
                device=DeviceConfig("test_nmos", "XM1", "m1", 1),
                sweep=SweepConfig(
                    length=np.asarray([1.0, 2.0]),
                    vbs=two_values,
                    vgs=two_values,
                    vds=two_values,
                    finger_width=LinearRange(1.0, 2.0, 1.0),
                ),
            )
            storage = LutStorage(config, root / "staging")
            for length_index in range(2):
                for vbs_index in range(2):
                    for width_index in range(2):
                        value = 1.0 + length_index + 2 * vbs_index + 4 * width_index
                        block = np.full((2, 2), value, dtype=np.float32)
                        storage.put(
                            (length_index, vbs_index, width_index),
                            {"id": block, "gm": block * 2},
                        )
            storage.validate_and_flush()
            storage.write_archive(force=False)

            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["format"], "shapeic-lut")
                self.assertEqual(manifest["version"], 2)
                self.assertEqual(
                    manifest["models"][0]["axis_order"],
                    ["length", "vbs", "vgs", "vds", "finger_width"],
                )
                array = np.load(io.BytesIO(archive.read("models/0/parameters/id.npy")))
                self.assertEqual(array.shape, (2, 2, 2, 2, 2))
                self.assertTrue(np.isfinite(array).all())

    def test_normalizes_ngspice_raw_vector_names(self):
        self.assertEqual(_raw_column("id"), "i(shapeic_id)")
        self.assertEqual(_raw_column("weff"), "v(shapeic_weff)")
        self.assertEqual(_raw_column("gm"), "shapeic_gm")


if __name__ == "__main__":
    unittest.main()
