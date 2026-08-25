import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shapeic_lut_generation.config import (
    DeviceConfig,
    GenerationConfig,
    LinearRange,
    PdkConfig,
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
    @staticmethod
    def _write_config(root: Path, pdk_table: str = "") -> Path:
        config_path = root / "config.toml"
        config_path.write_text(
            f"""
{pdk_table}
[output]
path = "output/lut.npz"

[simulator]
binary = "true"
model_library = "libs.tech/ngspice/model.lib"
osdi_paths = ["libs.tech/ngspice/model.osdi"]
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
        return config_path

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
            self.assertIsNone(config.pdk)

    def test_resolves_declared_pdk_paths_from_pdk_root_and_pdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdk_directory = root / "pdks" / "sky130A"
            model_directory = pdk_directory / "libs.tech" / "ngspice"
            model_directory.mkdir(parents=True)
            (model_directory / "model.lib").touch()
            (model_directory / "model.osdi").touch()
            config_path = self._write_config(
                root,
                """[pdk]
name = "sky130A"
revision = "v0.1.0"
corner = "tt"
nominal_voltage = 1.8
""",
            )

            with patch.dict(
                os.environ,
                {"PDK_ROOT": str(root / "pdks"), "PDK": "sky130A"},
                clear=False,
            ):
                config = load_config(config_path)

            self.assertEqual(config.pdk.name, "sky130A")
            self.assertEqual(config.pdk.revision, "v0.1.0")
            self.assertEqual(config.pdk.corner, "tt")
            self.assertEqual(config.pdk.nominal_voltage, 1.8)
            self.assertEqual(config.pdk.directory, pdk_directory)
            self.assertEqual(config.simulator.model_library, model_directory / "model.lib")
            self.assertEqual(config.simulator.osdi_paths, (model_directory / "model.osdi",))
            self.assertEqual(config.output_path, root / "output" / "lut.npz")

    def test_rejects_pdk_environment_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pdks" / "gf180mcuD").mkdir(parents=True)
            config_path = self._write_config(
                root,
                """[pdk]
name = "sky130A"
corner = "tt"
nominal_voltage = 1.8
""",
            )
            with patch.dict(
                os.environ,
                {"PDK_ROOT": str(root / "pdks"), "PDK": "gf180mcuD"},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "PDK selects 'gf180mcuD'"):
                    load_config(config_path)

    def test_reports_a_missing_local_pdk_without_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pdks").mkdir()
            config_path = self._write_config(
                root,
                """[pdk]
name = "sky130A"
corner = "tt"
nominal_voltage = 1.8
""",
            )
            with patch.dict(
                os.environ,
                {"PDK_ROOT": str(root / "pdks"), "PDK": "sky130A"},
                clear=False,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "install it separately"):
                    load_config(config_path)

    def test_rejects_unsafe_pdk_names_and_nonpositive_nominal_voltage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, voltage, message in [
                ("../sky130A", "1.8", "single directory name"),
                ("sky130A", "0.0", "positive and finite"),
            ]:
                config_path = self._write_config(
                    root,
                    f"""[pdk]
name = "{name}"
corner = "tt"
nominal_voltage = {voltage}
""",
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_config(config_path)


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
                pdk=PdkConfig(
                    name="sky130A",
                    revision="v0.1.0",
                    corner="tt",
                    nominal_voltage=1.8,
                    directory=root / "pdks" / "sky130A",
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
                self.assertEqual(manifest["pdk"], "sky130A")
                self.assertEqual(manifest["pdk_revision"], "v0.1.0")
                self.assertEqual(manifest["corner"], "tt")
                self.assertEqual(manifest["temperature_c"], 27.0)
                self.assertEqual(manifest["nominal_voltage"], 1.8)
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
