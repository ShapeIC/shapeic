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
    MosTerminal,
    PdkConfig,
    SimulatorConfig,
    SpiceConfig,
    SpiceDirective,
    SpiceDirectiveKind,
    SpiceInstanceKind,
    SweepConfig,
    WidthConvention,
    load_config,
)
from shapeic_lut_generation.ngspice import _netlist, _raw_column
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

    @staticmethod
    def _write_typed_config(
        root: Path,
        directives: str,
        *,
        simulator_extra: str = "",
        instance: str = "XM1",
        instance_kind: str = "subcircuit",
        terminals: str = '["d", "g", "s", "b"]',
        geometry_unit_m: str = "1.0",
        parameter_map: str = 'gm = "gm_native"',
    ) -> Path:
        config_path = root / "typed.toml"
        config_path.write_text(
            f"""
[pdk]
name = "test_pdk"
revision = "test-revision"
corner = "tt"
nominal_voltage = 1.8

[output]
path = "output/lut.npz"

[simulator]
binary = "true"
temperature_c = 27.0
workers = 1
parameters = ["id", "gm"]
{simulator_extra}

[spice]
{directives}

[device]
name = "test_nmos"
instance = "{instance}"
instance_kind = "{instance_kind}"
terminals = {terminals}
length_parameter = "L"
width_parameter = "W"
finger_parameter = "nf"
width_convention = "per_finger"
geometry_unit_m = {geometry_unit_m}
hierarchy = "n.xm1.m0"
nf = 1

[device.parameter_map]
{parameter_map}

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

    @staticmethod
    def _typed_pdk(root: Path) -> Path:
        pdk = root / "pdks" / "test_pdk"
        model_directory = pdk / "models"
        model_directory.mkdir(parents=True)
        for name in ["design.spice", "models.lib", "model.osdi"]:
            (model_directory / name).touch()
        return pdk

    @staticmethod
    def _typed_environment(root: Path):
        return patch.dict(
            os.environ,
            {"PDK_ROOT": str(root / "pdks"), "PDK": "test_pdk"},
            clear=False,
        )

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
            self.assertEqual(config.spice.directives[0].path, model)
            self.assertEqual(config.spice.directives[0].kind, SpiceDirectiveKind.LIBRARY)
            self.assertEqual(config.spice.directives[1].path, osdi)
            self.assertEqual(config.output_path, root / "output" / "lut.npz")
            self.assertEqual(config.simulator.parameters, ("id", "gm"))
            self.assertIsNone(config.pdk)
            netlist = _netlist(config, 1.0, 0.25, 2.0, 4, ("id", "gm"), root / "raw")
            self.assertIn(f".lib '{model}' mos_tt", netlist)
            self.assertIn("XM1 ND NG 0 NB test_nmos l=1 w=8 ng=4", netlist)
            self.assertIn("@m1[gm]", netlist)

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
            self.assertEqual(config.spice.directives[0].path, model_directory / "model.lib")
            self.assertEqual(config.spice.directives[1].path, model_directory / "model.osdi")
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

    def test_loads_ordered_typed_spice_description(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdk = self._typed_pdk(root)
            config_path = self._write_typed_config(
                root,
                """[[spice.directives]]
kind = "include"
path = "models/design.spice"

[[spice.directives]]
kind = "library"
path = "models/models.lib"
section = "tt"

[[spice.directives]]
kind = "osdi"
path = "models/model.osdi"
""",
                terminals='["b", "g", "s", "d"]',
            )
            with self._typed_environment(root):
                config = load_config(config_path)

            self.assertEqual(
                [directive.kind for directive in config.spice.directives],
                [
                    SpiceDirectiveKind.INCLUDE,
                    SpiceDirectiveKind.LIBRARY,
                    SpiceDirectiveKind.OSDI,
                ],
            )
            self.assertEqual(config.spice.directives[0].path, pdk / "models/design.spice")
            self.assertEqual(config.spice.directives[1].section, "tt")
            self.assertEqual(
                config.device.terminals,
                (
                    MosTerminal.BULK,
                    MosTerminal.GATE,
                    MosTerminal.SOURCE,
                    MosTerminal.DRAIN,
                ),
            )
            self.assertEqual(config.device.parameter_map, {"gm": "gm_native"})
            self.assertEqual(config.device.geometry_unit_m, 1.0)

            netlist = _netlist(config, 1.0, 0.25, 2.0, 4, ("id", "gm"), root / "raw")
            include = f".include '{pdk / 'models/design.spice'}'"
            library = f".lib '{pdk / 'models/models.lib'}' tt"
            osdi = f"pre_osdi '{pdk / 'models/model.osdi'}'"
            self.assertLess(netlist.index(include), netlist.index(library))
            self.assertLess(netlist.index(library), netlist.index("VGS NG 0 DC=0"))
            self.assertLess(netlist.index(".control"), netlist.index(osdi))
            self.assertIn("XM1 NB NG 0 ND test_nmos L=1 W=2 nf=4", netlist)
            self.assertIn("@n.xm1.m0[gm_native]", netlist)

    def test_rejects_mixed_legacy_and_typed_spice_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._typed_pdk(root)
            config_path = self._write_typed_config(
                root,
                """[[spice.directives]]
kind = "library"
path = "models/models.lib"
section = "tt"
""",
                simulator_extra='model_library = "models/models.lib"',
            )
            with self._typed_environment(root):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    load_config(config_path)

    def test_rejects_invalid_directive_order_and_library_section(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._typed_pdk(root)
            cases = [
                (
                    """[[spice.directives]]
kind = "library"
path = "models/models.lib"
""",
                    "section",
                ),
                (
                    """[[spice.directives]]
kind = "osdi"
path = "models/model.osdi"
[[spice.directives]]
kind = "include"
path = "models/design.spice"
""",
                    "must precede",
                ),
            ]
            for directives, message in cases:
                with self.subTest(message=message):
                    config_path = self._write_typed_config(root, directives)
                    with self._typed_environment(root):
                        with self.assertRaisesRegex(ValueError, message):
                            load_config(config_path)

    def test_rejects_invalid_typed_device_description(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._typed_pdk(root)
            directive = """[[spice.directives]]
kind = "library"
path = "models/models.lib"
section = "tt"
"""
            cases = [
                ({"terminals": '["d", "g", "s", "s"]'}, "exactly once"),
                ({"instance": "MM1"}, "must start with 'X'"),
                ({"geometry_unit_m": "0.0"}, "positive and finite"),
                ({"geometry_unit_m": "nan"}, "positive and finite"),
                ({"parameter_map": ""}, "does not match parameters"),
            ]
            for arguments, message in cases:
                with self.subTest(message=message):
                    config_path = self._write_typed_config(root, directive, **arguments)
                    with self._typed_environment(root):
                        with self.assertRaisesRegex(ValueError, message):
                            load_config(config_path)

    def test_supports_direct_model_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._typed_pdk(root)
            config_path = self._write_typed_config(
                root,
                """[[spice.directives]]
kind = "library"
path = "models/models.lib"
section = "tt"
""",
                instance="M1",
                instance_kind="model",
            )
            with self._typed_environment(root):
                config = load_config(config_path)
            self.assertEqual(config.device.instance_kind, SpiceInstanceKind.MODEL)
            netlist = _netlist(config, 1.0, 0.0, 2.0, 3, ("gm",), root / "raw")
            self.assertIn("M1 ND NG 0 NB test_nmos L=1 W=2 nf=3", netlist)

    def test_converts_si_geometry_to_the_device_spice_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._typed_pdk(root)
            config_path = self._write_typed_config(
                root,
                """[[spice.directives]]
kind = "library"
path = "models/models.lib"
section = "tt"
""",
                geometry_unit_m="1.0e-6",
            )
            with self._typed_environment(root):
                config = load_config(config_path)

            netlist = _netlist(
                config,
                0.6e-6,
                0.0,
                0.42e-6,
                4,
                ("gm",),
                root / "raw",
            )
            self.assertEqual(config.device.geometry_unit_m, 1.0e-6)
            spice_length = 0.6e-6 / 1.0e-6
            spice_width = 0.42e-6 / 1.0e-6
            self.assertIn(
                f"XM1 ND NG 0 NB test_nmos L={spice_length:.17g} "
                f"W={spice_width:.17g} nf=4",
                netlist,
            )


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
                    parameters=("id", "gm"),
                ),
                spice=SpiceConfig(
                    (
                        SpiceDirective(
                            SpiceDirectiveKind.LIBRARY,
                            root / "model.lib",
                            "tt",
                        ),
                    )
                ),
                device=DeviceConfig(
                    name="test_nmos",
                    instance="XM1",
                    instance_kind=SpiceInstanceKind.SUBCIRCUIT,
                    terminals=tuple(MosTerminal),
                    length_parameter="l",
                    width_parameter="w",
                    finger_parameter="nf",
                    width_convention=WidthConvention.PER_FINGER,
                    hierarchy="m1",
                    parameter_map={"gm": "gm"},
                    nf=1,
                ),
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
