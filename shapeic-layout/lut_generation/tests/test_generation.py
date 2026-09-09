from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLKIT_ROOT = ROOT.parents[1] / "shapeic-cellkit"
sys.path.insert(0, str(ROOT))


def _has_ihp_backend() -> bool:
    try:
        return (
            importlib.metadata.version("gdsfactory") == "9.44.0"
            and importlib.metadata.version("ihp-gdsfactory") == "2.0.0"
        )
    except importlib.metadata.PackageNotFoundError:
        return False


def _has_gf180_backend() -> bool:
    try:
        return (
            importlib.metadata.version("gdsfactory") == "9.40.1"
            and importlib.metadata.version("gf180mcu") == "1.0.0"
        )
    except importlib.metadata.PackageNotFoundError:
        return False

from shapeic_layout_generation.config import (
    DeviceCorrectionBias,
    DeviceCorrectionConfig,
    ExtractorConfig,
    GenerationConfig,
    load_config,
)
from shapeic_layout_generation.cellkit import load_cellkit
from shapeic_layout_generation.cellkit_device_capacitance import (
    CellKitDeviceCapacitanceAdapter,
)
from shapeic_layout_generation.device_capacitance import Geometry
from shapeic_layout_generation.device_correction import (
    validate_device_correction,
)
from shapeic_layout_generation.extractor import parse_rc_spice, write_magic_pex
from shapeic_layout_generation.generator import _generate_primitive, generate
from shapeic_layout_generation.macro_pex import (
    prepare_macro_pex,
    validate_macro_pex,
)
from shapeic_layout_generation.reducer import reduce_first_order
from shapeic_layout_generation.writer import write_archive


class GenerationTest(unittest.TestCase):
    def test_magic_configuration_requires_cellkit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.toml"
            path.write_text(
                """primitives = ["simplediffpair"]
[output]
path = "physical.npz"
[sweep]
length = [0.4]
finger_width = [1.0]
nf = [1]
[extraction]
backend = "magic"
""",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "requires.*cellkit"):
                load_config(path)

    def test_prepares_a_generic_cellkit_macro_pex(self) -> None:
        spice = """.subckt macro OUT IN VDD VSS
X1 OUT IN VSS VDD nmos w=1u l=0.4u
R1 OUT n1 10
C1 n1 VSS 2f
.ends macro
"""
        technology = SimpleNamespace(
            normalize_macro_pex=lambda text, macro, bulk_ports: text.replace(
                "nmos", f"{macro}_nmos"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.spice"
            output = Path(directory) / "normalized.spice"
            source.write_text(spice, encoding="utf-8")

            topology = prepare_macro_pex(
                source,
                output,
                macro_name="amplifier",
                port_order=("OUT", "IN", "VDD", "VSS"),
                technology=technology,
                bulk_ports={"nmos": "VSS"},
                expected_subcircuit="macro",
            )

            self.assertIn("amplifier_nmos", output.read_text(encoding="utf-8"))
            self.assertEqual(topology.transistor_count, 1)
            self.assertEqual(topology.resistor_count, 1)
            self.assertEqual(topology.capacitor_count, 1)

    def test_prepares_macro_pex_in_the_manifest_port_order(self) -> None:
        spice = """.subckt macro VINP VINN IBIAS
+ VOUT VDD VSS
X1 VOUT VINP IBIAS VSS nmos
+ w=1u l=0.4u
X2 nref VINN IBIAS VSS nmos
X3 VOUT nref VDD VDD pmos
X4 nref nref VDD VDD pmos
.ends macro
"""
        technology = SimpleNamespace(
            normalize_macro_pex=lambda text, macro, bulk_ports: text
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.spice"
            output = Path(directory) / "normalized.spice"
            source.write_text(spice, encoding="utf-8")

            prepare_macro_pex(
                source,
                output,
                macro_name="ota_4t",
                port_order=("VOUT", "VINP", "VINN", "IBIAS", "VDD", "VSS"),
                technology=technology,
                bulk_ports={"nmos": "VSS", "pmos": "VDD"},
                expected_subcircuit="macro",
            )

            self.assertTrue(
                output.read_text(encoding="utf-8").startswith(
                    ".subckt macro VOUT VINP VINN IBIAS VDD VSS\n"
                )
            )
            self.assertIn(
                "X1 VOUT VINP IBIAS VSS nmos\n+ w=1u l=0.4u\n",
                output.read_text(encoding="utf-8"),
            )

    def test_generic_macro_pex_rejects_a_wrong_port_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "ports must be"):
            validate_macro_pex(
                ".subckt macro IN OUT\nM1 OUT IN 0 0 nmos\n.ends macro\n",
                ("OUT", "IN"),
            )

    def test_generic_macro_pex_rejects_a_collapsed_external_port(self) -> None:
        spice = """.subckt macro OUT IN VDD VSS
X1 VDD IN VSS VSS nmos
.ends macro
"""
        with self.assertRaisesRegex(ValueError, "collapsed external ports: OUT"):
            validate_macro_pex(spice, ("OUT", "IN", "VDD", "VSS"))

    def test_cellkit_device_adapter_derives_topology_and_prepares_netlists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            branches = (
                SimpleNamespace(name="m1", drain="DOUT", gate="DREF", source="S", bulk="B"),
                SimpleNamespace(name="m2", drain="DREF", gate="DREF", source="S", bulk="B"),
            )
            layout = SimpleNamespace(
                lut_primitive="currentmirror",
                polarity=SimpleNamespace(value="pmos"),
                operating_point_branch="m1",
                port_order=("DOUT", "DREF", "S", "B"),
                branches=branches,
            )
            model = SimpleNamespace(
                name="pmos_model",
                instance_kind="subcircuit",
                osdi_paths=(root / "model.osdi",),
                model_statements=(f".include '{root / 'models.spice'}'",),
                spice_geometry=lambda length, width, nf: (
                    f"pmos_model l={length:.17e} w={width * nf:.17e} ng={nf}"
                ),
            )
            correction = DeviceCorrectionConfig(
                backend="ngspice",
                finger_counts=np.asarray([1.0]),
                biases={},
                binary="ngspice",
                model_library=None,
                library_section="",
                osdi_paths=(),
                temperature_c=27.0,
                frequencies_hz=(1.0e6, 1.0e7),
                workers=1,
                frequency_consistency=0.01,
            )
            technology = SimpleNamespace(
                normalize_pex=lambda text, primitive: text,
                normalize_mos_device=lambda fields, primitive: (
                    fields[:4] + ["B"] + fields[5:]
                ),
            )
            physical = SimpleNamespace(
                pdk="test-pdk",
                primitives=("currentmirror",),
                cellkit=SimpleNamespace(technology=technology),
                electrical_models=SimpleNamespace(
                    for_polarity=lambda polarity: model
                ),
                device_correction=correction,
                primitive_layout=lambda primitive: layout,
            )
            adapter = CellKitDeviceCapacitanceAdapter(physical)

            definition = adapter.definition("currentmirror")
            self.assertEqual(
                definition.bias_variables, ("vds", "vgs", "zero", "vbs")
            )
            self.assertEqual(
                adapter.simulator_for("currentmirror").model_statements,
                model.model_statements,
            )
            aggregate = adapter.aggregate_spice(
                "currentmirror", Geometry(0.4e-6, 1.5e-6, 3)
            )
            self.assertIn("w=4.50000000000000011e-06 ng=3", aggregate)

            raw = root / "raw.pex.spice"
            raw.write_text(
                ".subckt mirror_flat DOUT DREF S B\n"
                "X1 DOUT DREF S well pmos_model l=0.4u w=4.5u ng=3\n"
                "X2 DREF DREF S well pmos_model l=0.4u w=4.5u ng=3\n"
                "C0 DOUT DREF 1f\n"
                ".ends mirror_flat\n",
                encoding="ascii",
            )
            prepared = adapter.prepare_extracted_geometry(
                "currentmirror",
                Geometry(0.4e-6, 1.5e-6, 3),
                raw,
                "mirror_flat",
                root / "prepared",
            )
            mos_only = prepared.pex_path.read_text(encoding="utf-8")
            self.assertNotIn("C0", mos_only)
            self.assertIn("X1 DOUT DREF S B pmos_model", mos_only)

    def test_magic_generation_uses_the_cellkit_pcell_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendered_geometries = []
            written_paths = []

            class Component:
                def write_gds(self, path):
                    written_paths.append(path)

            class Layout:
                def render(self, geometry):
                    rendered_geometries.append(geometry)
                    return SimpleNamespace(
                        component=Component(), cell_name="cellkit_primitive"
                    )

            geometry_type = lambda length, finger_width, nf: SimpleNamespace(
                length_m=length,
                finger_width_m=finger_width,
                nf=nf,
            )
            config = GenerationConfig(
                source_path=root / "config.toml",
                output_path=root / "physical.npz",
                pdk="test-pdk",
                layout_policy="test-policy",
                lengths=np.asarray([0.4e-6]),
                finger_widths=np.asarray([1.5e-6]),
                finger_counts=np.asarray([3.0]),
                primitives=("currentmirror",),
                extractor=ExtractorConfig(
                    backend="magic",
                    magic_binary="magic",
                    magic_rcfile=root / "magicrc",
                    work_directory=root / "work",
                    keep_work=True,
                ),
                device_correction=None,
                cellkit=SimpleNamespace(
                    geometry=geometry_type,
                    technology=SimpleNamespace(
                        normalize_pex=lambda text, primitive: text
                    ),
                ),
                port_orders={"currentmirror": ("DOUT", "DREF", "S", "B")},
                primitive_catalog_names={"currentmirror": "simplecurrentmirror"},
                primitive_layouts={"currentmirror": Layout()},
            )
            matrix = np.zeros((4, 4))
            extraction = SimpleNamespace(
                conductance=matrix,
                capacitance=matrix,
                pex=SimpleNamespace(
                    spice_path=root / "raw.pex.spice",
                    subcircuit_name="cellkit_primitive_flat",
                ),
            )
            with (
                patch(
                    "shapeic_layout_generation.generator.extract_primitive",
                    return_value=extraction,
                ) as extract,
            ):
                _generate_primitive(config, "currentmirror", None)

            self.assertEqual(len(rendered_geometries), 1)
            self.assertAlmostEqual(rendered_geometries[0].length_m, 0.4e-6)
            self.assertAlmostEqual(rendered_geometries[0].finger_width_m, 1.5e-6)
            self.assertEqual(rendered_geometries[0].nf, 3)
            self.assertEqual(written_paths, [root / "work/currentmirror_l0_w0_n0/layout.gds"])
            self.assertEqual(extract.call_args.args[1], "cellkit_primitive")

    def test_magic_inserts_provider_startup_before_reading_gds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gds = root / "primitive.gds"
            rcfile = root / "magicrc"
            work = root / "magic"
            gds.write_bytes(b"gds")
            rcfile.write_text("", encoding="ascii")

            def run_magic(*args, **kwargs):
                del args, kwargs
                (work / "primitive.pex.spice").write_text(
                    ".subckt primitive P N\nC0 P N 1f\n.ends\n",
                    encoding="ascii",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "shapeic_layout_generation.extractor.subprocess.run",
                side_effect=run_magic,
            ):
                result = write_magic_pex(
                    gds,
                    "primitive",
                    magic_binary="magic",
                    magic_rcfile=rcfile,
                    work_directory=work,
                    magic_startup_commands=("tech load selected",),
                )
            script = result.script_path.read_text(encoding="ascii").splitlines()
            self.assertLess(
                script.index("tech load selected"), script.index(f"gds read {gds}")
            )

    def test_cellkit_config_resolves_selected_pdk_and_manifest_ports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / "pdks/ihp-sg13g2"
            rcfile = installed / "libs.tech/magic/ihp-sg13g2.magicrc"
            rcfile.parent.mkdir(parents=True)
            rcfile.write_text("", encoding="ascii")
            model_library = installed / "models.lib"
            model_library.write_text("* models\n", encoding="ascii")
            nmos = root / "nmos.toml"
            pmos = root / "pmos.toml"
            for path, model in ((nmos, "nmos_model"), (pmos, "pmos_model")):
                path.write_text(
                    f'''[pdk]
name = "ihp-sg13g2"
revision = "fixture-revision"
corner = "tt"
[simulator]
binary = "ngspice"
temperature_c = 27.0
[spice]
[[spice.directives]]
kind = "include"
path = "models.lib"
[device]
name = "{model}"
instance_kind = "subcircuit"
terminals = ["d", "g", "s", "b"]
length_parameter = "l"
width_parameter = "w"
finger_parameter = "nf"
width_convention = "total"
geometry_unit_m = 1.0
''',
                    encoding="ascii",
                )
            config_path = root / "physical.toml"
            config_path.write_text(
                f'''primitives = ["simplediffpair", "currentmirror"]

[cellkit]
root = "{CELLKIT_ROOT}"

[pdk]

[electrical_models]
nmos = "{nmos}"
pmos = "{pmos}"

[physical]
backend = "synthetic"
work_directory = "{root / 'work'}"

[output]
path = "{root / 'physical.npz'}"

[sweep]
length = [0.4]
finger_width = [1.0]
nf = [1]

[device_capacitance_correction]
backend = "synthetic"
nf = [1]

[device_capacitance_correction.simplediffpair]
vbs = [0.0]
vgs = [0.5]
vds = [0.5]

[device_capacitance_correction.currentmirror]
vbs = [0.0]
vgs = [-0.5]
vds = [-0.5]
''',
                encoding="ascii",
            )
            with (
                patch.dict(
                    os.environ,
                    {"PDK_ROOT": str(root / "pdks"), "PDK": "ihp-sg13g2"},
                ),
            ):
                config = load_config(config_path)
                generated = generate(config_path)

            self.assertEqual(config.pdk, "ihp-sg13g2")
            self.assertEqual(config.pdk_revision, "fixture-revision")
            self.assertEqual(config.extractor.magic_rcfile, rcfile)
            self.assertEqual(
                config.port_order("simplediffpair"),
                ("DP", "DN", "GP", "GN", "S", "B"),
            )
            self.assertEqual(
                config.catalog_name("currentmirror"), "simplecurrentmirror"
            )
            self.assertEqual(config.electrical_models.nmos.source_path, nmos)
            with zipfile.ZipFile(generated) as archive:
                manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["pdk_revision"], "fixture-revision")
            self.assertEqual(manifest["cellkit"]["catalog"], "shapeic-cellkit")
            self.assertEqual(manifest["cellkit"]["technology"], "ihp-sg13g2")
            self.assertEqual(len(manifest["cellkit"]["technology_sha256"]), 64)
            self.assertEqual(
                manifest["electrical_models"]["nmos"]["sha256"],
                config.electrical_models.nmos.digest,
            )
            self.assertEqual(
                manifest["primitives"][0]["cellkit"]["catalog_primitive"],
                "simplediffpair",
            )
            correction_manifest = manifest["primitives"][0][
                "device_capacitance_correction"
            ]
            self.assertEqual(
                correction_manifest["electrical_model_sha256"],
                config.electrical_models.nmos.digest,
            )
            self.assertNotIn("library_section", correction_manifest)

    def test_cellkit_config_does_not_install_a_missing_pdk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "physical.toml"
            config_path.write_text(
                f'''[cellkit]
root = "{CELLKIT_ROOT}"
[pdk]
root = "{root / 'pdks'}"
name = "ihp-sg13g2"
[electrical_models]
nmos = "missing-nmos.toml"
pmos = "missing-pmos.toml"
[physical]
backend = "synthetic"
[output]
path = "physical.npz"
[sweep]
length = [0.4]
finger_width = [1.0]
nf = [1]
''',
                encoding="ascii",
            )
            with self.assertRaisesRegex(Exception, "was not found"):
                load_config(config_path)

    def test_synthetic_archive_matches_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "smoke.toml"
            output = root / "physical.npz"
            config.write_text(
                f'''pdk = "test-pdk"
layout_policy = "test-policy"
primitives = ["simplediffpair", "currentmirror"]

[output]
path = "{output}"

[sweep]
length = [0.4, 0.8]
finger_width = [0.15, 10.0]
nf = [1, 2, 20]

[extraction]
backend = "synthetic"
work_directory = "{root / 'work'}"
''',
                encoding="ascii",
            )
            self.assertEqual(generate(config), output)
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["format"], "shapeic-physical-lut")
                self.assertEqual(manifest["version"], 1)
                self.assertNotIn("vbs", manifest["units"])
                self.assertEqual(len(manifest["primitives"]), 2)
                self.assertNotIn(
                    "device_capacitance_correction",
                    manifest["primitives"][0],
                )
                with archive.open("primitives/0/conductance.npy") as member:
                    conductance = np.load(member, allow_pickle=False)
                self.assertEqual(conductance.shape, (2, 2, 3, 6, 6))
                np.testing.assert_allclose(conductance.sum(axis=-1), 0.0, atol=1e-15)

    def test_synthetic_archive_stores_device_correction_as_version_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "correction.toml"
            output = root / "physical-v2.npz"
            config.write_text(
                f'''pdk = "test-pdk"
layout_policy = "test-policy"
primitives = ["simplediffpair", "currentmirror"]

[output]
path = "{output}"

[sweep]
length = [0.4, 0.8]
finger_width = [1.0, 2.0]
nf = [1, 2, 10]

[extraction]
backend = "synthetic"
work_directory = "{root / 'work'}"

[device_capacitance_correction]
backend = "synthetic"
nf = [1, 10]
frequencies_hz = [1.0e6, 1.0e7]

[device_capacitance_correction.simplediffpair]
vbs = [-1.0, 0.0]
vgs = [0.1, 1.2]
vds = [0.1, 1.2]

[device_capacitance_correction.currentmirror]
vbs = [0.0, 1.0]
vgs = [-1.2, -0.1]
vds = [-1.2, -0.1]
''',
                encoding="ascii",
            )
            self.assertEqual(generate(config), output)
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["version"], 2)
                self.assertEqual(manifest["units"]["vbs"], "V")
                for primitive in manifest["primitives"]:
                    correction = primitive["device_capacitance_correction"]
                    self.assertEqual(
                        correction["definition"],
                        "pex_mos_only_minus_aggregate_compact_model",
                    )
                    self.assertEqual(
                        correction["axis_order"],
                        ["length", "finger_width", "nf", "vbs", "vgs", "vds"],
                    )
                    with archive.open(correction["capacitance"]) as member:
                        values = np.load(member, allow_pickle=False)
                    ports = len(primitive["port_order"])
                    self.assertEqual(
                        values.shape,
                        (2, 2, 2, 2, 2, 2, ports, ports),
                    )
                    np.testing.assert_allclose(
                        values.sum(axis=-1),
                        0.0,
                        atol=1e-30,
                    )
                    np.testing.assert_allclose(
                        values.sum(axis=-2),
                        0.0,
                        atol=1e-30,
                    )

    def test_device_correction_nf_must_reuse_physical_geometries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "invalid-correction.toml"
            config.write_text(
                f'''primitives = ["simplediffpair"]

[output]
path = "{root / 'physical.npz'}"

[sweep]
length = [0.4]
finger_width = [1.0]
nf = [1, 10]

[extraction]
backend = "synthetic"
work_directory = "{root / 'work'}"

[device_capacitance_correction]
backend = "synthetic"
nf = [1, 2]

[device_capacitance_correction.simplediffpair]
vbs = [0.0]
vgs = [0.5]
vds = [0.5]
''',
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                ValueError,
                "subset of the physical nf sweep",
            ):
                load_config(config)

    def test_device_correction_allows_nonsymmetric_charge_conserving_delta(
        self,
    ) -> None:
        values = np.asarray(
            [
                [1.0, -1.0, 0.0],
                [0.0, 1.0, -1.0],
                [-1.0, 0.0, 1.0],
            ]
        )
        validate_device_correction("test", values)
        values[0, 0] += 1.0e-3
        with self.assertRaisesRegex(ValueError, "must conserve charge"):
            validate_device_correction("test", values)

    def test_invalid_version_two_data_does_not_replace_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "physical-v2.npz"
            output.write_bytes(b"existing archive")
            correction = DeviceCorrectionConfig(
                backend="synthetic",
                finger_counts=np.asarray([1.0]),
                biases={
                    "currentmirror": DeviceCorrectionBias(
                        vbs=np.asarray([0.0]),
                        vgs=np.asarray([-0.5]),
                        vds=np.asarray([-0.5]),
                    )
                },
                binary="ngspice",
                model_library=None,
                library_section="mos_tt",
                osdi_paths=(),
                temperature_c=27.0,
                frequencies_hz=(1.0e6, 1.0e7),
                workers=1,
                frequency_consistency=0.01,
            )
            config = GenerationConfig(
                source_path=root / "config.toml",
                output_path=output,
                pdk="test",
                layout_policy="test",
                lengths=np.asarray([0.4e-6]),
                finger_widths=np.asarray([1.0e-6]),
                finger_counts=np.asarray([1.0]),
                primitives=("currentmirror",),
                extractor=ExtractorConfig(
                    backend="synthetic",
                    magic_binary="magic",
                    magic_rcfile=None,
                    work_directory=root / "work",
                    keep_work=True,
                ),
                device_correction=correction,
            )
            interconnect = np.zeros((1, 1, 1, 4, 4))
            invalid = np.zeros((1, 1, 1, 1, 1, 1, 4, 4))
            invalid[..., 0, 0] = 1.0e-15
            with self.assertRaisesRegex(ValueError, "must conserve charge"):
                write_archive(
                    config,
                    {"currentmirror": (interconnect, interconnect)},
                    {"currentmirror": invalid},
                    force=True,
                )
            self.assertEqual(output.read_bytes(), b"existing archive")
            self.assertEqual(
                list(root.glob(".physical-v2.npz.*.tmp")),
                [],
            )

    def test_real_correction_reuses_each_magic_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Component:
                @staticmethod
                def write_gds(path):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"gds")

            branch = SimpleNamespace(
                name="m1", drain="DOUT", gate="DREF", source="S", bulk="B"
            )
            layout = SimpleNamespace(
                polarity=SimpleNamespace(value="pmos"),
                port_order=("DOUT", "DREF", "S", "B"),
                branches=(branch,),
                render=lambda geometry: SimpleNamespace(
                    component=Component(), cell_name=f"mirror_nf{geometry.nf}"
                ),
            )
            technology = SimpleNamespace(
                normalize_pex=lambda text, primitive: text,
                validate_primitive_pex=lambda *arguments: None,
            )
            correction = DeviceCorrectionConfig(
                backend="ngspice",
                finger_counts=np.asarray([1.0]),
                biases={
                    "currentmirror": DeviceCorrectionBias(
                        vbs=np.asarray([0.0]),
                        vgs=np.asarray([-0.5]),
                        vds=np.asarray([-0.5]),
                    )
                },
                binary="ngspice",
                model_library=Path("/model.lib"),
                library_section="mos_tt",
                osdi_paths=(),
                temperature_c=27.0,
                frequencies_hz=(1.0e6, 1.0e7),
                workers=1,
                frequency_consistency=0.01,
            )
            config = GenerationConfig(
                source_path=root / "config.toml",
                output_path=root / "physical.npz",
                pdk="ihp-sg13g2",
                layout_policy="test",
                lengths=np.asarray([0.4e-6]),
                finger_widths=np.asarray([1.0e-6]),
                finger_counts=np.asarray([1.0, 2.0]),
                primitives=("currentmirror",),
                extractor=ExtractorConfig(
                    backend="magic",
                    magic_binary="magic",
                    magic_rcfile=Path("/magicrc"),
                    work_directory=root / "work",
                    keep_work=True,
                ),
                device_correction=correction,
                cellkit=SimpleNamespace(
                    geometry=lambda length, finger_width, nf: SimpleNamespace(
                        length_m=length, finger_width_m=finger_width, nf=nf
                    ),
                    technology=technology,
                ),
                port_orders={"currentmirror": ("DOUT", "DREF", "S", "B")},
                primitive_catalog_names={"currentmirror": "simplecurrentmirror"},
                primitive_layouts={"currentmirror": layout},
            )
            port_matrix = np.zeros((4, 4))
            extracted_paths = [root / "nf1.pex", root / "nf2.pex"]
            extractions = [
                SimpleNamespace(
                    conductance=port_matrix,
                    capacitance=port_matrix,
                    pex=SimpleNamespace(
                        spice_path=path,
                        subcircuit_name=f"pex_{index}",
                    ),
                )
                for index, path in enumerate(extracted_paths)
            ]
            correction_values = np.zeros((1, 1, 1, 4, 4))
            with (
                patch(
                    "shapeic_layout_generation.generator.extract_primitive",
                    side_effect=extractions,
                ) as extract,
                patch(
                    "shapeic_layout_generation.generator."
                    "characterize_geometry_correction",
                    return_value=(correction_values, 0.0),
                ) as characterize,
            ):
                _, generated_correction = _generate_primitive(
                    config,
                    "currentmirror",
                    SimpleNamespace(),  # type: ignore[arg-type]
                )
            self.assertEqual(extract.call_count, 2)
            characterize.assert_called_once()
            self.assertEqual(
                characterize.call_args.args[4],
                extracted_paths[0],
            )
            np.testing.assert_array_equal(
                generated_correction,
                np.zeros((1, 1, 1, 1, 1, 1, 4, 4)),
            )

    def test_parses_and_reduces_rc_network(self) -> None:
        network = parse_rc_spice(
            """
            R1 P X 10
            R2 X N 20
            C1 P X 2pF
            C2 X N 3p
            """
        )
        conductance, capacitance = reduce_first_order(
            network.conductance, network.capacitance, [0, 2]
        )
        self.assertAlmostEqual(conductance[0, 0], 1.0 / 30.0)
        self.assertEqual(conductance.shape, (2, 2))
        np.testing.assert_allclose(conductance.sum(axis=1), 0.0, atol=1e-15)
        np.testing.assert_allclose(capacitance.sum(axis=1), 0.0, atol=1e-24)

    def test_parser_combines_signed_magic_capacitance_corrections(self) -> None:
        network = parse_rc_spice(
            """
            .subckt primitive P N UNUSED
            C0 P N 2f
            C1 P N -0.25f
            .ends
            """
        )
        p = network.nodes.index("P")
        n = network.nodes.index("N")
        unused = network.nodes.index("UNUSED")
        self.assertAlmostEqual(network.capacitance[p, n], -1.75e-15)
        self.assertEqual(network.capacitance[unused, unused], 0.0)

        with self.assertRaisesRegex(ValueError, "invalid extracted element"):
            parse_rc_spice("R0 P N -1")

    def test_parser_rejects_hierarchical_magic_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "remains hierarchical"):
            parse_rc_spice(
                """
                .subckt child A B
                C0 A B 1f
                .ends
                .subckt top P N
                X0 P N child
                .ends
                """
            )

    def test_reduces_a_purely_capacitive_internal_node(self) -> None:
        network = parse_rc_spice(
            """
            C0 P X 2p
            C1 X N 3p
            """
        )
        conductance, capacitance = reduce_first_order(
            network.conductance,
            network.capacitance,
            [network.nodes.index("P"), network.nodes.index("N")],
        )
        np.testing.assert_array_equal(conductance, np.zeros((2, 2)))
        expected = 1.2e-12 * np.array([[1.0, -1.0], [-1.0, 1.0]])
        np.testing.assert_allclose(capacitance, expected, rtol=1e-12, atol=1e-24)

    @unittest.skipUnless(
        _has_ihp_backend(),
        "requires the IHP layout backend with its pinned versions",
    )
    def test_ihp_pcell_generates_domain_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("MPLCONFIGDIR", directory)
            root = Path(directory)
            pdk_root = root / "pdks"
            rcfile = pdk_root / "ihp-sg13g2/libs.tech/magic/ihp-sg13g2.magicrc"
            rcfile.parent.mkdir(parents=True)
            rcfile.write_text("", encoding="ascii")
            cellkit = load_cellkit(CELLKIT_ROOT, pdk_root, "ihp-sg13g2")
            endpoints = (
                ("minimum", 0.4e-6, 0.15e-6, 1),
                ("maximum", 0.8e-6, 10.0e-6, 20),
            )
            for primitive in ("simplediffpair", "currentmirror"):
                descriptor = cellkit.catalog.primitive_descriptor_for_lut(primitive)
                layout = cellkit.catalog.primitive(descriptor.catalog_name)
                for label, length, finger_width, nf in endpoints:
                    output = root / f"{primitive}-{label}.gds"
                    rendered = layout.render(
                        cellkit.geometry(length, finger_width, nf)
                    )
                    rendered.component.write_gds(output)
                    self.assertTrue(rendered.cell_name.startswith(primitive))
                    self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(
        _has_ihp_backend(),
        "requires the IHP layout backend with its pinned versions",
    )
    def test_ihp_cellkit_ota_macro_generates_six_port_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("MPLCONFIGDIR", directory)
            output = Path(directory) / "ota.gds"
            pdk_root = Path(directory) / "pdks"
            rcfile = pdk_root / "ihp-sg13g2/libs.tech/magic/ihp-sg13g2.magicrc"
            rcfile.parent.mkdir(parents=True)
            rcfile.write_text("", encoding="ascii")
            cellkit = load_cellkit(CELLKIT_ROOT, pdk_root, "ihp-sg13g2")
            rendered = cellkit.catalog.macro_layout("ota_4t").render(
                {
                    "xdp": cellkit.geometry(1.0e-6, 10.0e-6, 20),
                    "xcm": cellkit.geometry(1.2e-6, 10.0e-6, 20),
                }
            )
            rendered.component.write_gds(output)
            self.assertTrue(rendered.cell_name.startswith("ota_4t"))
            self.assertEqual(
                rendered.port_order,
                ("VOUT", "VINP", "VINN", "IBIAS", "VDD", "VSS"),
            )
            self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(
        _has_gf180_backend(),
        "requires the GF180MCU layout backend with its pinned versions",
    )
    def test_gf180_pcell_generates_smoke_domain_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("MPLCONFIGDIR", directory)
            root = Path(directory)
            pdk_root = root / "pdks"
            rcfile = pdk_root / "gf180mcuD/libs.tech/magic/gf180mcuD.magicrc"
            rcfile.parent.mkdir(parents=True)
            rcfile.write_text("", encoding="ascii")
            cellkit = load_cellkit(CELLKIT_ROOT, pdk_root, "gf180mcuD")
            for primitive in ("simplediffpair", "currentmirror"):
                descriptor = cellkit.catalog.primitive_descriptor_for_lut(primitive)
                layout = cellkit.catalog.primitive(descriptor.catalog_name)
                for nf in (1, 2):
                    output = root / f"{primitive}-nf{nf}.gds"
                    rendered = layout.render(
                        cellkit.geometry(0.4e-6, 0.22e-6, nf)
                    )
                    rendered.component.write_gds(output)
                    self.assertTrue(rendered.cell_name.startswith(primitive))
                    self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(
        _has_gf180_backend(),
        "requires the GF180MCU layout backend with its pinned versions",
    )
    def test_gf180_cellkit_ota_macro_generates_six_port_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("MPLCONFIGDIR", directory)
            output = Path(directory) / "ota.gds"
            pdk_root = Path(directory) / "pdks"
            rcfile = pdk_root / "gf180mcuD/libs.tech/magic/gf180mcuD.magicrc"
            rcfile.parent.mkdir(parents=True)
            rcfile.write_text("", encoding="ascii")
            cellkit = load_cellkit(CELLKIT_ROOT, pdk_root, "gf180mcuD")
            rendered = cellkit.catalog.macro_layout("ota_4t").render(
                {
                    "xdp": cellkit.geometry(0.4e-6, 0.22e-6, 2),
                    "xcm": cellkit.geometry(0.4e-6, 0.22e-6, 1),
                }
            )
            rendered.component.write_gds(output)
            self.assertTrue(rendered.cell_name.startswith("ota_4t"))
            self.assertEqual(
                rendered.port_order,
                ("VOUT", "VINP", "VINN", "IBIAS", "VDD", "VSS"),
            )
            self.assertGreater(output.stat().st_size, 0)

    def test_gf180_smoke_configuration_requests_device_correction(self) -> None:
        config_path = ROOT / "configs/gf180mcuD_ota_magic_smoke.toml"
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        correction = raw["device_capacitance_correction"]
        self.assertEqual(correction["backend"], "ngspice")
        self.assertEqual(correction["nf"], [1, 2])
        self.assertEqual(correction["frequencies_hz"], [1.0e6, 1.0e7])
        self.assertIn("simplediffpair", correction)
        self.assertIn("currentmirror", correction)
        self.assertIn("_v2_", raw["output"]["path"])

    @unittest.skipUnless(
        _has_ihp_backend(),
        "requires the IHP layout backend with its pinned versions",
    )
    def test_magic_extracts_all_multifinger_primitive_buses(self) -> None:
        magic, rcfile = self._magic_backend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ.setdefault("MPLCONFIGDIR", directory)
            pdk_root = Path(os.environ["PDK_ROOT"])
            pdk = os.environ["PDK"]
            cellkit = load_cellkit(CELLKIT_ROOT, pdk_root, pdk)
            for primitive in ("simplediffpair", "currentmirror"):
                descriptor = cellkit.catalog.primitive_descriptor_for_lut(primitive)
                layout = cellkit.catalog.primitive(descriptor.catalog_name)
                for nf in (1, 20):
                    point = root / f"{primitive}-nf{nf}"
                    gds = point / "layout.gds"
                    rendered = layout.render(
                        cellkit.geometry(0.8e-6, 0.15e-6, nf)
                    )
                    gds.parent.mkdir(parents=True, exist_ok=True)
                    rendered.component.write_gds(gds)
                    extracted = write_magic_pex(
                        gds,
                        rendered.cell_name,
                        magic_binary=magic,
                        magic_rcfile=rcfile,
                        work_directory=point / "magic",
                    )
                    cellkit.technology.validate_primitive_pex(
                        extracted.spice_path.read_text(encoding="utf-8"),
                        primitive,
                        layout.polarity,
                        layout.port_order,
                        layout.branches,
                        extracted.subcircuit_name,
                    )

    @unittest.skipUnless(
        _has_ihp_backend(),
        "requires the IHP layout backend with its pinned versions",
    )
    def test_magic_validates_the_complete_ota_routing(self) -> None:
        magic, rcfile = self._magic_backend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ.setdefault("MPLCONFIGDIR", directory)
            gds = root / "ota.gds"
            pdk_root = Path(os.environ["PDK_ROOT"])
            cellkit = load_cellkit(CELLKIT_ROOT, pdk_root, os.environ["PDK"])
            rendered = cellkit.catalog.macro_layout("ota_4t").render(
                {
                    "xdp": cellkit.geometry(0.8e-6, 9.0e-6, 2),
                    "xcm": cellkit.geometry(0.4e-6, 5.0e-6, 1),
                }
            )
            rendered.component.write_gds(gds)
            extracted = write_magic_pex(
                gds,
                rendered.cell_name,
                magic_binary=magic,
                magic_rcfile=rcfile,
                work_directory=root / "magic",
            )
            normalized = root / "ota.normalized.spice"
            topology = prepare_macro_pex(
                extracted.spice_path,
                normalized,
                macro_name="ota_4t",
                port_order=rendered.port_order,
                technology=cellkit.technology,
                bulk_ports={"nmos": "VSS", "pmos": "VDD"},
                expected_subcircuit=extracted.subcircuit_name,
            )
            self.assertEqual(topology.transistor_count, 12)
            self.assertTrue(gds.is_file())
            self.assertTrue(normalized.is_file())

    def _magic_backend(self) -> tuple[str, Path]:
        magic = shutil.which("magic")
        pdk_root = os.environ.get("PDK_ROOT")
        pdk = os.environ.get("PDK")
        if magic is None or pdk_root is None or pdk is None:
            self.skipTest("requires Magic, PDK_ROOT, and PDK")
        if pdk != "ihp-sg13g2":
            self.skipTest("requires PDK=ihp-sg13g2")
        rcfile = (Path(pdk_root) / pdk / "libs.tech/magic/ihp-sg13g2.magicrc").resolve()
        if not rcfile.is_file():
            self.skipTest(f"Magic rcfile is absent: {rcfile}")
        return magic, rcfile


if __name__ == "__main__":
    unittest.main()
