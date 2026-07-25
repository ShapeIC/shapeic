from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapeic_layout_generation.extractor import parse_rc_spice, write_magic_pex
from shapeic_layout_generation.generator import generate
from shapeic_layout_generation.ota_pex import (
    normalize_bulk_nodes,
    prepare_ota_pex,
    validate_ota_pex,
    validate_primitive_pex,
)
from shapeic_layout_generation.pcell import (
    IHP_TAP_SIZE_UM,
    _ihp_mos_device,
    _wire,
    write_ota_gds,
    write_primitive_gds,
)
from shapeic_layout_generation.reducer import reduce_first_order


class GenerationTest(unittest.TestCase):
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
                self.assertEqual(len(manifest["primitives"]), 2)
                with archive.open("primitives/0/conductance.npy") as member:
                    conductance = np.load(member, allow_pickle=False)
                self.assertEqual(conductance.shape, (2, 2, 3, 6, 6))
                np.testing.assert_allclose(conductance.sum(axis=-1), 0.0, atol=1e-15)

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

    def test_normalizes_and_validates_full_ota_pex(self) -> None:
        raw = """
        .subckt ota_flat VOUT VINP VINN IBIAS VDD VSS
        XDP1 IBIAS VINP VOUT sub sg13_lv_nmos w=1u l=0.4u
        XDP2 N1 VINN IBIAS sub sg13_lv_nmos w=1u l=0.4u
        XDPN1 IBIAS IBIAS IBIAS sub sg13_lv_nmos w=1u l=0.4u
        XDPN2 IBIAS IBIAS IBIAS sub sg13_lv_nmos w=1u l=0.4u
        XCM1 VDD N1 VOUT well sg13_lv_pmos w=1u l=0.4u
        XCM2 N1 N1 VDD well sg13_lv_pmos w=1u l=0.4u
        XCMP1 VDD VDD VDD well sg13_lv_pmos w=1u l=0.4u
        XCMP2 VDD VDD VDD well sg13_lv_pmos w=1u l=0.4u
        C0 VOUT N1 2f
        .ends
        """
        normalized = normalize_bulk_nodes(raw)
        self.assertIn("XDP1 IBIAS VINP VOUT VSS sg13_lv_nmos", normalized)
        self.assertIn("XCM1 VDD N1 VOUT VDD sg13_lv_pmos", normalized)
        topology = validate_ota_pex(
            normalized,
            expected_subcircuit="ota_flat",
        )
        self.assertEqual(topology.transistor_count, 8)
        self.assertEqual(topology.capacitor_count, 1)

    def test_prepare_ota_pex_preserves_normalized_file_on_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.spice"
            output = root / "normalized.spice"
            source.write_text(
                """
                .subckt ota_flat VOUT VINP VINN IBIAS VDD VSS
                XDP1 VOUT VINP IBIAS sub sg13_lv_nmos
                XDP2 N1 VINN IBIAS sub sg13_lv_nmos
                XCM1 VOUT WRONG VDD well sg13_lv_pmos
                XCM2 N1 N1 VDD well sg13_lv_pmos
                .ends
                """,
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "output current-mirror"):
                prepare_ota_pex(source, output)
            self.assertTrue(output.is_file())
            self.assertIn(" VSS sg13_lv_nmos", output.read_text(encoding="utf-8"))

    def test_ota_pex_validation_rejects_a_disconnected_mirror_gate(self) -> None:
        text = """
        .subckt ota_flat VOUT VINP VINN IBIAS VDD VSS
        XDP1 VOUT VINP IBIAS VSS sg13_lv_nmos
        XDP2 N1 VINN IBIAS VSS sg13_lv_nmos
        XCM1 VOUT WRONG VDD VDD sg13_lv_pmos
        XCM2 N1 N1 VDD VDD sg13_lv_pmos
        .ends
        """
        with self.assertRaisesRegex(ValueError, "output current-mirror"):
            validate_ota_pex(text)

    def test_ota_pex_validation_rejects_one_unbussed_extra_finger(self) -> None:
        text = """
        .subckt ota_flat VOUT VINP VINN IBIAS VDD VSS
        XDP1 VOUT VINP IBIAS VSS sg13_lv_nmos
        XDP1_BAD internal_drain internal_gate IBIAS VSS sg13_lv_nmos
        XDP2 N1 VINN IBIAS VSS sg13_lv_nmos
        XCM1 VOUT N1 VDD VDD sg13_lv_pmos
        XCM2 N1 N1 VDD VDD sg13_lv_pmos
        .ends
        """
        with self.assertRaisesRegex(ValueError, "outside the logical terminal buses"):
            validate_ota_pex(text)

    def test_validates_all_multifinger_primitive_terminal_buses(self) -> None:
        diff_pair = """
        .subckt diff_flat DP DN GP GN S B
        XGP0 DP GP S sub sg13_lv_nmos
        XGP1 S GP DP sub sg13_lv_nmos
        XGN0 DN GN S sub sg13_lv_nmos
        XGN1 S GN DN sub sg13_lv_nmos
        XD0 S S S sub sg13_lv_nmos
        XD1 S S S sub sg13_lv_nmos
        .ends
        """
        topology = validate_primitive_pex(
            diff_pair,
            "simplediffpair",
            expected_subcircuit="diff_flat",
        )
        self.assertEqual(topology.transistor_count, 6)

        mirror = """
        .subckt mirror_flat DOUT DREF S B
        XO0 DOUT DREF S well sg13_lv_pmos
        XO1 S DREF DOUT well sg13_lv_pmos
        XR0 DREF DREF S well sg13_lv_pmos
        XR1 S DREF DREF well sg13_lv_pmos
        XD0 S S S well sg13_lv_pmos
        XD1 S S S well sg13_lv_pmos
        .ends
        """
        topology = validate_primitive_pex(mirror, "currentmirror")
        self.assertEqual(topology.transistor_count, 6)

    def test_primitive_pex_rejects_one_unbussed_finger(self) -> None:
        text = """
        .subckt diff_flat DP DN GP GN S B
        XGP0 DP GP S sub sg13_lv_nmos
        XGP1 internal_gate internal_drain S sub sg13_lv_nmos
        XGN0 DN GN S sub sg13_lv_nmos
        XD0 S S S sub sg13_lv_nmos
        XD1 S S S sub sg13_lv_nmos
        .ends
        """
        with self.assertRaisesRegex(ValueError, "outside the logical terminal buses"):
            validate_primitive_pex(text, "simplediffpair")

    def test_wire_descends_before_joining_the_horizontal_bus(self) -> None:
        class Component:
            def __init__(self) -> None:
                self.polygons: list[list[tuple[float, float]]] = []

            def add_polygon(self, points, *, layer) -> None:
                self.assert_layer = layer
                self.polygons.append(points)

        component = Component()
        _wire(component, (2.0, 3.0), (0.0, -1.0))
        self.assertEqual(component.assert_layer, "Metal1drawing")
        self.assertEqual(len(component.polygons), 2)
        vertical, horizontal = component.polygons
        self.assertTrue(all(1.8 < x < 2.2 for x, _ in vertical))
        self.assertTrue(all(-1.2 < y < -0.8 for _, y in horizontal))

    def test_ihp_mos_uses_total_width_and_validates_finger_width(self) -> None:
        calls: list[dict[str, object]] = []
        tech = SimpleNamespace(
            nmos_min_length=0.13,
            nmos_max_length=10.0,
            nmos_min_width=0.15,
            nmos_max_width=10.0,
            nmos_max_nf=20,
        )

        def mos_core(**arguments):
            calls.append(arguments)
            return arguments

        result = _ihp_mos_device(mos_core, tech, "nmos", 0.4, 10.0, 20)
        self.assertEqual(result["width"], 200.0)
        self.assertEqual(result["nf"], 20)
        self.assertEqual(result["is_pmos"], False)
        self.assertEqual(IHP_TAP_SIZE_UM, 0.78)

        with self.assertRaisesRegex(ValueError, "finger width"):
            _ihp_mos_device(mos_core, tech, "nmos", 0.4, 10.1, 20)

    @unittest.skipUnless(
        importlib.util.find_spec("gdsfactory") and importlib.util.find_spec("ihp"),
        "requires the optional IHP layout backend",
    )
    def test_ihp_pcell_generates_domain_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("MPLCONFIGDIR", directory)
            root = Path(directory)
            endpoints = (
                ("minimum", 0.4e-6, 0.15e-6, 1),
                ("maximum", 0.8e-6, 10.0e-6, 20),
            )
            for primitive in ("simplediffpair", "currentmirror"):
                for label, length, finger_width, nf in endpoints:
                    output = root / f"{primitive}-{label}.gds"
                    name = write_primitive_gds(
                        primitive, length, finger_width, nf, output
                    )
                    self.assertTrue(name.startswith(primitive))
                    self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(
        importlib.util.find_spec("gdsfactory") and importlib.util.find_spec("ihp"),
        "requires the optional IHP layout backend",
    )
    def test_ihp_full_ota_pcell_generates_six_port_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("MPLCONFIGDIR", directory)
            output = Path(directory) / "ota.gds"
            name = write_ota_gds(
                1.0e-6,
                10.0e-6,
                20,
                1.2e-6,
                10.0e-6,
                20,
                output,
            )
            self.assertTrue(name.startswith("ota_4t"))
            self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(
        importlib.util.find_spec("gdsfactory") and importlib.util.find_spec("ihp"),
        "requires the optional IHP layout backend",
    )
    def test_magic_extracts_all_multifinger_primitive_buses(self) -> None:
        magic, rcfile = self._magic_backend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ.setdefault("MPLCONFIGDIR", directory)
            for primitive in ("simplediffpair", "currentmirror"):
                for nf in (1, 20):
                    point = root / f"{primitive}-nf{nf}"
                    gds = point / "layout.gds"
                    cell_name = write_primitive_gds(
                        primitive,
                        0.8e-6,
                        0.15e-6,
                        nf,
                        gds,
                    )
                    extracted = write_magic_pex(
                        gds,
                        cell_name,
                        magic_binary=magic,
                        magic_rcfile=rcfile,
                        work_directory=point / "magic",
                    )
                    topology = validate_primitive_pex(
                        extracted.spice_path.read_text(encoding="utf-8"),
                        primitive,
                        expected_subcircuit=extracted.subcircuit_name,
                    )
                    self.assertEqual(topology.transistor_count, 4 * nf)

    @unittest.skipUnless(
        importlib.util.find_spec("gdsfactory") and importlib.util.find_spec("ihp"),
        "requires the optional IHP layout backend",
    )
    def test_magic_validates_the_complete_ota_routing(self) -> None:
        magic, rcfile = self._magic_backend()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.environ.setdefault("MPLCONFIGDIR", directory)
            gds = root / "ota.gds"
            cell_name = write_ota_gds(
                0.8e-6,
                9.0e-6,
                2,
                0.4e-6,
                5.0e-6,
                1,
                gds,
            )
            extracted = write_magic_pex(
                gds,
                cell_name,
                magic_binary=magic,
                magic_rcfile=rcfile,
                work_directory=root / "magic",
            )
            normalized = root / "ota.normalized.spice"
            topology = prepare_ota_pex(
                extracted.spice_path,
                normalized,
                expected_subcircuit=extracted.subcircuit_name,
            )
            self.assertEqual(topology.transistor_count, 12)
            self.assertTrue(gds.is_file())
            self.assertTrue(normalized.is_file())

    def _magic_backend(self) -> tuple[str, Path]:
        magic = shutil.which("magic")
        pdk_root = os.environ.get("IHP_PDK_ROOT")
        if magic is None or pdk_root is None:
            self.skipTest("requires magic and IHP_PDK_ROOT")
        rcfile = (
            Path(pdk_root) / "libs.tech/magic/ihp-sg13g2.magicrc"
        ).resolve()
        if not rcfile.is_file():
            self.skipTest(f"Magic rcfile is absent: {rcfile}")
        return magic, rcfile


if __name__ == "__main__":
    unittest.main()
