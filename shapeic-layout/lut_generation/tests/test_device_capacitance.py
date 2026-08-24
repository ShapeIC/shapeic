from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapeic_layout_generation.device_capacitance import (
    Bias,
    Geometry,
    SimulatorConfig,
    _parse_raw,
    admittance_from_port_currents,
    aggregate_primitive_spice,
    bias_port_voltages,
    charge_conservation_error,
    extract_port_admittance,
    mos_only_pex,
    port_admittance_netlist,
    primitive_device_definition,
    relative_matrix_error,
)


class DeviceCapacitanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.diff_pair = primitive_device_definition("simplediffpair")
        self.mirror = primitive_device_definition("currentmirror")
        self.simulator = SimulatorConfig(
            binary="ngspice",
            model_library=Path("/models.lib"),
            library_section="mos_tt",
            osdi_paths=(Path("/psp.osdi"),),
            temperature_c=27.0,
            frequencies_hz=(1.0e6, 1.0e7),
        )

    def test_mos_only_pex_preserves_mos_geometry_and_normalizes_bulk(self) -> None:
        raw = """
        * Magic extraction with continued geometry parameters
        .subckt primitive DP DN GP GN S B
        X0 DP GP S substrate sg13_lv_nmos
        + w=1u l=0.4u ad=2p as=3p pd=4u ps=5u
        Xdummy S S S substrate sg13_lv_nmos w=1u l=0.4u
        R0 DP internal 12
        C0 DP B 4f
        Xignored DP GP S substrate unrelated_model w=1u
        .ends primitive
        """
        filtered = mos_only_pex(raw, self.diff_pair)
        self.assertIn(".subckt primitive DP DN GP GN S B", filtered)
        self.assertIn(
            "X0 DP GP S B sg13_lv_nmos "
            "w=1u l=0.4u ad=2p as=3p pd=4u ps=5u",
            filtered,
        )
        self.assertIn("Xdummy S S S B sg13_lv_nmos", filtered)
        self.assertNotIn("substrate", filtered)
        self.assertNotIn("R0", filtered)
        self.assertNotIn("C0", filtered)
        self.assertNotIn("Xignored", filtered)

    def test_mos_only_pex_rejects_an_incompatible_interface(self) -> None:
        with self.assertRaisesRegex(ValueError, "ports must be"):
            mos_only_pex(
                ".subckt primitive D G S B\n"
                "X0 D G S B sg13_lv_nmos w=1u\n"
                ".ends\n",
                self.diff_pair,
            )

    def test_aggregate_uses_total_width_and_preserves_finger_count(self) -> None:
        geometry = Geometry(length=0.8e-6, finger_width=3.0e-6, nf=4)
        diff = aggregate_primitive_spice(self.diff_pair, geometry)
        mirror = aggregate_primitive_spice(self.mirror, geometry)
        for netlist in (diff, mirror):
            self.assertIn("w=1.20000000000000003e-05", netlist)
            self.assertIn("ng=4", netlist)
        self.assertEqual(diff.count("sg13_lv_nmos"), 2)
        self.assertEqual(mirror.count("sg13_lv_pmos"), 2)
        self.assertIn("XCM2 DREF DREF S B", mirror)

    def test_biases_are_source_referenced_for_both_primitives(self) -> None:
        bias = Bias(vbs=-0.2, vgs=0.3, vds=0.4)
        self.assertEqual(
            bias_port_voltages(self.diff_pair, bias),
            {
                "DP": 0.4,
                "DN": 0.4,
                "GP": 0.3,
                "GN": 0.3,
                "S": 0.0,
                "B": -0.2,
            },
        )
        self.assertEqual(
            bias_port_voltages(self.mirror, bias),
            {"DOUT": 0.4, "DREF": 0.3, "S": 0.0, "B": -0.2},
        )

    def test_port_netlist_excites_one_port_at_both_frequencies(self) -> None:
        netlist = port_admittance_netlist(
            self.simulator,
            ("D", "G", "S", "B"),
            {"D": 0.4, "G": 0.3, "S": 0.0, "B": -0.2},
            "D",
            Path("/primitive.spice"),
            "primitive",
            Path("/admittance.raw"),
        )
        self.assertIn(
            "ac dec 1 1.00000000000000000e+06 1.00000000000000000e+07",
            netlist,
        )
        self.assertIn("VPORT_D D 0 DC=4.00000000000000022e-01 AC=1", netlist)
        self.assertIn("VPORT_G G 0 DC=2.99999999999999989e-01 AC=0", netlist)
        self.assertIn("save i(VPORT_D) i(VPORT_G) i(VPORT_S) i(VPORT_B)", netlist)
        self.assertIn("pre_osdi '/psp.osdi'", netlist)

    def test_reconstructs_complete_matrices_with_ngspice_current_sign(self) -> None:
        frequencies = np.asarray((1.0e6, 1.0e7))
        conductance = 1.0e-6 * np.asarray(
            [
                [2.0, -1.0, -1.0],
                [-1.0, 3.0, -2.0],
                [-1.0, -2.0, 3.0],
            ]
        )
        capacitance = 1.0e-15 * np.asarray(
            [
                [3.0, -1.0, -2.0],
                [-1.0, 2.0, -1.0],
                [-2.0, -1.0, 3.0],
            ]
        )
        omega = 2.0 * np.pi * frequencies[:, None, None]
        source_currents = -(conductance + 1j * omega * capacitance)

        result = admittance_from_port_currents(frequencies, source_currents)

        np.testing.assert_allclose(result.conductance, conductance, atol=1e-21)
        np.testing.assert_allclose(result.capacitance, capacitance, atol=1e-30)
        self.assertLess(result.frequency_consistency, 1.0e-15)
        self.assertLess(charge_conservation_error(result.conductance), 1.0e-15)
        self.assertLess(charge_conservation_error(result.capacitance), 1.0e-15)

    def test_reports_frequency_dependent_capacitance(self) -> None:
        frequencies = np.asarray((1.0e6, 1.0e7))
        low = np.asarray([[1.0, -1.0], [-1.0, 1.0]]) * 1.0e-15
        high = low * 1.1
        matrices = np.stack((low, high))
        source_currents = -1j * (
            2.0 * np.pi * frequencies[:, None, None] * matrices
        )
        result = admittance_from_port_currents(frequencies, source_currents)
        self.assertAlmostEqual(
            result.frequency_consistency,
            relative_matrix_error(low, high),
        )
        np.testing.assert_allclose(result.capacitance, (low + high) / 2.0)

    def test_rejects_frequencies_that_are_not_one_decade_apart(self) -> None:
        invalid = SimulatorConfig(
            binary="ngspice",
            model_library=Path("/models.lib"),
            library_section="mos_tt",
            osdi_paths=(),
            temperature_c=27.0,
            frequencies_hz=(1.0e6, 2.0e6),
        )
        with self.assertRaisesRegex(ValueError, "decade endpoints"):
            port_admittance_netlist(
                invalid,
                ("D", "S"),
                {"D": 0.4, "S": 0.0},
                "D",
                Path("/primitive.spice"),
                "primitive",
                Path("/admittance.raw"),
            )

    def test_extracts_every_matrix_column_in_separate_process(self) -> None:
        ports = self.mirror.ports
        frequencies = np.asarray(self.simulator.frequencies_hz)
        conductance = 1.0e-6 * np.asarray(
            [
                [2.0, -1.0, -0.5, -0.5],
                [-1.0, 2.0, -0.5, -0.5],
                [-0.5, -0.5, 2.0, -1.0],
                [-0.5, -0.5, -1.0, 2.0],
            ]
        )
        capacitance = conductance * 1.0e-9
        admittance = conductance + 1j * (
            2.0 * np.pi * frequencies[:, None, None] * capacitance
        )
        calls: list[str] = []

        def fake_run(command, **_arguments):
            netlist = Path(command[-1])
            calls.append(netlist.parent.name)
            (netlist.parent / "admittance.raw").touch()
            return subprocess.CompletedProcess(command, 0)

        def fake_parse(path: Path) -> np.ndarray:
            excitation = path.parent.name.removeprefix("excite_").upper()
            column = ports.index(excitation)
            dtype = [("frequency", np.complex128)] + [
                (f"i(vport_{port.lower()})", np.complex128) for port in ports
            ]
            data = np.empty(2, dtype=dtype)
            data["frequency"] = frequencies
            for row, port in enumerate(ports):
                data[f"i(vport_{port.lower()})"] = -admittance[:, row, column]
            return data

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "shapeic_layout_generation.device_capacitance.subprocess.run",
                    side_effect=fake_run,
                ),
                patch(
                    "shapeic_layout_generation.device_capacitance._parse_raw",
                    side_effect=fake_parse,
                ),
            ):
                result = extract_port_admittance(
                    self.simulator,
                    self.mirror,
                    Bias(vbs=0.0, vgs=-0.5, vds=-0.4),
                    Path("/primitive.spice"),
                    "primitive",
                    Path(directory),
                )

        self.assertEqual(calls, [f"excite_{port.lower()}" for port in ports])
        np.testing.assert_allclose(result.conductance, conductance, atol=1e-21)
        np.testing.assert_allclose(result.capacitance, capacitance, atol=1e-30)

    def test_parses_complex_ngspice_binary_raw_data(self) -> None:
        dtype = np.dtype(
            {
                "names": ("frequency", "i(vport_d)"),
                "formats": (np.complex128, np.complex128),
            }
        )
        expected = np.asarray(
            [(1.0e6 + 0.0j, -2.0e-6 + 3.0e-9j)], dtype=dtype
        )
        header = (
            b"Title: test\n"
            b"Plotname: AC Analysis\n"
            b"Flags: complex\n"
            b"No. Variables: 2\n"
            b"No. Points: 1\n"
            b"Variables:\n"
            b"0 frequency frequency\n"
            b"1 i(vport_d) current\n"
            b"Binary:\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.raw"
            path.write_bytes(header + expected.tobytes())
            actual = _parse_raw(path)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
