from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapeic_layout_generation.ota_routing_diagnostic import (
    GLOBAL_NODES,
    normalize_primitive_bulk,
    reduce_explicit_interconnect,
    stamp_port_matrix,
)


class OtaRoutingDiagnosticTest(unittest.TestCase):
    def test_bulk_normalization_binds_device_and_explicit_capacitance_nodes(
        self,
    ) -> None:
        raw = """.subckt primitive D G S B
X0 D G S sub sg13_lv_nmos w=1u l=0.4u
C0 D sub 2f
C1 B sub 1f
.ends primitive
"""
        normalized = normalize_primitive_bulk(raw, "sg13_lv_nmos")
        reduced = reduce_explicit_interconnect(normalized, ("D", "G", "S", "B"))

        self.assertIn("X0 D G S B sg13_lv_nmos", normalized)
        self.assertIn("C0 D B 2f", normalized)
        self.assertAlmostEqual(reduced.capacitance[0, 0], 2.0e-15)
        self.assertAlmostEqual(reduced.capacitance[0, 3], -2.0e-15)
        self.assertAlmostEqual(reduced.capacitance[3, 3], 2.0e-15)

    def test_port_stamping_merges_short_circuited_primitive_ports(self) -> None:
        ports = ("D", "S", "B")
        matrix = np.asarray(
            (
                (3.0, -1.0, -2.0),
                (-1.0, 2.0, -1.0),
                (-2.0, -1.0, 3.0),
            )
        )
        stamped = stamp_port_matrix(
            matrix,
            ports,
            {"D": "VOUT", "S": "IBIAS", "B": "IBIAS"},
        )
        vout = GLOBAL_NODES.index("VOUT")
        ibias = GLOBAL_NODES.index("IBIAS")

        self.assertEqual(stamped[vout, vout], 3.0)
        self.assertEqual(stamped[vout, ibias], -3.0)
        self.assertEqual(stamped[ibias, vout], -3.0)
        self.assertEqual(stamped[ibias, ibias], 3.0)
        np.testing.assert_allclose(stamped.sum(axis=0), 0.0)
        np.testing.assert_allclose(stamped.sum(axis=1), 0.0)

    def test_grounded_port_is_removed_without_losing_shunt_capacitance(self) -> None:
        matrix = np.asarray(((2.0, -2.0), (-2.0, 2.0)))
        stamped = stamp_port_matrix(
            matrix,
            ("D", "B"),
            {"D": "VOUT", "B": "0"},
        )
        vout = GLOBAL_NODES.index("VOUT")
        self.assertEqual(stamped[vout, vout], 2.0)
        self.assertEqual(np.count_nonzero(stamped), 1)


if __name__ == "__main__":
    unittest.main()
