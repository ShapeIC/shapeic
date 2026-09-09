import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shapeic_lut_generation.config import (
    CapacitanceConvention,
    CapacitanceNfMode,
    SpiceDirectiveKind,
    WidthConvention,
    load_config,
)
from shapeic_lut_generation.ngspice import _netlist, simulate_block


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
REVISION = "026824c7969ce6f4fc9678e6ca04b0a06a596c4b"
PARAMETER_MAP = {
    "weff": "weff",
    "vth": "vth",
    "vdsat": "vdsat",
    "gm": "gm",
    "gds": "gds",
    "cgg": "cgg",
    "cgd": "cgd",
    "cgs": "cgs",
    "cdg": "cdg",
    "cdd": "cdd",
    "cds": "cds",
    "csg": "csg",
    "csd": "csd",
    "css": "css",
    "cgsol": "cgso",
    "cgdol": "cgdo",
    "cjs": "capbs",
    "cjd": "capbd",
}


class Sky130IntegrationTests(unittest.TestCase):
    @staticmethod
    def _pdk(root: Path) -> Path:
        pdk = root / "pdks" / "sky130A"
        model_root = pdk / "libs.tech" / "combined"
        model_root.mkdir(parents=True)
        (model_root / "sky130.lib.spice").touch()
        return pdk

    @staticmethod
    def _environment(root: Path):
        return patch.dict(
            os.environ,
            {"PDK_ROOT": str(root / "pdks"), "PDK": "sky130A"},
            clear=False,
        )

    def test_all_sky130_configs_use_the_expected_typed_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdk = self._pdk(root)
            paths = sorted(CONFIG_ROOT.glob("sky130A_1v8_*.toml"))
            self.assertEqual(len(paths), 4)

            with self._environment(root):
                configs = [(path, load_config(path)) for path in paths]

            for path, config in configs:
                polarity = "nmos" if "nmos" in path.name else "pmos"
                native = "nfet" if polarity == "nmos" else "pfet"
                with self.subTest(config=path.name):
                    self.assertEqual(config.pdk.name, "sky130A")
                    self.assertEqual(config.pdk.revision, REVISION)
                    self.assertEqual(config.pdk.corner, "tt")
                    self.assertEqual(config.pdk.nominal_voltage, 1.8)
                    self.assertEqual(
                        [directive.kind for directive in config.spice.directives],
                        [SpiceDirectiveKind.LIBRARY],
                    )
                    self.assertEqual(
                        config.spice.directives[0].path,
                        pdk / "libs.tech/combined/sky130.lib.spice",
                    )
                    self.assertEqual(config.spice.directives[0].section, "tt")
                    self.assertEqual(
                        config.device.name,
                        f"sky130_fd_pr__{native}_01v8",
                    )
                    self.assertEqual(
                        config.device.hierarchy,
                        f"m.xm1.msky130_fd_pr__{native}_01v8",
                    )
                    self.assertEqual(
                        config.device.width_convention,
                        WidthConvention.TOTAL,
                    )
                    self.assertEqual(config.device.geometry_unit_m, 1.0e-6)
                    self.assertEqual(
                        config.device.capacitance_nf_mode,
                        CapacitanceNfMode.LINEAR,
                    )
                    self.assertEqual(
                        config.device.capacitance_convention,
                        CapacitanceConvention.SIGNED_NODAL,
                    )
                    self.assertEqual(config.device.parameter_map, PARAMETER_MAP)
                    self.assertEqual(
                        config.device.capacitance_nf_samples,
                        (1, 2, 3, 4),
                    )

    def test_full_and_smoke_axes_follow_the_approved_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._pdk(root)
            with self._environment(root):
                nmos = load_config(CONFIG_ROOT / "sky130A_1v8_nmos.toml")
                pmos = load_config(CONFIG_ROOT / "sky130A_1v8_pmos.toml")
                nmos_smoke = load_config(
                    CONFIG_ROOT / "sky130A_1v8_nmos_smoke.toml"
                )
                pmos_smoke = load_config(
                    CONFIG_ROOT / "sky130A_1v8_pmos_smoke.toml"
                )

            np.testing.assert_allclose(
                nmos.sweep.length,
                np.asarray([0.4, 0.8, 1.6, 3.2, 6.4]) * 1.0e-6,
            )
            self.assertEqual(nmos.sweep.vgs.values().size, 35)
            self.assertEqual(nmos.sweep.vds.values().size, 35)
            self.assertEqual(nmos.sweep.vbs.values().size, 17)
            self.assertEqual(nmos.sweep.finger_width.values().size, 20)
            np.testing.assert_allclose(pmos.sweep.vgs.values(), -nmos.sweep.vgs.values())
            np.testing.assert_allclose(pmos.sweep.vds.values(), -nmos.sweep.vds.values())
            np.testing.assert_allclose(pmos.sweep.vbs.values(), -nmos.sweep.vbs.values())

            for config in [nmos_smoke, pmos_smoke]:
                self.assertEqual(config.sweep.length.size, 2)
                self.assertEqual(config.sweep.vbs.values().size, 2)
                self.assertEqual(config.sweep.vgs.values().size, 2)
                self.assertEqual(config.sweep.vds.values().size, 2)
                self.assertEqual(config.sweep.finger_width.values().size, 2)

    def test_sky130_decks_bind_geometry_and_native_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdk = self._pdk(root)
            with self._environment(root):
                configs = [
                    load_config(CONFIG_ROOT / "sky130A_1v8_nmos_smoke.toml"),
                    load_config(CONFIG_ROOT / "sky130A_1v8_pmos_smoke.toml"),
                ]

            for config in configs:
                for nf in [1, 2, 3, 4]:
                    with self.subTest(device=config.device.name, nf=nf):
                        deck = _netlist(
                            config,
                            0.6e-6,
                            0.0,
                            0.42e-6,
                            nf,
                            config.simulator.parameters,
                            root / "output.raw",
                        )
                        library = (
                            f".lib '{pdk / 'libs.tech/combined/sky130.lib.spice'}' tt"
                        )
                        self.assertLess(deck.index(library), deck.index("VGS NG 0 DC=0"))
                        self.assertIn(
                            f"{config.device.finger_parameter}={nf}",
                            deck,
                        )
                        spice_width = config.device.spice_width(0.42e-6, nf)
                        self.assertIn(f"w={spice_width:.17g}", deck)
                        self.assertIn(
                            f"@{config.device.hierarchy}[capbs]",
                            deck,
                        )
                        self.assertIn(
                            f"@{config.device.hierarchy}[capbd]",
                            deck,
                        )
                        self.assertIn(
                            f"@{config.device.hierarchy}[cgso]",
                            deck,
                        )
                        self.assertIn(
                            f"@{config.device.hierarchy}[cgdo]",
                            deck,
                        )

    def test_linear_nf_mode_simulates_one_finger_and_scales_the_anchors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._pdk(root)
            with self._environment(root):
                config = load_config(CONFIG_ROOT / "sky130A_1v8_nmos_smoke.toml")

            base = {
                parameter: np.asarray([[float(index + 1)]], dtype=np.float32)
                for index, parameter in enumerate(config.simulator.parameters)
            }
            with patch(
                "shapeic_lut_generation.ngspice._simulate_nf_block",
                return_value=base,
            ) as simulator:
                result = simulate_block(config, 0.3e-6, 0.0, 0.42e-6)

            simulator.assert_called_once()
            for parameter in ["cgsol", "cgdol", "cjs", "cjd"]:
                np.testing.assert_array_equal(result[parameter], base[parameter])
                for nf in [2, 3, 4]:
                    np.testing.assert_array_equal(
                        result[f"{parameter}_nf{nf}"],
                        base[parameter] * nf,
                    )


if __name__ == "__main__":
    unittest.main()
