import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from shapeic_lut_generation.config import (
    CapacitanceNfMode,
    MosTerminal,
    SpiceDirectiveKind,
    SpiceInstanceKind,
    WidthConvention,
    load_config,
)
from shapeic_lut_generation.ngspice import _netlist


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
REVISION = "22f2a25f1734796de3debbbf29cf697cbbc54081"
PARAMETERS = (
    "weff",
    "id",
    "vth",
    "vsat",
    "gm",
    "gds",
    "cgg",
    "cgd",
    "cgs",
    "cdg",
    "cdd",
    "cds",
    "csg",
    "csd",
    "css",
    "cgsol",
    "cgdol",
    "cjs",
    "cjd",
)


class IhpMigrationTests(unittest.TestCase):
    @staticmethod
    def _pdk(root: Path) -> Path:
        pdk = root / "pdks" / "ihp-sg13g2"
        model_root = pdk / "libs.tech" / "ngspice"
        (model_root / "models").mkdir(parents=True)
        (model_root / "osdi").mkdir()
        (model_root / "models" / "cornerMOSlv.lib").touch()
        (model_root / "osdi" / "psp103.osdi").touch()
        (model_root / "osdi" / "psp103_nqs.osdi").touch()
        return pdk

    @staticmethod
    def _environment(root: Path):
        return patch.dict(
            os.environ,
            {"PDK_ROOT": str(root / "pdks"), "PDK": "ihp-sg13g2"},
            clear=False,
        )

    @staticmethod
    def _legacy_config(root: Path, polarity: str) -> Path:
        is_nmos = polarity == "nmos"
        model = f"sg13_lv_{polarity}"
        hierarchy = f"n.xm1.n{model}"
        vgs_start, vgs_stop, vgs_step = (
            (0.4, 0.6, 0.2) if is_nmos else (-0.4, -0.6, -0.2)
        )
        vbs_stop, vbs_step = (-0.1, -0.1) if is_nmos else (0.1, 0.1)
        parameters = ", ".join(f'"{parameter}"' for parameter in PARAMETERS)
        path = root / f"legacy_{polarity}.toml"
        path.write_text(
            f"""
[output]
path = "legacy_{polarity}.npz"

[simulator]
binary = "true"
temperature_c = 27.0
workers = 1
model_library = "${{PDK_ROOT}}/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib"
library_section = "mos_tt"
osdi_paths = [
  "${{PDK_ROOT}}/ihp-sg13g2/libs.tech/ngspice/osdi/psp103.osdi",
  "${{PDK_ROOT}}/ihp-sg13g2/libs.tech/ngspice/osdi/psp103_nqs.osdi",
]
parameters = [{parameters}]

[device]
name = "{model}"
instance = "XM1"
hierarchy = "{hierarchy}"
nf = 1
capacitance_nf_samples = [1, 2, 3, 4]

[sweep]
length = [0.4e-6, 0.8e-6]

[sweep.vgs]
start = {vgs_start}
stop = {vgs_stop}
step = {vgs_step}

[sweep.vds]
start = {vgs_start}
stop = {vgs_stop}
step = {vgs_step}

[sweep.vbs]
start = 0.0
stop = {vbs_stop}
step = {vbs_step}

[sweep.finger_width]
start = 0.5e-6
stop = 1.0e-6
step = 0.5e-6
""",
            encoding="ascii",
        )
        return path

    def test_all_ihp_configs_use_the_typed_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdk = self._pdk(root)
            paths = sorted(CONFIG_ROOT.glob("ihp_sg13g2_lv_*.toml"))
            self.assertEqual(len(paths), 4)

            with self._environment(root):
                configs = [(path, load_config(path)) for path in paths]

            expected_map = {
                parameter: parameter for parameter in PARAMETERS if parameter != "id"
            }
            for path, config in configs:
                with self.subTest(config=path.name):
                    with path.open("rb") as handle:
                        raw = tomllib.load(handle)
                    self.assertNotIn("model_library", raw["simulator"])
                    self.assertNotIn("library_section", raw["simulator"])
                    self.assertNotIn("osdi_paths", raw["simulator"])
                    self.assertEqual(raw["pdk"]["revision"], REVISION)

                    self.assertEqual(config.pdk.name, "ihp-sg13g2")
                    self.assertEqual(config.pdk.revision, REVISION)
                    self.assertEqual(config.pdk.corner, "mos_tt")
                    self.assertEqual(config.pdk.nominal_voltage, 1.2)
                    self.assertEqual(
                        [directive.kind for directive in config.spice.directives],
                        [
                            SpiceDirectiveKind.LIBRARY,
                            SpiceDirectiveKind.OSDI,
                            SpiceDirectiveKind.OSDI,
                        ],
                    )
                    self.assertEqual(
                        config.spice.directives[0].path,
                        pdk / "libs.tech/ngspice/models/cornerMOSlv.lib",
                    )
                    self.assertEqual(config.device.instance_kind, SpiceInstanceKind.SUBCIRCUIT)
                    self.assertEqual(config.device.terminals, tuple(MosTerminal))
                    self.assertEqual(config.device.length_parameter, "l")
                    self.assertEqual(config.device.width_parameter, "w")
                    self.assertEqual(config.device.finger_parameter, "ng")
                    self.assertEqual(config.device.width_convention, WidthConvention.TOTAL)
                    self.assertEqual(
                        config.device.capacitance_nf_mode,
                        CapacitanceNfMode.SIMULATE,
                    )
                    self.assertEqual(config.device.parameter_map, expected_map)
                    self.assertEqual(config.simulator.parameters, PARAMETERS)

    def test_typed_nmos_and_pmos_decks_match_the_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._pdk(root)
            raw_path = root / "output.raw"
            with self._environment(root):
                for polarity in ["nmos", "pmos"]:
                    typed = load_config(
                        CONFIG_ROOT / f"ihp_sg13g2_lv_{polarity}_smoke.toml"
                    )
                    legacy = load_config(self._legacy_config(root, polarity))
                    for nf in [1, 2, 3, 4]:
                        with self.subTest(polarity=polarity, nf=nf):
                            typed_deck = _netlist(
                                typed,
                                length=0.8e-6,
                                vbs=-0.1 if polarity == "nmos" else 0.1,
                                finger_width=1.0e-6,
                                nf=nf,
                                parameters=PARAMETERS,
                                raw_path=raw_path,
                            )
                            legacy_deck = _netlist(
                                legacy,
                                length=0.8e-6,
                                vbs=-0.1 if polarity == "nmos" else 0.1,
                                finger_width=1.0e-6,
                                nf=nf,
                                parameters=PARAMETERS,
                                raw_path=raw_path,
                            )
                            self.assertEqual(typed_deck, legacy_deck)


if __name__ == "__main__":
    unittest.main()
