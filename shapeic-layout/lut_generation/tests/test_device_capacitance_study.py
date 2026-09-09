from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapeic_layout_generation.device_capacitance import (
    Admittance,
    Bias,
    Geometry,
)
from shapeic_layout_generation.device_capacitance_study import (
    Acceptance,
    BiasGrid,
    DeviceSample,
    PrimitiveStageOne,
    StageTwo,
    StudyConfig,
    StudyExecutionError,
    evaluate_bias_models,
    evaluate_geometry_bias_model,
    load_study_config,
    multilinear_interpolate,
    run_study,
)


class DeviceCapacitanceStudyTest(unittest.TestCase):
    def test_multilinear_interpolation_reproduces_a_six_dimensional_model(
        self,
    ) -> None:
        axes = tuple((0.0, 2.0) for _ in range(6))
        values = np.empty((2, 2, 2, 2, 2, 2, 2, 2))
        for indices in np.ndindex(2, 2, 2, 2, 2, 2):
            coordinates = tuple(
                axes[index][value] for index, value in enumerate(indices)
            )
            values[indices] = np.eye(2) * sum(
                (axis + 1.0) * coordinate
                for axis, coordinate in enumerate(coordinates)
            )
        point = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
        expected = np.eye(2) * sum(
            (axis + 1.0) * coordinate
            for axis, coordinate in enumerate(point)
        )
        np.testing.assert_allclose(
            multilinear_interpolate(axes, values, point),
            expected,
        )

    def test_stage_one_validates_bias_interpolation_separately_from_constant(
        self,
    ) -> None:
        geometry = Geometry(0.8e-6, 4.0e-6, 4)
        grid = BiasGrid(
            vbs=(-0.5, 0.0),
            vgs=(0.2, 0.4),
            vds=(0.3, 0.7),
            validation=(Bias(-0.4, 0.25, 0.4),),
        )
        samples = [
            self._sample("simplediffpair", geometry, bias)
            for bias in (*grid.calibration_points(), *grid.validation)
        ]
        rows = evaluate_bias_models(
            "simplediffpair",
            PrimitiveStageOne(geometry, grid),
            samples,
        )
        interpolated = next(row for row in rows if row.model == "bias_trilinear")
        constant = next(row for row in rows if row.model == "constant")
        self.assertLess(interpolated.capacitance_error, 1.0e-12)
        self.assertGreater(constant.capacitance_error, 0.0)

    def test_stage_two_validates_geometry_and_bias_interpolation(self) -> None:
        bias_grid = BiasGrid(
            vbs=(-0.5, 0.0),
            vgs=(0.2, 0.4),
            vds=(0.3, 0.7),
            validation=(Bias(-0.25, 0.3, 0.5),),
        )
        stage_two = StageTwo(
            lengths=(0.4e-6, 0.8e-6),
            finger_widths=(1.0e-6, 3.0e-6),
            finger_counts=(1, 3),
            validation_geometry=(Geometry(0.6e-6, 2.0e-6, 2),),
        )
        samples = []
        for geometry in (
            *stage_two.calibration_geometry(),
            *stage_two.validation_geometry,
        ):
            for bias in (*bias_grid.calibration_points(), *bias_grid.validation):
                samples.append(self._sample("simplediffpair", geometry, bias))
        rows = evaluate_geometry_bias_model(
            "simplediffpair",
            PrimitiveStageOne(stage_two.validation_geometry[0], bias_grid),
            stage_two,
            samples,
        )
        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0].capacitance_error, 1.0e-12)
        self.assertLess(rows[0].conductance_error, 1.0e-12)

    def test_loads_study_configuration_without_running_external_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = root / "physical.toml"
            physical.write_text(
                f'''pdk = "ihp-sg13g2"
layout_policy = "test"
primitives = ["simplediffpair", "currentmirror"]
[output]
path = "{root / 'physical.npz'}"
[sweep]
length = [0.4, 0.8]
finger_width = [1.0, 2.0]
nf = [1, 2]
[extraction]
backend = "synthetic"
work_directory = "{root / 'work'}"
''',
                encoding="ascii",
            )
            study = root / "study.toml"
            study.write_text(
                self._study_toml(physical, root / "output"),
                encoding="ascii",
            )
            adapter = SimpleNamespace(
                name="test-pdk",
                primitives=("simplediffpair", "currentmirror"),
                workers=2,
                definition=lambda primitive: SimpleNamespace(model=primitive),
                simulator_for=lambda primitive: SimpleNamespace(
                    frequencies_hz=(1.0e6, 1.0e7)
                ),
            )
            with (
                patch(
                    "shapeic_layout_generation.device_capacitance_study.load_config",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "shapeic_layout_generation.device_capacitance_study."
                    "CellKitDeviceCapacitanceAdapter",
                    return_value=adapter,
                ),
            ):
                config = load_study_config(study)
        self.assertEqual(config.adapter.name, "test-pdk")
        self.assertEqual(
            config.adapter.definition("simplediffpair").model,
            "simplediffpair",
        )
        self.assertEqual(
            config.adapter.simulator_for("simplediffpair").frequencies_hz,
            (1.0e6, 1.0e7),
        )
        self.assertEqual(config.stage_two.finger_counts, (1, 3))

    def test_failure_keeps_machine_readable_and_human_readable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class FailingAdapter:
                name = "ihp-sg13g2"
                physical = SimpleNamespace(source_path=Path("/physical.toml"))
                primitives = ("simplediffpair",)
                workers = 1

                @staticmethod
                def simulator_for(_primitive):
                    return SimpleNamespace(frequencies_hz=(1.0e6, 1.0e7))

                @staticmethod
                def validate_environment() -> None:
                    raise RuntimeError("deliberate environment failure")

            config = StudyConfig(
                source_path=Path("/study.toml"),
                output_root=root,
                adapter=FailingAdapter(),  # type: ignore[arg-type]
                acceptance=Acceptance(0.01, 0.05),
                stage_one={},
                stage_two=StageTwo((1.0, 2.0), (1.0, 2.0), (1, 2), ()),
            )
            with (
                patch(
                    "shapeic_layout_generation.device_capacitance_study."
                    "load_study_config",
                    return_value=config,
                ),
                patch(
                    "shapeic_layout_generation.device_capacitance_study.time.time_ns",
                    return_value=123,
                ),
            ):
                with self.assertRaises(StudyExecutionError) as caught:
                    run_study(Path("ignored.toml"))
            output = caught.exception.output_root
            status = json.loads((output / "status.json").read_text())
            report = (output / "report.md").read_text()
            self.assertEqual(status["status"], "failed")
            self.assertIn(
                "deliberate environment failure",
                status["error"]["message"],
            )
            self.assertIn("Status: FAILED", report)
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "device_delta_samples.npz").is_file())

    @staticmethod
    def _sample(primitive: str, geometry: Geometry, bias: Bias) -> DeviceSample:
        base = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
        scale = (
            geometry.length * 1.0e6
            + 2.0 * geometry.finger_width * 1.0e6
            + 0.5 * geometry.nf
            + 3.0 * bias.vbs
            + 4.0 * bias.vgs
            + 5.0 * bias.vds
        )
        zero = np.zeros((2, 2))
        aggregate = Admittance(zero, zero, 0.0)
        pex = Admittance(base * scale * 1.0e-6, base * scale * 1.0e-15, 0.0)
        return DeviceSample(primitive, geometry, bias, aggregate, pex)

    @staticmethod
    def _study_toml(physical: Path, output: Path) -> str:
        primitive = '''
[stage1.{name}.geometry]
length = 0.4
finger_width = 1.0
nf = 1
[stage1.{name}.calibration]
vbs = [-0.5, 0.0]
vgs = [0.2, 0.4]
vds = [0.3, 0.7]
[stage1.{name}.validation]
points = [[-0.25, 0.3, 0.5]]
'''
        return f'''pdk = "ihp-sg13g2"
physical_config = "{physical}"
[output]
root = "{output}"
[simulator]
binary = "ngspice"
model_library = "/model.lib"
library_section = "mos_tt"
osdi_paths = ["/psp.osdi"]
frequencies_hz = [1.0e6, 1.0e7]
workers = 2
[acceptance]
frequency_consistency = 0.01
matrix_relative_error = 0.05
{primitive.format(name='simplediffpair')}
{primitive.format(name='currentmirror')}
[stage2]
length = [0.4, 0.8]
finger_width = [1.0, 3.0]
nf = [1, 3]
validation_geometry = [[0.6, 2.0, 2]]
'''


if __name__ == "__main__":
    unittest.main()
