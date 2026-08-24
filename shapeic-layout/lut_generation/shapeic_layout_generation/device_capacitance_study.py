from __future__ import annotations

import itertools
import json
import math
import os
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import load_config
from .device_capacitance_adapter import DeviceCapacitanceAdapter
from .device_capacitance import (
    Admittance,
    Bias,
    Geometry,
    SimulatorConfig,
    extract_port_admittance,
    relative_matrix_error,
)
from .ihp_device_capacitance import IhpSg13g2DeviceCapacitanceAdapter


@dataclass(frozen=True)
class BiasGrid:
    vbs: tuple[float, ...]
    vgs: tuple[float, ...]
    vds: tuple[float, ...]
    validation: tuple[Bias, ...]

    def calibration_points(self) -> tuple[Bias, ...]:
        return tuple(
            Bias(*coordinates)
            for coordinates in itertools.product(self.vbs, self.vgs, self.vds)
        )


@dataclass(frozen=True)
class PrimitiveStageOne:
    geometry: Geometry
    bias: BiasGrid


@dataclass(frozen=True)
class StageTwo:
    lengths: tuple[float, ...]
    finger_widths: tuple[float, ...]
    finger_counts: tuple[int, ...]
    validation_geometry: tuple[Geometry, ...]

    def calibration_geometry(self) -> tuple[Geometry, ...]:
        return tuple(
            Geometry(*coordinates)
            for coordinates in itertools.product(
                self.lengths,
                self.finger_widths,
                self.finger_counts,
            )
        )


@dataclass(frozen=True)
class Acceptance:
    frequency_consistency: float
    matrix_relative_error: float


@dataclass(frozen=True)
class StudyConfig:
    source_path: Path
    output_root: Path
    adapter: DeviceCapacitanceAdapter
    acceptance: Acceptance
    stage_one: dict[str, PrimitiveStageOne]
    stage_two: StageTwo


@dataclass(frozen=True)
class DeviceSample:
    primitive: str
    geometry: Geometry
    bias: Bias
    aggregate: Admittance
    pex: Admittance

    @property
    def delta_conductance(self) -> np.ndarray:
        return self.pex.conductance - self.aggregate.conductance

    @property
    def delta_capacitance(self) -> np.ndarray:
        return self.pex.capacitance - self.aggregate.capacitance


@dataclass(frozen=True)
class PredictionRow:
    stage: str
    primitive: str
    geometry: Geometry
    bias: Bias
    model: str
    conductance_error: float
    capacitance_error: float
    frequency_consistency: float


@dataclass(frozen=True)
class StudyResult:
    output_root: Path
    stage_one_passed: bool
    stage_two_ran: bool
    stage_two_passed: bool | None


class StudyExecutionError(RuntimeError):
    def __init__(self, output_root: Path, cause: Exception) -> None:
        self.output_root = output_root
        self.cause = cause
        super().__init__(f"device-capacitance study failed: {cause}")


def load_study_config(path: Path) -> StudyConfig:
    source = path.resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    pdk = str(raw.get("pdk", "ihp-sg13g2")).casefold()
    if pdk != "ihp-sg13g2":
        raise ValueError(f"unsupported device-capacitance study PDK '{pdk}'")

    physical_path = _path(_required_string(raw, "physical_config"), source.parent)
    physical = load_config(physical_path)
    simulator_raw = _required_table(raw, "simulator")
    simulator = SimulatorConfig(
        binary=str(simulator_raw.get("binary", "ngspice")),
        model_library=_path(
            _required_string(simulator_raw, "model_library"), source.parent
        ),
        library_section=str(simulator_raw.get("library_section", "mos_tt")),
        osdi_paths=tuple(
            _path(str(value), source.parent)
            for value in simulator_raw.get("osdi_paths", ())
        ),
        temperature_c=float(simulator_raw.get("temperature_c", 27.0)),
        frequencies_hz=tuple(
            float(value)
            for value in simulator_raw.get(
                "frequencies_hz",
                (1.0e6, 1.0e7),
            )
        ),  # type: ignore[arg-type]
    )
    adapter = IhpSg13g2DeviceCapacitanceAdapter(
        physical,
        simulator,
        workers=int(simulator_raw.get("workers", 1)),
    )
    stage_one_raw = _required_table(raw, "stage1")
    stage_one = {
        primitive: _primitive_stage_one(
            _required_table(stage_one_raw, primitive),
            primitive,
        )
        for primitive in adapter.primitives
    }
    stage_two_raw = _required_table(raw, "stage2")
    stage_two = StageTwo(
        lengths=_micrometers(stage_two_raw, "length"),
        finger_widths=_micrometers(stage_two_raw, "finger_width"),
        finger_counts=_finger_counts(stage_two_raw, "nf"),
        validation_geometry=tuple(
            Geometry(
                float(row[0]) * 1.0e-6,
                float(row[1]) * 1.0e-6,
                _integer(row[2], "validation geometry nf"),
            )
            for row in _rows(stage_two_raw, "validation_geometry", 3)
        ),
    )
    acceptance_raw = _required_table(raw, "acceptance")
    config = StudyConfig(
        source_path=source,
        output_root=_path(
            _required_string(_required_table(raw, "output"), "root"),
            source.parent,
        ),
        adapter=adapter,
        acceptance=Acceptance(
            frequency_consistency=float(
                acceptance_raw.get("frequency_consistency", 0.01)
            ),
            matrix_relative_error=float(
                acceptance_raw.get("matrix_relative_error", 0.05)
            ),
        ),
        stage_one=stage_one,
        stage_two=stage_two,
    )
    _validate_study_config(config)
    return config


def run_study(config_path: Path, *, stage: str = "auto") -> StudyResult:
    if stage not in {"1", "2", "auto"}:
        raise ValueError("stage must be '1', '2', or 'auto'")
    config = load_study_config(config_path)
    output_root = config.output_root / str(time.time_ns())
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "config.json").write_text(
        json.dumps(_config_manifest(config, stage), indent=2) + "\n",
        encoding="ascii",
    )

    samples: list[DeviceSample] = []
    rows: list[PredictionRow] = []
    stage_one_passed: bool | None = None
    stage_two_ran = False
    stage_two_passed: bool | None = None
    try:
        config.adapter.validate_environment()
        for primitive, primitive_config in config.stage_one.items():
            primitive_samples = _characterize_geometry(
                config,
                primitive,
                primitive_config.geometry,
                (
                    *primitive_config.bias.calibration_points(),
                    *primitive_config.bias.validation,
                ),
                output_root / "stage1" / primitive,
            )
            samples.extend(primitive_samples)
            rows.extend(
                evaluate_bias_models(
                    primitive,
                    primitive_config,
                    primitive_samples,
                )
            )
        stage_one_passed = _stage_passed(
            [row for row in rows if row.stage == "stage1"],
            config.acceptance,
            "bias_trilinear",
        )

        stage_two_ran = stage == "2" or (stage == "auto" and stage_one_passed)
        if stage_two_ran:
            for primitive, primitive_config in config.stage_one.items():
                primitive_samples: list[DeviceSample] = []
                geometries = tuple(
                    dict.fromkeys(
                        (
                            *config.stage_two.calibration_geometry(),
                            *config.stage_two.validation_geometry,
                        )
                    )
                )
                for geometry in geometries:
                    geometry_samples = _characterize_geometry(
                        config,
                        primitive,
                        geometry,
                        (
                            *primitive_config.bias.calibration_points(),
                            *primitive_config.bias.validation,
                        ),
                        output_root
                        / "stage2"
                        / primitive
                        / _geometry_name(geometry),
                    )
                    samples.extend(geometry_samples)
                    primitive_samples.extend(geometry_samples)
                stage_two_rows = evaluate_geometry_bias_model(
                    primitive,
                    primitive_config,
                    config.stage_two,
                    primitive_samples,
                )
                rows.extend(stage_two_rows)
            stage_two_passed = _stage_passed(
                [row for row in rows if row.stage == "stage2"],
                config.acceptance,
                "geometry_bias_multilinear",
            )

        _write_samples(output_root / "device_delta_samples.npz", samples)
        _write_rows(output_root / "summary.csv", rows)
        _write_report(
            output_root / "report.md",
            config,
            rows,
            stage_one_passed,
            stage_two_ran,
            stage_two_passed,
        )
        _write_status(output_root, "complete", None)
        return StudyResult(
            output_root,
            stage_one_passed,
            stage_two_ran,
            stage_two_passed,
        )
    except Exception as error:
        _write_samples(output_root / "device_delta_samples.npz", samples)
        _write_rows(output_root / "summary.csv", rows)
        _write_report(
            output_root / "report.md",
            config,
            rows,
            stage_one_passed,
            stage_two_ran,
            stage_two_passed,
            failure=error,
        )
        _write_status(output_root, "failed", error)
        raise StudyExecutionError(output_root, error) from error


def evaluate_bias_models(
    primitive: str,
    config: PrimitiveStageOne,
    samples: list[DeviceSample],
) -> list[PredictionRow]:
    calibration_biases = config.bias.calibration_points()
    calibration = {
        sample.bias: sample
        for sample in samples
        if sample.geometry == config.geometry
        and sample.bias in calibration_biases
    }
    expected_count = len(calibration_biases)
    if len(calibration) != expected_count:
        raise ValueError(
            f"{primitive} Stage 1 has {len(calibration)} calibration samples, "
            f"expected {expected_count}"
        )
    constant_g = np.mean(
        [sample.delta_conductance for sample in calibration.values()], axis=0
    )
    constant_c = np.mean(
        [sample.delta_capacitance for sample in calibration.values()], axis=0
    )
    port_count = next(iter(calibration.values())).delta_capacitance.shape[0]
    shape = (
        len(config.bias.vbs),
        len(config.bias.vgs),
        len(config.bias.vds),
        port_count,
        port_count,
    )
    grid_g = np.empty(shape)
    grid_c = np.empty(shape)
    for indices in itertools.product(
        range(len(config.bias.vbs)),
        range(len(config.bias.vgs)),
        range(len(config.bias.vds)),
    ):
        bias = Bias(
            config.bias.vbs[indices[0]],
            config.bias.vgs[indices[1]],
            config.bias.vds[indices[2]],
        )
        grid_g[indices] = calibration[bias].delta_conductance
        grid_c[indices] = calibration[bias].delta_capacitance

    axes = (config.bias.vbs, config.bias.vgs, config.bias.vds)
    rows = []
    for sample in samples:
        if (
            sample.geometry != config.geometry
            or sample.bias not in config.bias.validation
        ):
            continue
        point = (sample.bias.vbs, sample.bias.vgs, sample.bias.vds)
        for model, predicted_g, predicted_c in (
            ("constant", constant_g, constant_c),
            (
                "bias_trilinear",
                multilinear_interpolate(axes, grid_g, point),
                multilinear_interpolate(axes, grid_c, point),
            ),
        ):
            rows.append(
                _prediction_row(
                    "stage1",
                    sample,
                    model,
                    predicted_g,
                    predicted_c,
                )
            )
    return rows


def evaluate_geometry_bias_model(
    primitive: str,
    primitive_config: PrimitiveStageOne,
    stage_two: StageTwo,
    samples: list[DeviceSample],
) -> list[PredictionRow]:
    calibration_geometry = stage_two.calibration_geometry()
    calibration_bias = primitive_config.bias.calibration_points()
    lookup = {
        (sample.geometry, sample.bias): sample
        for sample in samples
        if sample.geometry in calibration_geometry
        and sample.bias in calibration_bias
    }
    expected_count = len(calibration_geometry) * len(calibration_bias)
    if len(lookup) != expected_count:
        raise ValueError(
            f"{primitive} Stage 2 has {len(lookup)} calibration samples, "
            f"expected {expected_count}"
        )
    port_count = next(iter(lookup.values())).delta_capacitance.shape[0]
    grid_shape = (
        len(stage_two.lengths),
        len(stage_two.finger_widths),
        len(stage_two.finger_counts),
        len(primitive_config.bias.vbs),
        len(primitive_config.bias.vgs),
        len(primitive_config.bias.vds),
        port_count,
        port_count,
    )
    grid_g = np.empty(grid_shape)
    grid_c = np.empty(grid_shape)
    for indices in itertools.product(
        range(len(stage_two.lengths)),
        range(len(stage_two.finger_widths)),
        range(len(stage_two.finger_counts)),
        range(len(primitive_config.bias.vbs)),
        range(len(primitive_config.bias.vgs)),
        range(len(primitive_config.bias.vds)),
    ):
        geometry = Geometry(
            stage_two.lengths[indices[0]],
            stage_two.finger_widths[indices[1]],
            stage_two.finger_counts[indices[2]],
        )
        bias = Bias(
            primitive_config.bias.vbs[indices[3]],
            primitive_config.bias.vgs[indices[4]],
            primitive_config.bias.vds[indices[5]],
        )
        sample = lookup[(geometry, bias)]
        grid_g[indices] = sample.delta_conductance
        grid_c[indices] = sample.delta_capacitance

    axes = (
        stage_two.lengths,
        stage_two.finger_widths,
        tuple(float(value) for value in stage_two.finger_counts),
        primitive_config.bias.vbs,
        primitive_config.bias.vgs,
        primitive_config.bias.vds,
    )
    rows = []
    for sample in samples:
        if (
            sample.geometry not in stage_two.validation_geometry
            or sample.bias not in primitive_config.bias.validation
        ):
            continue
        point = (
            sample.geometry.length,
            sample.geometry.finger_width,
            float(sample.geometry.nf),
            sample.bias.vbs,
            sample.bias.vgs,
            sample.bias.vds,
        )
        rows.append(
            _prediction_row(
                "stage2",
                sample,
                "geometry_bias_multilinear",
                multilinear_interpolate(axes, grid_g, point),
                multilinear_interpolate(axes, grid_c, point),
            )
        )
    return rows


def multilinear_interpolate(
    axes: tuple[tuple[float, ...], ...],
    values: np.ndarray,
    point: tuple[float, ...],
) -> np.ndarray:
    if len(axes) != len(point) or values.ndim < len(axes):
        raise ValueError("interpolation dimensions do not match")
    brackets = [
        _bracket(np.asarray(axis), coordinate)
        for axis, coordinate in zip(axes, point)
    ]
    output = np.zeros(values.shape[len(axes) :], dtype=np.float64)
    choices = [
        ((lower, 1.0),)
        if lower == upper
        else ((lower, 1.0 - weight), (upper, weight))
        for lower, upper, weight in brackets
    ]
    for corner in itertools.product(*choices):
        indices = tuple(index for index, _ in corner)
        weight = math.prod(corner_weight for _, corner_weight in corner)
        output += weight * values[indices]
    return output


def _characterize_geometry(
    config: StudyConfig,
    primitive: str,
    geometry: Geometry,
    biases: tuple[Bias, ...],
    output_root: Path,
) -> list[DeviceSample]:
    prepared = config.adapter.prepare_geometry(
        primitive,
        geometry,
        output_root,
    )
    definition = config.adapter.definition(primitive)
    unique_biases = tuple(dict.fromkeys(biases))

    def characterize(item: tuple[int, Bias]) -> DeviceSample:
        index, bias = item
        bias_root = output_root / f"bias_{index:04d}"
        pex = extract_port_admittance(
            config.adapter.simulator,
            definition,
            bias,
            prepared.pex_path,
            prepared.pex_subcircuit,
            bias_root / "pex",
        )
        aggregate = extract_port_admittance(
            config.adapter.simulator,
            definition,
            bias,
            prepared.aggregate_path,
            prepared.aggregate_subcircuit,
            bias_root / "aggregate",
        )
        sample = DeviceSample(primitive, geometry, bias, aggregate, pex)
        _write_bias_sample(bias_root / "matrices.npz", sample)
        return sample

    with ThreadPoolExecutor(max_workers=config.adapter.workers) as executor:
        return list(executor.map(characterize, enumerate(unique_biases)))


def _prediction_row(
    stage: str,
    sample: DeviceSample,
    model: str,
    predicted_g: np.ndarray,
    predicted_c: np.ndarray,
) -> PredictionRow:
    return PredictionRow(
        stage=stage,
        primitive=sample.primitive,
        geometry=sample.geometry,
        bias=sample.bias,
        model=model,
        conductance_error=relative_matrix_error(
            predicted_g,
            sample.delta_conductance,
        ),
        capacitance_error=relative_matrix_error(
            predicted_c,
            sample.delta_capacitance,
        ),
        frequency_consistency=max(
            sample.aggregate.frequency_consistency,
            sample.pex.frequency_consistency,
        ),
    )


def _write_bias_sample(path: Path, sample: DeviceSample) -> None:
    np.savez_compressed(
        path,
        coordinates=np.asarray(
            (
                sample.geometry.length,
                sample.geometry.finger_width,
                sample.geometry.nf,
                sample.bias.vbs,
                sample.bias.vgs,
                sample.bias.vds,
            )
        ),
        aggregate_g=sample.aggregate.conductance,
        aggregate_c=sample.aggregate.capacitance,
        pex_g=sample.pex.conductance,
        pex_c=sample.pex.capacitance,
        delta_g=sample.delta_conductance,
        delta_c=sample.delta_capacitance,
    )


def _write_samples(path: Path, samples: list[DeviceSample]) -> None:
    arrays: dict[str, np.ndarray] = {}
    primitives = tuple(dict.fromkeys(sample.primitive for sample in samples))
    for primitive in primitives:
        selected = [sample for sample in samples if sample.primitive == primitive]
        arrays[f"{primitive}/coordinates"] = np.asarray(
            [
                (
                    sample.geometry.length,
                    sample.geometry.finger_width,
                    sample.geometry.nf,
                    sample.bias.vbs,
                    sample.bias.vgs,
                    sample.bias.vds,
                )
                for sample in selected
            ]
        )
        for name, getter in (
            ("aggregate_g", lambda sample: sample.aggregate.conductance),
            ("aggregate_c", lambda sample: sample.aggregate.capacitance),
            ("pex_g", lambda sample: sample.pex.conductance),
            ("pex_c", lambda sample: sample.pex.capacitance),
            ("delta_g", lambda sample: sample.delta_conductance),
            ("delta_c", lambda sample: sample.delta_capacitance),
        ):
            arrays[f"{primitive}/{name}"] = np.stack(
                [getter(sample) for sample in selected]
            )
    np.savez_compressed(path, **arrays)


def _write_rows(path: Path, rows: list[PredictionRow]) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(
            "stage,primitive,length_m,finger_width_m,nf,vbs_v,vgs_v,vds_v,"
            "model,conductance_relative_error,capacitance_relative_error,"
            "frequency_consistency\n"
        )
        for row in rows:
            handle.write(
                f"{row.stage},{row.primitive},{row.geometry.length:.17e},"
                f"{row.geometry.finger_width:.17e},{row.geometry.nf},"
                f"{row.bias.vbs:.17e},{row.bias.vgs:.17e},{row.bias.vds:.17e},"
                f"{row.model},{row.conductance_error:.17e},"
                f"{row.capacitance_error:.17e},"
                f"{row.frequency_consistency:.17e}\n"
            )


def _write_report(
    path: Path,
    config: StudyConfig,
    rows: list[PredictionRow],
    stage_one_passed: bool | None,
    stage_two_ran: bool,
    stage_two_passed: bool | None,
    *,
    failure: Exception | None = None,
) -> None:
    lines = [
        "# Shapeic device-capacitance study",
        "",
        f"- Status: {'FAILED' if failure else 'complete'}",
        f"- PDK adapter: `{config.adapter.name}`",
        "",
        "## Interpolation",
        "",
        "| stage | primitive | model | max C error | max frequency inconsistency |",
        "|---|---|---|---:|---:|",
    ]
    groups = sorted({(row.stage, row.primitive, row.model) for row in rows})
    for stage, primitive, model in groups:
        selected = [
            row
            for row in rows
            if (row.stage, row.primitive, row.model) == (stage, primitive, model)
        ]
        lines.append(
            f"| {stage} | {primitive} | {model} | "
            f"{max(row.capacitance_error for row in selected):.3%} | "
            f"{max(row.frequency_consistency for row in selected):.3%} |"
        )
    if not groups:
        lines.append("| - | - | - | - | - |")
    lines.extend(
        (
            "",
            "## Decision",
            "",
            f"- Stage 1: {_result_label(stage_one_passed)}",
            (
                f"- Stage 2: {_result_label(stage_two_passed)}"
                if stage_two_ran
                else "- Stage 2: not run"
            ),
            (
                "- Frequency consistency limit: "
                f"{config.acceptance.frequency_consistency:.3%}"
            ),
            (
                "- Matrix relative-error limit: "
                f"{config.acceptance.matrix_relative_error:.3%}"
            ),
        )
    )
    if failure is not None:
        lines.extend(("", "## Failure", "", f"`{type(failure).__name__}: {failure}`"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_status(output_root: Path, status: str, error: Exception | None) -> None:
    payload = {
        "format": "shapeic-device-capacitance-study-status",
        "version": 1,
        "status": status,
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    (output_root / "status.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="ascii",
        errors="replace",
    )


def _stage_passed(
    rows: list[PredictionRow],
    acceptance: Acceptance,
    preferred_model: str,
) -> bool:
    preferred = [row for row in rows if row.model == preferred_model]
    return (
        bool(preferred)
        and max(row.capacitance_error for row in preferred)
        <= acceptance.matrix_relative_error
        and max(row.frequency_consistency for row in preferred)
        <= acceptance.frequency_consistency
    )


def _bracket(axis: np.ndarray, coordinate: float) -> tuple[int, int, float]:
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("interpolation axis must be a non-empty vector")
    tolerance = max(abs(float(axis[0])), abs(float(axis[-1])), 1.0) * 1.0e-12
    if coordinate < axis[0] - tolerance or coordinate > axis[-1] + tolerance:
        raise ValueError(
            f"coordinate {coordinate} is outside [{axis[0]}, {axis[-1]}]"
        )
    if coordinate <= axis[0]:
        return 0, 0, 0.0
    if coordinate >= axis[-1]:
        last = axis.size - 1
        return last, last, 0.0
    upper = int(np.searchsorted(axis, coordinate, side="right"))
    lower = upper - 1
    weight = float((coordinate - axis[lower]) / (axis[upper] - axis[lower]))
    return lower, upper, weight


def _primitive_stage_one(
    raw: dict[str, object],
    primitive: str,
) -> PrimitiveStageOne:
    geometry_raw = _required_table(raw, "geometry")
    calibration = _required_table(raw, "calibration")
    validation = _required_table(raw, "validation")
    return PrimitiveStageOne(
        geometry=Geometry(
            float(geometry_raw["length"]) * 1.0e-6,
            float(geometry_raw["finger_width"]) * 1.0e-6,
            _integer(geometry_raw["nf"], f"{primitive} nf"),
        ),
        bias=BiasGrid(
            vbs=_float_tuple(calibration, "vbs"),
            vgs=_float_tuple(calibration, "vgs"),
            vds=_float_tuple(calibration, "vds"),
            validation=tuple(
                Bias(float(row[0]), float(row[1]), float(row[2]))
                for row in _rows(validation, "points", 3)
            ),
        ),
    )


def _validate_study_config(config: StudyConfig) -> None:
    frequencies = config.adapter.simulator.frequencies_hz
    if (
        len(frequencies) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in frequencies)
        or not math.isclose(frequencies[1] / frequencies[0], 10.0, rel_tol=1.0e-12)
    ):
        raise ValueError("simulator frequencies must be positive decade endpoints")
    if config.adapter.workers < 1:
        raise ValueError("simulator workers must be positive")
    for primitive, primitive_config in config.stage_one.items():
        geometry = primitive_config.geometry
        _validate_geometry(geometry, f"Stage 1 {primitive}")
        for axis in (
            primitive_config.bias.vbs,
            primitive_config.bias.vgs,
            primitive_config.bias.vds,
        ):
            _validate_axis(axis, f"Stage 1 {primitive} bias")
        for validation in primitive_config.bias.validation:
            for value, axis in zip(
                (validation.vbs, validation.vgs, validation.vds),
                (
                    primitive_config.bias.vbs,
                    primitive_config.bias.vgs,
                    primitive_config.bias.vds,
                ),
            ):
                if not axis[0] <= value <= axis[-1]:
                    raise ValueError(
                        f"Stage 1 {primitive} validation bias is outside its axes"
                    )
    for axis in (
        config.stage_two.lengths,
        config.stage_two.finger_widths,
        config.stage_two.finger_counts,
    ):
        _validate_axis(axis, "Stage 2 geometry")
    for geometry in config.stage_two.validation_geometry:
        _validate_geometry(geometry, "Stage 2 validation")
        for name, value, axis in (
            ("length", geometry.length, config.stage_two.lengths),
            ("finger_width", geometry.finger_width, config.stage_two.finger_widths),
            ("nf", geometry.nf, config.stage_two.finger_counts),
        ):
            if not axis[0] <= value <= axis[-1]:
                raise ValueError(
                    f"Stage 2 validation {name}={value} is outside its axis"
                )
    if not 0.0 < config.acceptance.frequency_consistency < 1.0:
        raise ValueError("frequency_consistency must be in (0, 1)")
    if not 0.0 < config.acceptance.matrix_relative_error < 1.0:
        raise ValueError("matrix_relative_error must be in (0, 1)")


def _validate_axis(axis, description: str) -> None:
    if len(axis) < 2 or any(
        not math.isfinite(float(value)) for value in axis
    ) or any(upper <= lower for lower, upper in zip(axis, axis[1:])):
        raise ValueError(f"{description} axis must be finite and increasing")


def _validate_geometry(geometry: Geometry, description: str) -> None:
    if (
        not math.isfinite(geometry.length)
        or not math.isfinite(geometry.finger_width)
        or geometry.length <= 0.0
        or geometry.finger_width <= 0.0
        or geometry.nf < 1
    ):
        raise ValueError(f"{description} geometry is invalid")


def _required_table(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _required_string(raw: dict[str, object], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{name}' must be a non-empty string")
    return value


def _float_tuple(raw: dict[str, object], name: str) -> tuple[float, ...]:
    value = raw.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty list")
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"'{name}' must contain finite values")
    return values


def _micrometers(raw: dict[str, object], name: str) -> tuple[float, ...]:
    return tuple(value * 1.0e-6 for value in _float_tuple(raw, name))


def _finger_counts(raw: dict[str, object], name: str) -> tuple[int, ...]:
    values = _float_tuple(raw, name)
    if any(value < 1 or value > 50 or not value.is_integer() for value in values):
        raise ValueError("Stage 2 nf values must be integers in [1, 50]")
    return tuple(int(value) for value in values)


def _rows(
    raw: dict[str, object],
    name: str,
    columns: int,
) -> tuple[tuple[object, ...], ...]:
    value = raw.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{name}' must be a non-empty list of rows")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"every '{name}' row must have {columns} values")
        rows.append(tuple(row))
    return tuple(rows)


def _integer(value: object, description: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{description} must be an integer")
    return int(number)


def _path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"path contains an undefined environment variable: {value}")
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _geometry_name(geometry: Geometry) -> str:
    return (
        f"l{geometry.length * 1.0e6:.3f}_"
        f"wf{geometry.finger_width * 1.0e6:.3f}_nf{geometry.nf}"
    ).replace(".", "p")


def _result_label(value: bool | None) -> str:
    if value is None:
        return "not completed"
    return "PASS" if value else "FAIL"


def _config_manifest(config: StudyConfig, stage: str) -> dict[str, object]:
    return {
        "format": "shapeic-device-capacitance-study",
        "version": 1,
        "source_config": str(config.source_path),
        "pdk_adapter": config.adapter.name,
        "physical_config": str(config.adapter.physical.source_path),
        "frequencies_hz": list(config.adapter.simulator.frequencies_hz),
        "requested_stage": stage,
        "acceptance": {
            "frequency_consistency": config.acceptance.frequency_consistency,
            "matrix_relative_error": config.acceptance.matrix_relative_error,
        },
    }
