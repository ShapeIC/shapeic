from __future__ import annotations

import math
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cellkit import ResolvedCellKit, load_cellkit
from .electrical_model import ElectricalMosModel, load_electrical_model


PORTS = {
    "simplediffpair": ("DP", "DN", "GP", "GN", "S", "B"),
    "currentmirror": ("DOUT", "DREF", "S", "B"),
}
MAX_FINGER_COUNT = 50


@dataclass(frozen=True)
class ExtractorConfig:
    backend: str
    magic_binary: str
    magic_rcfile: Path | None
    work_directory: Path
    keep_work: bool
    magic_startup_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class ElectricalModelsConfig:
    """Electrical LUT generator inputs used to build aggregate MOS devices."""

    nmos: ElectricalMosModel
    pmos: ElectricalMosModel

    def for_polarity(self, polarity: str) -> ElectricalMosModel:
        if polarity == "nmos":
            return self.nmos
        if polarity == "pmos":
            return self.pmos
        raise ValueError(f"unsupported MOS polarity '{polarity}'")


@dataclass(frozen=True)
class DeviceCorrectionBias:
    vbs: np.ndarray
    vgs: np.ndarray
    vds: np.ndarray


@dataclass(frozen=True)
class DeviceCorrectionConfig:
    backend: str
    finger_counts: np.ndarray
    biases: dict[str, DeviceCorrectionBias]
    binary: str
    model_library: Path | None
    library_section: str
    osdi_paths: tuple[Path, ...]
    temperature_c: float
    frequencies_hz: tuple[float, float]
    workers: int
    frequency_consistency: float
    primitive_finger_counts: dict[str, np.ndarray] | None = None

    def finger_counts_for(self, primitive: str) -> np.ndarray:
        if self.primitive_finger_counts is not None:
            override = self.primitive_finger_counts.get(primitive)
            if override is not None:
                return override
        return self.finger_counts


@dataclass(frozen=True)
class PrimitiveSweep:
    lengths: np.ndarray
    finger_widths: np.ndarray
    finger_counts: np.ndarray


@dataclass(frozen=True)
class GenerationConfig:
    source_path: Path
    output_path: Path
    pdk: str
    layout_policy: str
    lengths: np.ndarray
    finger_widths: np.ndarray
    finger_counts: np.ndarray
    primitives: tuple[str, ...]
    extractor: ExtractorConfig
    device_correction: DeviceCorrectionConfig | None
    pdk_revision: str | None = None
    cellkit: ResolvedCellKit | None = None
    electrical_models: ElectricalModelsConfig | None = None
    port_orders: dict[str, tuple[str, ...]] | None = None
    primitive_catalog_names: dict[str, str] | None = None
    primitive_layouts: dict[str, Any] | None = None
    primitive_sweeps: dict[str, PrimitiveSweep] | None = None

    def sweep_for(self, primitive: str) -> PrimitiveSweep:
        if self.primitive_sweeps is not None:
            override = self.primitive_sweeps.get(primitive)
            if override is not None:
                return override
        return PrimitiveSweep(
            lengths=self.lengths,
            finger_widths=self.finger_widths,
            finger_counts=self.finger_counts,
        )

    def port_order(self, primitive: str) -> tuple[str, ...]:
        if self.port_orders is not None:
            try:
                return self.port_orders[primitive]
            except KeyError as error:
                raise ValueError(f"unknown configured primitive '{primitive}'") from error
        try:
            return PORTS[primitive]
        except KeyError as error:
            raise ValueError(f"unknown legacy primitive '{primitive}'") from error

    def catalog_name(self, primitive: str) -> str:
        if self.primitive_catalog_names is None:
            return primitive
        return self.primitive_catalog_names[primitive]

    def primitive_layout(self, primitive: str):
        if self.primitive_layouts is None:
            return None
        try:
            return self.primitive_layouts[primitive]
        except KeyError as error:
            raise ValueError(f"unknown configured primitive '{primitive}'") from error


def load_config(path: Path) -> GenerationConfig:
    source = path.resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    if "cellkit" in raw or "physical" in raw or isinstance(raw.get("pdk"), dict):
        return _load_cellkit_config(source, raw)
    return _load_legacy_config(source, raw)


def _load_legacy_config(source: Path, raw: dict[str, Any]) -> GenerationConfig:
    output = _table(raw, "output")
    sweep = _table(raw, "sweep")
    extraction = _table(raw, "extraction")
    backend = str(extraction.get("backend", "magic")).lower()
    if backend != "synthetic":
        raise ValueError(
            "physical Magic generation requires [cellkit], [pdk], and "
            "[electrical_models] configuration"
        )

    primitives = tuple(str(value).lower() for value in raw.get("primitives", PORTS))
    if not primitives or len(primitives) != len(set(primitives)):
        raise ValueError("primitives must be a non-empty list without duplicates")
    unknown = sorted(set(primitives) - set(PORTS))
    if unknown:
        raise ValueError(f"unsupported primitives: {', '.join(unknown)}")

    lengths = np.asarray(sweep["length"], dtype=np.float64) * 1.0e-6
    finger_widths = (
        np.asarray(sweep["finger_width"], dtype=np.float64) * 1.0e-6
    )
    finger_counts = np.asarray(sweep["nf"], dtype=np.float64)
    config = GenerationConfig(
        source_path=source,
        output_path=_path(str(output["path"]), source.parent),
        pdk=str(raw.get("pdk", "synthetic")),
        layout_policy=str(
            raw.get("layout_policy", "synthetic-test-fixture")
        ),
        lengths=lengths,
        finger_widths=finger_widths,
        finger_counts=finger_counts,
        primitives=primitives,
        extractor=ExtractorConfig(
            backend=backend,
            magic_binary=str(extraction.get("magic_binary", "magic")),
            magic_rcfile=(
                _path(str(extraction["magic_rcfile"]), source.parent)
                if extraction.get("magic_rcfile")
                else None
            ),
            work_directory=_path(
                str(extraction.get("work_directory", "../work")), source.parent
            ),
            keep_work=bool(extraction.get("keep_work", False)),
        ),
        device_correction=_device_correction(
            raw.get("device_capacitance_correction"),
            source.parent,
            primitives,
        ),
        primitive_sweeps=_primitive_sweeps(
            sweep,
            primitives,
            lengths,
            finger_widths,
            finger_counts,
        ),
    )
    _validate(config)
    return config


def _load_cellkit_config(
    source: Path, raw: dict[str, Any]
) -> GenerationConfig:
    output = _table(raw, "output")
    sweep = _table(raw, "sweep")
    cellkit_raw = _table(raw, "cellkit")
    pdk_raw = _table(raw, "pdk")
    physical = _table(raw, "physical")
    electrical_raw = _table(raw, "electrical_models")

    cellkit_root = _path(_required_string(cellkit_raw, "root"), source.parent)
    pdk_name = _selector(pdk_raw.get("name"), "PDK")
    pdk_root = _path(_selector(pdk_raw.get("root"), "PDK_ROOT"), source.parent)
    resolved = load_cellkit(cellkit_root, pdk_root, pdk_name)
    technology = resolved.technology

    backend = str(physical.get("backend", "magic")).lower()
    if backend not in {"magic", "synthetic"}:
        raise ValueError("physical.backend must be 'magic' or 'synthetic'")
    primitive_names = tuple(
        str(value).lower()
        for value in raw.get("primitives", ("simplediffpair", "currentmirror"))
    )
    if not primitive_names or len(primitive_names) != len(set(primitive_names)):
        raise ValueError("primitives must be a non-empty list without duplicates")
    descriptors = {
        primitive: resolved.catalog.primitive_descriptor_for_lut(primitive)
        for primitive in primitive_names
    }
    port_orders = {
        primitive: descriptor.port_order
        for primitive, descriptor in descriptors.items()
    }
    catalog_names = {
        primitive: descriptor.catalog_name
        for primitive, descriptor in descriptors.items()
    }
    primitive_layouts = {
        primitive: resolved.catalog.primitive(descriptor.catalog_name)
        for primitive, descriptor in descriptors.items()
    }
    policies = {
        layout.provider.LAYOUT_POLICY for layout in primitive_layouts.values()
    }
    if len(policies) != 1:
        raise ValueError(
            "configured primitive PCells must use one common layout policy; found "
            + ", ".join(sorted(policies))
        )
    provider_policy = next(iter(policies))
    requested_policy = physical.get("layout_policy")
    if requested_policy is not None and str(requested_policy) != provider_policy:
        raise ValueError(
            f"physical.layout_policy '{requested_policy}' does not match CellKit "
            f"provider policy '{provider_policy}'"
        )
    nmos_path = _path(_required_string(electrical_raw, "nmos"), source.parent)
    pmos_path = _path(_required_string(electrical_raw, "pmos"), source.parent)
    electrical_models = ElectricalModelsConfig(
        nmos=load_electrical_model(
            nmos_path, pdk=pdk_name, pdk_directory=resolved.pdk_root
        ),
        pmos=load_electrical_model(
            pmos_path, pdk=pdk_name, pdk_directory=resolved.pdk_root
        ),
    )
    revisions = {
        model.revision
        for model in (electrical_models.nmos, electrical_models.pmos)
        if model.revision is not None
    }
    if len(revisions) > 1:
        raise ValueError("NMOS and PMOS electrical models declare different PDK revisions")
    inferred_revision = next(iter(revisions), technology.revision)
    lengths = np.asarray(sweep["length"], dtype=np.float64) * 1.0e-6
    finger_widths = (
        np.asarray(sweep["finger_width"], dtype=np.float64) * 1.0e-6
    )
    finger_counts = np.asarray(sweep["nf"], dtype=np.float64)
    config = GenerationConfig(
        source_path=source,
        output_path=_path(str(output["path"]), source.parent),
        pdk=pdk_name,
        pdk_revision=str(pdk_raw.get("revision", inferred_revision)),
        layout_policy=provider_policy,
        lengths=lengths,
        finger_widths=finger_widths,
        finger_counts=finger_counts,
        primitives=primitive_names,
        extractor=ExtractorConfig(
            backend=backend,
            magic_binary=str(physical.get("magic_binary", "magic")),
            magic_rcfile=technology.magic_rcfile,
            work_directory=_path(
                str(physical.get("work_directory", "../work")), source.parent
            ),
            keep_work=bool(physical.get("keep_work", False)),
            magic_startup_commands=tuple(technology.magic_startup_commands),
        ),
        device_correction=_device_correction(
            raw.get("device_capacitance_correction"),
            source.parent,
            primitive_names,
        ),
        cellkit=resolved,
        electrical_models=electrical_models,
        port_orders=port_orders,
        primitive_catalog_names=catalog_names,
        primitive_layouts=primitive_layouts,
        primitive_sweeps=_primitive_sweeps(
            sweep,
            primitive_names,
            lengths,
            finger_widths,
            finger_counts,
        ),
    )
    _validate(config)
    return config


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] table")
    return value


def _required_string(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _selector(value: object, environment_name: str) -> str:
    if value is None:
        value = os.environ.get(environment_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{environment_name} must be set in the environment or in [pdk]"
        )
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(
            f"{environment_name} references an undefined environment variable"
        )
    return expanded


def _path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"path contains an undefined environment variable: {value}")
    path = Path(expanded)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _primitive_sweeps(
    raw: dict[str, Any],
    primitives: tuple[str, ...],
    lengths: np.ndarray,
    finger_widths: np.ndarray,
    finger_counts: np.ndarray,
) -> dict[str, PrimitiveSweep]:
    overrides = raw.get("primitives", {})
    if not isinstance(overrides, dict):
        raise ValueError("sweep.primitives must be a table")
    unknown_primitives = sorted(set(overrides) - set(primitives))
    if unknown_primitives:
        raise ValueError(
            "sweep overrides reference unconfigured primitives: "
            + ", ".join(unknown_primitives)
        )
    resolved = {}
    allowed = {"length", "finger_width", "nf"}
    for primitive, value in overrides.items():
        if not isinstance(value, dict):
            raise ValueError(f"sweep.primitives.{primitive} must be a table")
        unknown_fields = sorted(set(value) - allowed)
        if unknown_fields:
            raise ValueError(
                f"sweep.primitives.{primitive} contains unsupported fields: "
                + ", ".join(unknown_fields)
            )
        if not value:
            raise ValueError(
                f"sweep.primitives.{primitive} must override at least one axis"
            )
        resolved[primitive] = PrimitiveSweep(
            lengths=(
                np.asarray(value["length"], dtype=np.float64) * 1.0e-6
                if "length" in value
                else lengths
            ),
            finger_widths=(
                np.asarray(value["finger_width"], dtype=np.float64) * 1.0e-6
                if "finger_width" in value
                else finger_widths
            ),
            finger_counts=(
                np.asarray(value["nf"], dtype=np.float64)
                if "nf" in value
                else finger_counts
            ),
        )
    return resolved


def _device_correction(
    raw: object,
    base: Path,
    primitives: tuple[str, ...],
) -> DeviceCorrectionConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("device_capacitance_correction must be a table")
    backend = str(raw.get("backend", "ngspice")).lower()
    if backend not in {"ngspice", "synthetic"}:
        raise ValueError(
            "device_capacitance_correction.backend must be 'ngspice' or "
            "'synthetic'"
        )
    finger_counts = np.asarray(raw.get("nf", ()), dtype=np.float64)
    biases = {}
    primitive_finger_counts = {}
    for primitive in primitives:
        value = raw.get(primitive)
        if not isinstance(value, dict):
            raise ValueError(
                f"missing [device_capacitance_correction.{primitive}] table"
            )
        biases[primitive] = DeviceCorrectionBias(
            vbs=np.asarray(value.get("vbs", ()), dtype=np.float64),
            vgs=np.asarray(value.get("vgs", ()), dtype=np.float64),
            vds=np.asarray(value.get("vds", ()), dtype=np.float64),
        )
        if "nf" in value:
            primitive_finger_counts[primitive] = np.asarray(
                value["nf"], dtype=np.float64
            )
    model_library = raw.get("model_library")
    frequencies = tuple(
        float(value)
        for value in raw.get("frequencies_hz", (1.0e6, 1.0e7))
    )
    return DeviceCorrectionConfig(
        backend=backend,
        finger_counts=finger_counts,
        biases=biases,
        binary=str(raw.get("binary", "ngspice")),
        model_library=(
            _path(str(model_library), base) if model_library is not None else None
        ),
        library_section=str(raw.get("library_section", "mos_tt")),
        osdi_paths=tuple(
            _path(str(value), base) for value in raw.get("osdi_paths", ())
        ),
        temperature_c=float(raw.get("temperature_c", 27.0)),
        frequencies_hz=frequencies,  # type: ignore[arg-type]
        workers=int(raw.get("workers", 1)),
        frequency_consistency=float(raw.get("frequency_consistency", 0.01)),
        primitive_finger_counts=primitive_finger_counts,
    )


def _validate_axis(axis: np.ndarray, name: str) -> None:
    if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if axis.size > 1 and not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} must be strictly increasing")


def _validate(config: GenerationConfig) -> None:
    if config.output_path.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    if not config.pdk or not config.layout_policy:
        raise ValueError("pdk and layout_policy must not be empty")
    _validate_sweep(
        PrimitiveSweep(
            config.lengths,
            config.finger_widths,
            config.finger_counts,
        ),
        "sweep",
    )
    for primitive in config.primitives:
        sweep = config.sweep_for(primitive)
        _validate_sweep(sweep, f"{primitive} sweep")
        layout = config.primitive_layout(primitive)
        limits = getattr(layout, "geometry_limits", None)
        if limits is not None:
            if limits.required_nf is not None and np.any(
                sweep.finger_counts != limits.required_nf
            ):
                raise ValueError(
                    f"{primitive} sweep requires nf={limits.required_nf}"
                )
            if limits.max_finger_width_m is not None and np.any(
                sweep.finger_widths > limits.max_finger_width_m
            ):
                raise ValueError(
                    f"{primitive} sweep requires finger_width <= "
                    f"{limits.max_finger_width_m} m"
                )
    if config.extractor.backend == "magic":
        if shutil.which(config.extractor.magic_binary) is None:
            raise FileNotFoundError(
                f"Magic binary not found: {config.extractor.magic_binary}"
            )
        if config.extractor.magic_rcfile is None:
            raise ValueError("magic_rcfile is required by the magic backend")
        if not config.extractor.magic_rcfile.is_file():
            raise FileNotFoundError(
                f"Magic rcfile not found: {config.extractor.magic_rcfile}"
            )
    correction = config.device_correction
    if correction is None:
        return
    _validate_nf_axis(correction.finger_counts, "device correction nf")
    for primitive, bias in correction.biases.items():
        correction_nf = correction.finger_counts_for(primitive)
        _validate_nf_axis(correction_nf, f"{primitive} device correction nf")
        if not set(correction_nf).issubset(
            set(config.sweep_for(primitive).finger_counts)
        ):
            raise ValueError(
                f"{primitive} device correction nf samples must be a subset "
                "of the physical nf sweep for that primitive"
            )
        for name, axis in (
            ("vbs", bias.vbs),
            ("vgs", bias.vgs),
            ("vds", bias.vds),
        ):
            _validate_axis(axis, f"{primitive} device correction {name}")
    if (
        len(correction.frequencies_hz) != 2
        or correction.frequencies_hz[0] <= 0.0
        or not math.isclose(
            correction.frequencies_hz[1] / correction.frequencies_hz[0],
            10.0,
            rel_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "device correction frequencies_hz must contain one increasing decade"
        )
    if correction.workers < 1:
        raise ValueError("device correction workers must be positive")
    if not math.isfinite(correction.temperature_c):
        raise ValueError("device correction temperature must be finite")
    if not 0.0 < correction.frequency_consistency < 1.0:
        raise ValueError("device correction frequency_consistency must be in (0, 1)")
    if correction.backend == "ngspice":
        if config.extractor.backend != "magic":
            raise ValueError(
                "the ngspice device correction backend requires Magic physical "
                "extraction"
            )
        if shutil.which(correction.binary) is None:
            raise FileNotFoundError(
                f"NGSpice binary not found: {correction.binary}"
            )
        if config.electrical_models is None and correction.model_library is None:
            raise ValueError(
                "legacy device correction requires model_library"
            )
        legacy_paths = (
            (correction.model_library, *correction.osdi_paths)
            if config.electrical_models is None
            else ()
        )
        for path in legacy_paths:
            assert path is not None
            if not path.is_file():
                raise FileNotFoundError(path)


def _validate_sweep(sweep: PrimitiveSweep, context: str) -> None:
    for name, axis in (
        ("length", sweep.lengths),
        ("finger_width", sweep.finger_widths),
        ("nf", sweep.finger_counts),
    ):
        _validate_axis(axis, f"{context} {name}")
    if np.any(sweep.lengths <= 0) or np.any(sweep.finger_widths <= 0):
        raise ValueError(f"{context} length and finger_width must be positive")
    _validate_nf_axis(sweep.finger_counts, f"{context} nf")


def _validate_nf_axis(axis: np.ndarray, name: str) -> None:
    _validate_axis(axis, name)
    if np.any(axis < 1) or np.any(axis > MAX_FINGER_COUNT):
        raise ValueError(f"{name} samples must be in [1, {MAX_FINGER_COUNT}]")
    if np.any(np.mod(axis, 1.0) != 0.0):
        raise ValueError(f"{name} samples must be integers")
