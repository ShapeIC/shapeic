"""PDK-independent device-capacitance adapter backed by ShapeIC CellKit."""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import GenerationConfig
from .device_capacitance import (
    Geometry,
    MosInstance,
    PrimitiveDeviceDefinition,
    SimulatorConfig,
    aggregate_primitive_spice,
    mos_only_pex,
)
from .device_capacitance_adapter import PreparedDeviceNetlists
from .extractor import write_magic_pex


class CellKitDeviceCapacitanceAdapter:
    """Compose manifest topology, electrical TOMLs, and a technology provider."""

    def __init__(self, physical: GenerationConfig) -> None:
        if physical.cellkit is None or physical.electrical_models is None:
            raise ValueError("CellKit device correction requires resolved electrical models")
        correction = physical.device_correction
        if correction is None:
            raise ValueError("CellKit device correction requires correction settings")
        self.name = physical.pdk
        self.physical = physical
        self.workers = correction.workers
        self._technology = physical.cellkit.technology
        self._definitions = {}
        self._simulators = {}
        for primitive in physical.primitives:
            layout = physical.primitive_layout(primitive)
            if layout is None:
                raise ValueError(f"missing CellKit layout for '{primitive}'")
            model = physical.electrical_models.for_polarity(layout.polarity.value)
            self._definitions[primitive] = _definition(layout, model)
            self._simulators[primitive] = SimulatorConfig(
                binary=correction.binary,
                model_library=None,
                library_section="",
                osdi_paths=model.osdi_paths,
                temperature_c=correction.temperature_c,
                frequencies_hz=correction.frequencies_hz,
                model_statements=model.model_statements,
            )

    @property
    def primitives(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def definition(self, primitive: str) -> PrimitiveDeviceDefinition:
        try:
            return self._definitions[primitive]
        except KeyError as error:
            raise ValueError(f"unknown CellKit primitive '{primitive}'") from error

    def simulator_for(self, primitive: str) -> SimulatorConfig:
        try:
            return self._simulators[primitive]
        except KeyError as error:
            raise ValueError(f"unknown CellKit primitive '{primitive}'") from error

    def validate_environment(self) -> None:
        if self.physical.extractor.backend != "magic":
            raise ValueError("real device correction requires the Magic backend")
        correction = self.physical.device_correction
        assert correction is not None
        if shutil.which(correction.binary) is None:
            raise FileNotFoundError(f"NGSpice binary not found: {correction.binary}")
        if self.workers < 1:
            raise ValueError("simulator workers must be positive")

    def prepare_geometry(
        self, primitive: str, geometry: Geometry, output_root: Path
    ) -> PreparedDeviceNetlists:
        output_root.mkdir(parents=True, exist_ok=True)
        assert self.physical.cellkit is not None
        layout = self.physical.primitive_layout(primitive)
        if layout is None:
            raise ValueError(f"missing CellKit layout for '{primitive}'")
        rendered = layout.render(
            self.physical.cellkit.geometry(
                geometry.length, geometry.finger_width, geometry.nf
            )
        )
        gds_path = output_root / "primitive.gds"
        rendered.component.write_gds(gds_path)
        rcfile = self.physical.extractor.magic_rcfile
        if rcfile is None:
            raise ValueError("real device correction requires a Magic rcfile")
        magic = write_magic_pex(
            gds_path,
            rendered.cell_name,
            magic_binary=self.physical.extractor.magic_binary,
            magic_rcfile=rcfile,
            work_directory=output_root / "magic",
            magic_startup_commands=self.physical.extractor.magic_startup_commands,
        )
        return self.prepare_extracted_geometry(
            primitive,
            geometry,
            magic.spice_path,
            magic.subcircuit_name,
            output_root,
        )

    def prepare_extracted_geometry(
        self,
        primitive: str,
        geometry: Geometry,
        raw_pex_path: Path,
        pex_subcircuit: str,
        output_root: Path,
    ) -> PreparedDeviceNetlists:
        output_root.mkdir(parents=True, exist_ok=True)
        raw = raw_pex_path.read_text(encoding="utf-8")
        original_path = output_root / "primitive.pex.spice"
        original_path.write_text(raw, encoding="utf-8")
        normalized = self._technology.normalize_pex(raw, primitive)
        definition = self.definition(primitive)
        mos_only = mos_only_pex(
            normalized,
            definition,
            lambda fields: self._technology.normalize_mos_device(fields, primitive),
        )
        if _subcircuit_name(mos_only).casefold() != pex_subcircuit.casefold():
            raise ValueError(
                f"expected PEX subcircuit '{pex_subcircuit}', "
                f"found '{_subcircuit_name(mos_only)}'"
            )
        pex_path = output_root / "primitive.mos_only.spice"
        pex_path.write_text(mos_only, encoding="utf-8")
        aggregate_path = output_root / "primitive.aggregate.spice"
        aggregate_path.write_text(
            self.aggregate_spice(primitive, geometry), encoding="ascii"
        )
        return PreparedDeviceNetlists(
            pex_path=pex_path,
            pex_subcircuit=pex_subcircuit,
            aggregate_path=aggregate_path,
            aggregate_subcircuit=f"aggregate_{primitive}",
        )

    def aggregate_spice(self, primitive: str, geometry: Geometry) -> str:
        assert self.physical.electrical_models is not None
        layout = self.physical.primitive_layout(primitive)
        assert layout is not None
        model = self.physical.electrical_models.for_polarity(layout.polarity.value)
        return aggregate_primitive_spice(
            self.definition(primitive),
            geometry,
            lambda definition, point: model.spice_geometry(
                point.length, point.finger_width, point.nf
            ),
            parallel_fingers=(
                getattr(model, "capacitance_nf_mode", "simulate") == "linear"
            ),
        )


def _definition(layout, model) -> PrimitiveDeviceDefinition:
    roles = {}
    operating = next(
        branch
        for branch in layout.branches
        if branch.name == layout.operating_point_branch
    )
    for port, role in (
        (operating.drain, "vds"),
        (operating.gate, "vgs"),
        (operating.source, "zero"),
        (operating.bulk, "vbs"),
    ):
        roles[port] = role
    for branch in layout.branches:
        for port, role in (
            (branch.drain, "vds"),
            (branch.gate, "vgs"),
            (branch.source, "zero"),
            (branch.bulk, "vbs"),
        ):
            roles.setdefault(port, role)
    missing = [port for port in layout.port_order if port not in roles]
    if missing:
        raise ValueError(
            f"primitive '{layout.lut_primitive}' cannot bias ports: {', '.join(missing)}"
        )
    prefix = "X" if model.instance_kind == "subcircuit" else "M"
    instances = tuple(
        MosInstance(
            f"{prefix}{branch.name.upper()}",
            branch.drain,
            branch.gate,
            branch.source,
            branch.bulk,
        )
        for branch in layout.branches
    )
    return PrimitiveDeviceDefinition(
        name=layout.lut_primitive,
        ports=layout.port_order,
        model=model.name,
        instances=instances,
        bias_variables=tuple(roles[port] for port in layout.port_order),
    )


def _subcircuit_name(text: str) -> str:
    headers = [
        fields
        for line in text.splitlines()
        if (fields := line.split()) and fields[0].casefold() == ".subckt"
    ]
    if len(headers) != 1 or len(headers[0]) < 2:
        raise ValueError("PEX must contain exactly one flattened subcircuit")
    return headers[0][1]
