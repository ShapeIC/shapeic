from __future__ import annotations

import shutil
from pathlib import Path

from .config import GenerationConfig
from .device_capacitance_adapter import PreparedDeviceNetlists
from .device_capacitance import (
    Geometry,
    MosInstance,
    PrimitiveDeviceDefinition,
    SimulatorConfig,
    aggregate_primitive_spice,
    mos_only_pex,
)
from .extractor import write_magic_pex
from .ota_pex import validate_primitive_pex
from .pcell import write_primitive_gds

class IhpSg13g2DeviceCapacitanceAdapter:
    """All IHP-specific operations needed by the independent study."""

    name = "ihp-sg13g2"

    def __init__(
        self,
        physical: GenerationConfig,
        simulator: SimulatorConfig,
        workers: int,
    ) -> None:
        self.physical = physical
        self.simulator = simulator
        self.workers = workers
        self._definitions = {
            "simplediffpair": PrimitiveDeviceDefinition(
                name="simplediffpair",
                ports=("DP", "DN", "GP", "GN", "S", "B"),
                model="sg13_lv_nmos",
                instances=(
                    MosInstance("XDP1", "DP", "GP", "S", "B"),
                    MosInstance("XDP2", "DN", "GN", "S", "B"),
                ),
                bias_variables=(
                    "vds",
                    "vds",
                    "vgs",
                    "vgs",
                    "zero",
                    "vbs",
                ),
            ),
            "currentmirror": PrimitiveDeviceDefinition(
                name="currentmirror",
                ports=("DOUT", "DREF", "S", "B"),
                model="sg13_lv_pmos",
                instances=(
                    MosInstance("XCM1", "DOUT", "DREF", "S", "B"),
                    MosInstance("XCM2", "DREF", "DREF", "S", "B"),
                ),
                bias_variables=("vds", "vgs", "zero", "vbs"),
            ),
        }

    @property
    def primitives(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def definition(self, primitive: str) -> PrimitiveDeviceDefinition:
        try:
            return self._definitions[primitive]
        except KeyError as error:
            raise ValueError(f"unsupported IHP primitive '{primitive}'") from error

    def mos_only_pex(self, text: str, primitive: str) -> str:
        return mos_only_pex(
            text,
            self.definition(primitive),
            self._normalize_mos_device,
        )

    def aggregate_spice(self, primitive: str, geometry: Geometry) -> str:
        return aggregate_primitive_spice(
            self.definition(primitive),
            geometry,
            self._aggregate_parameters,
        )

    def validate_environment(self) -> None:
        if self.physical.pdk != self.name:
            raise ValueError(
                f"IHP adapter received physical PDK '{self.physical.pdk}'"
            )
        if self.physical.extractor.backend != "magic":
            raise ValueError("the IHP device study requires the Magic backend")
        if shutil.which(self.simulator.binary) is None:
            raise FileNotFoundError(
                f"NGSpice binary not found: {self.simulator.binary}"
            )
        for path in (
            self.simulator.model_library,
            *self.simulator.osdi_paths,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.workers < 1:
            raise ValueError("simulator workers must be positive")

    def prepare_geometry(
        self,
        primitive: str,
        geometry: Geometry,
        output_root: Path,
    ) -> PreparedDeviceNetlists:
        output_root.mkdir(parents=True, exist_ok=True)
        gds_path = output_root / "primitive.gds"
        cell_name = write_primitive_gds(
            primitive,
            geometry.length,
            geometry.finger_width,
            geometry.nf,
            gds_path,
        )
        if self.physical.extractor.magic_rcfile is None:
            raise ValueError("the IHP device study requires a Magic rcfile")
        magic = write_magic_pex(
            gds_path,
            cell_name,
            magic_binary=self.physical.extractor.magic_binary,
            magic_rcfile=self.physical.extractor.magic_rcfile,
            work_directory=output_root / "magic",
        )
        raw_pex = magic.spice_path.read_text(encoding="utf-8")
        original_path = output_root / "primitive.pex.spice"
        original_path.write_text(raw_pex, encoding="utf-8")
        topology = validate_primitive_pex(
            raw_pex,
            primitive,
            expected_subcircuit=magic.subcircuit_name,
        )
        pex_path = output_root / "primitive.mos_only.spice"
        pex_path.write_text(
            self.mos_only_pex(raw_pex, primitive),
            encoding="utf-8",
        )
        aggregate_path = output_root / "primitive.aggregate.spice"
        aggregate_path.write_text(
            self.aggregate_spice(primitive, geometry),
            encoding="ascii",
        )
        aggregate_subcircuit = f"aggregate_{primitive}"
        return PreparedDeviceNetlists(
            pex_path=pex_path,
            pex_subcircuit=topology.subcircuit_name,
            aggregate_path=aggregate_path,
            aggregate_subcircuit=aggregate_subcircuit,
        )

    @staticmethod
    def _normalize_mos_device(fields: list[str]) -> list[str]:
        normalized = fields.copy()
        normalized[4] = "B"
        return normalized

    @staticmethod
    def _aggregate_parameters(
        definition: PrimitiveDeviceDefinition,
        geometry: Geometry,
    ) -> str:
        total_width = geometry.finger_width * geometry.nf
        return (
            f"{definition.model} l={geometry.length:.17e} "
            f"w={total_width:.17e} ng={geometry.nf}"
        )
