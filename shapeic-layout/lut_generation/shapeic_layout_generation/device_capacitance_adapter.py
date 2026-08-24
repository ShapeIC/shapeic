from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import GenerationConfig
from .device_capacitance import (
    Geometry,
    PrimitiveDeviceDefinition,
    SimulatorConfig,
)


@dataclass(frozen=True)
class PreparedDeviceNetlists:
    pex_path: Path
    pex_subcircuit: str
    aggregate_path: Path
    aggregate_subcircuit: str


class DeviceCapacitanceAdapter(Protocol):
    """PDK boundary used by the PDK-independent characterization study."""

    name: str
    physical: GenerationConfig
    simulator: SimulatorConfig
    workers: int

    @property
    def primitives(self) -> tuple[str, ...]: ...

    def definition(self, primitive: str) -> PrimitiveDeviceDefinition: ...

    def validate_environment(self) -> None: ...

    def prepare_geometry(
        self,
        primitive: str,
        geometry: Geometry,
        output_root: Path,
    ) -> PreparedDeviceNetlists: ...

    def prepare_extracted_geometry(
        self,
        primitive: str,
        geometry: Geometry,
        raw_pex_path: Path,
        pex_subcircuit: str,
        output_root: Path,
    ) -> PreparedDeviceNetlists: ...
