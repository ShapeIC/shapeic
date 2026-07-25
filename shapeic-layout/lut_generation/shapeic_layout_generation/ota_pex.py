from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OTA_PORTS = ("VOUT", "VINP", "VINN", "IBIAS", "VDD", "VSS")
NMOS_MODEL = "sg13_lv_nmos"
PMOS_MODEL = "sg13_lv_pmos"


@dataclass(frozen=True)
class PexTopology:
    subcircuit_name: str
    transistor_count: int
    resistor_count: int
    capacitor_count: int


@dataclass(frozen=True)
class _Mos:
    drain: str
    gate: str
    source: str
    bulk: str
    model: str


def prepare_ota_pex(
    source: Path,
    output: Path,
    *,
    expected_subcircuit: str | None = None,
) -> PexTopology:
    """Normalize implicit IHP bulk nodes and validate the extracted OTA topology."""
    text = source.read_text(encoding="utf-8")
    normalized = normalize_bulk_nodes(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(normalized, encoding="utf-8")
    topology = validate_ota_pex(
        normalized,
        expected_subcircuit=expected_subcircuit,
    )
    return topology


def normalize_bulk_nodes(text: str) -> str:
    """Bind extracted substrate/well nodes to the explicit OTA supply ports."""
    replacements: dict[str, str] = {}
    for line in _logical_lines(text):
        fields = line.split()
        if not fields or not fields[0].casefold().startswith("x") or len(fields) < 6:
            continue
        model = fields[5].casefold()
        replacement = {
            NMOS_MODEL: "VSS",
            PMOS_MODEL: "VDD",
        }.get(model)
        if replacement is None:
            continue
        key = fields[4].casefold()
        previous = replacements.get(key)
        if previous is not None and previous != replacement:
            raise ValueError(
                f"extracted bulk node '{fields[4]}' is shared by NMOS and PMOS devices"
            )
        replacements[key] = replacement
    if not replacements:
        raise ValueError("extracted OTA contains no recognized IHP MOS bulk nodes")

    output = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            output.append(raw)
            continue
        fields = raw.split()
        output.append(
            " ".join(replacements.get(field.casefold(), field) for field in fields)
        )
    return "\n".join(output) + "\n"


def validate_ota_pex(
    text: str,
    *,
    expected_subcircuit: str | None = None,
) -> PexTopology:
    lines = _logical_lines(text)
    subcircuits = [
        fields
        for line in lines
        if (fields := line.split())
        and fields[0].casefold() == ".subckt"
    ]
    if len(subcircuits) != 1 or len(subcircuits[0]) < 2:
        raise ValueError("OTA PEX must contain exactly one flattened subcircuit")
    header = subcircuits[0]
    subcircuit_name = header[1]
    ports = tuple(header[2 : 2 + len(OTA_PORTS)])
    if tuple(port.casefold() for port in ports) != tuple(
        port.casefold() for port in OTA_PORTS
    ):
        raise ValueError(f"OTA PEX ports must be {OTA_PORTS}, found {ports}")
    if len(header) != 2 + len(OTA_PORTS):
        raise ValueError(
            "OTA PEX subcircuit contains unexpected extra ports or parameters"
        )
    if (
        expected_subcircuit is not None
        and subcircuit_name.casefold() != expected_subcircuit.casefold()
    ):
        raise ValueError(
            f"expected OTA PEX subcircuit '{expected_subcircuit}', "
            f"found '{subcircuit_name}'"
        )

    resistors = []
    capacitors = []
    devices = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        kind = fields[0][0].casefold()
        if kind == "r" and len(fields) >= 4:
            resistors.append((fields[1], fields[2]))
        elif kind == "c" and len(fields) >= 4:
            capacitors.append((fields[1], fields[2]))
        elif kind == "x" and len(fields) >= 6:
            model = fields[5].casefold()
            if model in {NMOS_MODEL, PMOS_MODEL}:
                devices.append(
                    _Mos(
                        drain=fields[1],
                        gate=fields[2],
                        source=fields[3],
                        bulk=fields[4],
                        model=model,
                    )
                )
    nmos = [device for device in devices if device.model == NMOS_MODEL]
    pmos = [device for device in devices if device.model == PMOS_MODEL]
    if len(nmos) < 2 or len(pmos) < 2:
        raise ValueError(
            "OTA PEX must contain at least two NMOS and two PMOS devices"
        )

    connectivity = _Connectivity()
    for port in ports:
        connectivity.add(port)
    for device in devices:
        for node in (device.drain, device.gate, device.source, device.bulk):
            connectivity.add(node)
    for node_a, node_b in resistors:
        connectivity.union(node_a, node_b)

    equivalent = connectivity.equivalent

    def is_diff_output(device: _Mos) -> bool:
        return (
            equivalent(device.gate, "VINP")
            and equivalent(device.bulk, "VSS")
            and _channel_between(device, "VOUT", "IBIAS", equivalent)
        )

    def is_diff_reference(device: _Mos) -> bool:
        return (
            equivalent(device.gate, "VINN")
            and equivalent(device.bulk, "VSS")
            and _channel_touches(device, "IBIAS", equivalent)
        )

    diff_output_group = _device_group(
        nmos,
        is_diff_output,
        "VINP differential-pair transistor",
    )
    diff_reference_group = _device_group(
        nmos,
        is_diff_reference,
        "VINN differential-pair transistor",
    )
    diff_output = diff_output_group[0]
    diff_reference = diff_reference_group[0]
    internal_n1 = _other_channel_node(diff_reference, "IBIAS", equivalent)
    if equivalent(internal_n1, "IBIAS"):
        raise ValueError(
            "VINN differential-pair transistor has both channel terminals at IBIAS"
        )
    if not all(
        equivalent(
            _other_channel_node(device, "IBIAS", equivalent),
            internal_n1,
        )
        for device in diff_reference_group
    ):
        raise ValueError("VINN differential-pair fingers do not share one drain")
    if equivalent(internal_n1, "VOUT"):
        raise ValueError("both differential-pair drains are connected to VOUT")

    def is_mirror_output(device: _Mos) -> bool:
        return (
            equivalent(device.gate, internal_n1)
            and equivalent(device.bulk, "VDD")
            and _channel_between(device, "VOUT", "VDD", equivalent)
        )

    def is_mirror_reference(device: _Mos) -> bool:
        return (
            equivalent(device.gate, internal_n1)
            and equivalent(device.bulk, "VDD")
            and _channel_between(device, internal_n1, "VDD", equivalent)
        )

    _device_group(
        pmos,
        is_mirror_output,
        "output current-mirror transistor",
    )
    _device_group(
        pmos,
        is_mirror_reference,
        "diode-connected current-mirror transistor",
    )
    unmatched_nmos = [
        device
        for device in nmos
        if not is_diff_output(device)
        and not is_diff_reference(device)
        and not (
            equivalent(device.gate, "IBIAS")
            and equivalent(device.drain, "IBIAS")
            and equivalent(device.source, "IBIAS")
            and equivalent(device.bulk, "VSS")
        )
    ]
    unmatched_pmos = [
        device
        for device in pmos
        if not is_mirror_output(device)
        and not is_mirror_reference(device)
        and not (
            equivalent(device.gate, "VDD")
            and equivalent(device.drain, "VDD")
            and equivalent(device.source, "VDD")
            and equivalent(device.bulk, "VDD")
        )
    ]
    if unmatched_nmos or unmatched_pmos:
        device = (unmatched_nmos or unmatched_pmos)[0]
        raise ValueError(
            "OTA PEX has "
            f"{len(unmatched_nmos) + len(unmatched_pmos)} MOS finger(s) outside "
            f"the logical terminal buses; first unmatched channel is "
            f"{device.drain}/{device.source} with gate {device.gate}"
        )
    if equivalent(diff_output.gate, diff_reference.gate):
        raise ValueError("VINP and VINN are shorted in the OTA PEX")

    return PexTopology(
        subcircuit_name=subcircuit_name,
        transistor_count=len(devices),
        resistor_count=len(resistors),
        capacitor_count=len(capacitors),
    )


def validate_primitive_pex(
    text: str,
    primitive: str,
    *,
    expected_subcircuit: str | None = None,
) -> PexTopology:
    """Reject primitive PEX where any MOS finger misses its logical bus."""
    primitive_ports = {
        "simplediffpair": ("DP", "DN", "GP", "GN", "S", "B"),
        "currentmirror": ("DOUT", "DREF", "S", "B"),
    }
    expected_model = {
        "simplediffpair": NMOS_MODEL,
        "currentmirror": PMOS_MODEL,
    }
    try:
        ports = primitive_ports[primitive]
        model = expected_model[primitive]
    except KeyError as error:
        raise ValueError(f"unsupported primitive '{primitive}'") from error

    lines = _logical_lines(text)
    subcircuits = [
        fields
        for line in lines
        if (fields := line.split()) and fields[0].casefold() == ".subckt"
    ]
    if len(subcircuits) != 1 or len(subcircuits[0]) < 2:
        raise ValueError(
            f"{primitive} PEX must contain exactly one flattened subcircuit"
        )
    header = subcircuits[0]
    subcircuit_name = header[1]
    found_ports = tuple(header[2 : 2 + len(ports)])
    if tuple(port.casefold() for port in found_ports) != tuple(
        port.casefold() for port in ports
    ):
        raise ValueError(f"{primitive} PEX ports must be {ports}, found {found_ports}")
    if len(header) != 2 + len(ports):
        raise ValueError(
            f"{primitive} PEX subcircuit contains unexpected extra ports or parameters"
        )
    if (
        expected_subcircuit is not None
        and subcircuit_name.casefold() != expected_subcircuit.casefold()
    ):
        raise ValueError(
            f"expected {primitive} PEX subcircuit '{expected_subcircuit}', "
            f"found '{subcircuit_name}'"
        )

    resistors: list[tuple[str, str]] = []
    capacitors: list[tuple[str, str]] = []
    devices: list[_Mos] = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        kind = fields[0][0].casefold()
        if kind == "r" and len(fields) >= 4:
            resistors.append((fields[1], fields[2]))
        elif kind == "c" and len(fields) >= 4:
            capacitors.append((fields[1], fields[2]))
        elif kind == "x" and len(fields) >= 6:
            device_model = fields[5].casefold()
            if device_model in {NMOS_MODEL, PMOS_MODEL}:
                devices.append(
                    _Mos(
                        drain=fields[1],
                        gate=fields[2],
                        source=fields[3],
                        bulk=fields[4],
                        model=device_model,
                    )
                )
    if not devices:
        raise ValueError(f"{primitive} PEX contains no recognized IHP MOS devices")
    if any(device.model != model for device in devices):
        raise ValueError(f"{primitive} PEX contains an unexpected MOS polarity")

    connectivity = _Connectivity()
    for port in ports:
        connectivity.add(port)
    for device in devices:
        for node in (device.drain, device.gate, device.source, device.bulk):
            connectivity.add(node)
    for node_a, node_b in resistors:
        connectivity.union(node_a, node_b)
    equivalent = connectivity.equivalent

    if primitive == "simplediffpair":
        predicates = (
            (
                "GP differential-pair transistor",
                lambda device: equivalent(device.gate, "GP")
                and _channel_between(device, "DP", "S", equivalent),
            ),
            (
                "GN differential-pair transistor",
                lambda device: equivalent(device.gate, "GN")
                and _channel_between(device, "DN", "S", equivalent),
            ),
        )
    else:
        predicates = (
            (
                "output current-mirror transistor",
                lambda device: equivalent(device.gate, "DREF")
                and _channel_between(device, "DOUT", "S", equivalent),
            ),
            (
                "diode-connected current-mirror transistor",
                lambda device: equivalent(device.gate, "DREF")
                and _channel_between(device, "DREF", "S", equivalent),
            ),
        )
    for description, predicate in predicates:
        _device_group(devices, predicate, description)

    def is_dummy(device: _Mos) -> bool:
        return (
            equivalent(device.gate, "S")
            and equivalent(device.drain, "S")
            and equivalent(device.source, "S")
        )

    unmatched = [
        device
        for device in devices
        if not is_dummy(device)
        and not any(predicate(device) for _, predicate in predicates)
    ]
    if unmatched:
        device = unmatched[0]
        raise ValueError(
            f"{primitive} PEX has {len(unmatched)} MOS finger(s) outside the "
            f"logical terminal buses; first unmatched channel is "
            f"{device.drain}/{device.source} with gate {device.gate}"
        )

    return PexTopology(
        subcircuit_name=subcircuit_name,
        transistor_count=len(devices),
        resistor_count=len(resistors),
        capacitor_count=len(capacitors),
    )


def _device_group(devices, predicate, description: str) -> list[_Mos]:
    matches = [device for device in devices if predicate(device)]
    if not matches:
        raise ValueError(f"expected at least one {description}, found none")
    return matches


def _channel_touches(device: _Mos, node: str, equivalent) -> bool:
    return equivalent(device.drain, node) or equivalent(device.source, node)


def _channel_between(device: _Mos, node_a: str, node_b: str, equivalent) -> bool:
    return (
        equivalent(device.drain, node_a)
        and equivalent(device.source, node_b)
    ) or (
        equivalent(device.drain, node_b)
        and equivalent(device.source, node_a)
    )


def _other_channel_node(device: _Mos, node: str, equivalent) -> str:
    drain_matches = equivalent(device.drain, node)
    source_matches = equivalent(device.source, node)
    if not drain_matches and not source_matches:
        raise ValueError(
            f"MOS channel {device.drain}/{device.source} does not touch {node}"
        )
    if drain_matches:
        return device.source
    return device.drain


def _logical_lines(text: str) -> list[str]:
    logical = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and logical:
            logical[-1] += " " + line[1:].strip()
        else:
            logical.append(line)
    return logical


class _Connectivity:
    def __init__(self) -> None:
        self._parents: dict[str, str] = {}

    def add(self, node: str) -> str:
        key = node.casefold()
        self._parents.setdefault(key, key)
        return key

    def find(self, node: str) -> str:
        key = self.add(node)
        parent = self._parents[key]
        if parent != key:
            self._parents[key] = self.find(parent)
        return self._parents[key]

    def union(self, node_a: str, node_b: str) -> None:
        root_a = self.find(node_a)
        root_b = self.find(node_b)
        if root_a != root_b:
            self._parents[root_b] = root_a

    def equivalent(self, node_a: str, node_b: str) -> bool:
        return self.find(node_a) == self.find(node_b)
