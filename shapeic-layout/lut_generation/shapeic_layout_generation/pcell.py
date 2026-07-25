from __future__ import annotations

import math
from pathlib import Path

IHP_GDSFACTORY_VERSION = "2.0.0"
IHP_TAP_SIZE_UM = 0.78
IHP_ROUTE_WIDTH_UM = 0.3
IHP_GATE_POLY_OVERLAP_UM = 0.02
IHP_BUS_CLEARANCE_UM = 0.2
EXTERNAL_PORTS = {
    "simplediffpair": ("DP", "DN", "GP", "GN", "S", "B"),
    "currentmirror": ("DOUT", "DREF", "S", "B"),
    "ota_4t": ("VOUT", "VINP", "VINN", "IBIAS", "VDD", "VSS"),
}


def write_primitive_gds(
    primitive: str,
    length_m: float,
    finger_width_m: float,
    nf: int,
    output: Path,
) -> str:
    """Build the fixed primitive PCell and write it as GDS.

    Geometry arguments exposed by Shapeic are L, Wf and nf. ihp-gdsfactory's
    MOS core receives total width, therefore the backend passes Wf*nf.
    """
    try:
        import gdsfactory as gf
        import ihp
        from ihp import PDK, cells
        from ihp.cells.fet_transistors import TECH, _mos_core
    except ImportError as error:
        raise RuntimeError(
            "the magic backend requires gdsfactory and ihp-gdsfactory"
        ) from error
    if ihp.__version__ != IHP_GDSFACTORY_VERSION:
        raise RuntimeError(
            "the magic backend requires "
            f"ihp-gdsfactory=={IHP_GDSFACTORY_VERSION}, found {ihp.__version__}"
        )
    PDK.activate()
    length_um = length_m * 1.0e6
    finger_width_um = finger_width_m * 1.0e6
    if primitive == "simplediffpair":
        component = _simple_diff_pair(
            gf, cells, _mos_core, TECH, length_um, finger_width_um, nf
        )
    elif primitive == "currentmirror":
        component = _current_mirror(
            gf, cells, _mos_core, TECH, length_um, finger_width_um, nf
        )
    else:
        raise ValueError(f"unsupported primitive '{primitive}'")
    _validate_external_port_isolation(component, gf.kdb, EXTERNAL_PORTS[primitive])
    output.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(output)
    return component.name


def write_ota_gds(
    diff_length_m: float,
    diff_finger_width_m: float,
    diff_nf: int,
    mirror_length_m: float,
    mirror_finger_width_m: float,
    mirror_nf: int,
    output: Path,
) -> str:
    """Build and write the routed four-transistor OTA verification layout."""
    try:
        import gdsfactory as gf
        import ihp
        from ihp import PDK, cells
        from ihp.cells.fet_transistors import TECH, _mos_core
    except ImportError as error:
        raise RuntimeError(
            "the OTA PEX backend requires gdsfactory and ihp-gdsfactory"
        ) from error
    if ihp.__version__ != IHP_GDSFACTORY_VERSION:
        raise RuntimeError(
            "the OTA PEX backend requires "
            f"ihp-gdsfactory=={IHP_GDSFACTORY_VERSION}, found {ihp.__version__}"
        )
    PDK.activate()
    component = _ota_4t(
        gf,
        cells,
        _mos_core,
        TECH,
        diff_length_m * 1.0e6,
        diff_finger_width_m * 1.0e6,
        diff_nf,
        mirror_length_m * 1.0e6,
        mirror_finger_width_m * 1.0e6,
        mirror_nf,
    )
    _validate_external_port_isolation(component, gf.kdb, EXTERNAL_PORTS["ota_4t"])
    output.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(output)
    return component.name


def _simple_diff_pair(gf, cells, mos_core, tech, length: float, wf: float, nf: int):
    name = _cell_name("simplediffpair", length, wf, nf)
    component = gf.Component(name)
    device = _bussed_mos_device(gf, mos_core, tech, "nmos", length, wf, nf)
    pitch = float(device.dbbox().right) - float(device.dbbox().left) + 1.2
    left = component.add_ref(device)
    right = component.add_ref(device)
    left.move((-pitch / 2, 0))
    right.move((pitch / 2, 0))
    dummy_left = component.add_ref(device)
    dummy_right = component.add_ref(device)
    dummy_left.move((-3.0 * pitch / 2, 0))
    dummy_right.move((3.0 * pitch / 2, 0))

    devices = (left, right, dummy_left, dummy_right)
    source_y = min(float(ref.dbbox().bottom) for ref in devices) - 1.0
    source = (0.0, source_y)
    for ref in devices:
        _wire(component, _point(ref.ports["S"]), source)
    for ref in (dummy_left, dummy_right):
        _wire(component, _point(ref.ports["D"]), source)
        _wire(component, _point(ref.ports["G"]), source)

    bulk_ref = component.add_ref(
        cells.ptap1(width=IHP_TAP_SIZE_UM, length=IHP_TAP_SIZE_UM)
    )
    bulk_ref.move((0.0, source_y - 1.5))
    bulk = _point(bulk_ref.ports["TAP"])
    gate_left = _point(left.ports["G"])
    gate_right = _point(right.ports["G"])
    external = {
        "DP": _point(left.ports["D"]),
        "DN": _point(right.ports["D"]),
        "GP": gate_left,
        "GN": gate_right,
        "S": source,
        "B": bulk,
    }
    _add_external_ports(component, external)
    return component


def _current_mirror(gf, cells, mos_core, tech, length: float, wf: float, nf: int):
    name = _cell_name("currentmirror", length, wf, nf)
    component = gf.Component(name)
    device = _bussed_mos_device(gf, mos_core, tech, "pmos", length, wf, nf)
    pitch = float(device.dbbox().right) - float(device.dbbox().left) + 1.2
    output = component.add_ref(device)
    reference = component.add_ref(device)
    output.move((-pitch / 2, 0))
    reference.move((pitch / 2, 0))
    dummy_left = component.add_ref(device)
    dummy_right = component.add_ref(device)
    dummy_left.move((-3.0 * pitch / 2, 0))
    dummy_right.move((3.0 * pitch / 2, 0))

    devices = (output, reference, dummy_left, dummy_right)
    source_y = max(float(ref.dbbox().top) for ref in devices) + 1.0
    source = (0.0, source_y)
    for ref in devices:
        _wire(component, _point(ref.ports["S"]), source)
    for ref in (dummy_left, dummy_right):
        _wire(component, _point(ref.ports["D"]), source)
        _wire(component, _point(ref.ports["G"]), source)

    output_gate = _point(output.ports["G"])
    reference_gate = _point(reference.ports["G"])
    reference_drain = _point(reference.ports["D"])
    gate_bus = (
        0.0,
        min(float(ref.dbbox().bottom) for ref in devices) - 1.0,
    )
    for terminal in (output_gate, reference_gate, reference_drain):
        _wire(component, terminal, gate_bus)
    bulk_ref = component.add_ref(
        cells.ntap1(width=IHP_TAP_SIZE_UM, length=IHP_TAP_SIZE_UM)
    )
    bulk_ref.move((0.0, source_y + 1.5))
    bulk = _point(bulk_ref.ports["TAP"])
    component.add_polygon(
        [
            (-2.0 * pitch, -wf / 2 - 1.0),
            (2.0 * pitch, -wf / 2 - 1.0),
            (2.0 * pitch, bulk[1] + 1.0),
            (-2.0 * pitch, bulk[1] + 1.0),
        ],
        layer="NWelldrawing",
    )
    external = {
        "DOUT": _point(output.ports["D"]),
        "DREF": reference_drain,
        "S": source,
        "B": bulk,
    }
    _add_external_ports(component, external)
    return component


def _ota_4t(
    gf,
    cells,
    mos_core,
    tech,
    diff_length: float,
    diff_wf: float,
    diff_nf: int,
    mirror_length: float,
    mirror_wf: float,
    mirror_nf: int,
):
    name = (
        f"ota_4t_ldp{diff_length:.3f}_wdp{diff_wf:.3f}_ndp{diff_nf}"
        f"_lcm{mirror_length:.3f}_wcm{mirror_wf:.3f}_ncm{mirror_nf}"
    ).replace(".", "p")
    component = gf.Component(name)
    diff = component.add_ref(
        _simple_diff_pair(
            gf, cells, mos_core, tech, diff_length, diff_wf, diff_nf
        )
    )
    mirror = component.add_ref(
        _current_mirror(
            gf, cells, mos_core, tech, mirror_length, mirror_wf, mirror_nf
        )
    )
    vertical_clearance = 4.0
    mirror.move(
        (
            0.0,
            float(diff.dbbox().top)
            - float(mirror.dbbox().bottom)
            + vertical_clearance,
        )
    )

    diff_dp = _point(diff.ports["DP"])
    diff_dn = _point(diff.ports["DN"])
    mirror_dout = _point(mirror.ports["DOUT"])
    mirror_dref = _point(mirror.ports["DREF"])
    left_x = min(
        float(diff.dbbox().left),
        float(mirror.dbbox().left),
        diff_dp[0],
        mirror_dout[0],
    ) - 1.0
    right_x = max(
        float(diff.dbbox().right),
        float(mirror.dbbox().right),
        diff_dn[0],
        mirror_dref[0],
    ) + 1.0
    vout = (left_x, (diff_dp[1] + mirror_dout[1]) / 2.0)
    internal_n1 = (right_x, (diff_dn[1] + mirror_dref[1]) / 2.0)
    for terminal in (diff_dp, diff_dn, mirror_dout, mirror_dref, vout):
        _add_metal1_metal2_via(component, tech, terminal)
    for terminal in (diff_dp, mirror_dout):
        _wire(component, terminal, vout, layer="Metal2drawing")
    for terminal in (diff_dn, mirror_dref):
        _wire(component, terminal, internal_n1, layer="Metal2drawing")

    mirror_source = _point(mirror.ports["S"])
    mirror_bulk = _point(mirror.ports["B"])
    _wire(component, mirror_source, mirror_bulk)
    external = {
        "VOUT": vout,
        "VINP": _point(diff.ports["GP"]),
        "VINN": _point(diff.ports["GN"]),
        "IBIAS": _point(diff.ports["S"]),
        "VDD": mirror_bulk,
        "VSS": _point(diff.ports["B"]),
    }
    _add_external_ports(component, external)
    return component


def _ihp_mos_device(mos_core, tech, kind: str, length: float, wf: float, nf: int):
    if kind not in {"nmos", "pmos"}:
        raise ValueError(f"unsupported MOS kind '{kind}'")
    if not math.isfinite(length) or not math.isfinite(wf):
        raise ValueError("MOS length and finger width must be finite")

    minimum_length = getattr(tech, f"{kind}_min_length")
    maximum_length = getattr(tech, f"{kind}_max_length")
    minimum_width = getattr(tech, f"{kind}_min_width")
    maximum_width = getattr(tech, f"{kind}_max_width")
    maximum_nf = getattr(tech, f"{kind}_max_nf")
    if not minimum_length <= length <= maximum_length:
        raise ValueError(
            f"{kind} length={length} out of range "
            f"[{minimum_length}, {maximum_length}]"
        )
    if not minimum_width <= wf <= maximum_width:
        raise ValueError(
            f"{kind} finger width={wf} out of range "
            f"[{minimum_width}, {maximum_width}]"
        )
    if not isinstance(nf, int) or isinstance(nf, bool) or not 1 <= nf <= maximum_nf:
        raise ValueError(f"{kind} nf={nf} out of range [1, {maximum_nf}]")

    # IHP 2.0's public wrapper applies its 10 um per-finger limit to total
    # width. The core itself correctly divides total width by nf.
    return mos_core(
        width=wf * nf,
        length=length,
        nf=nf,
        is_pmos=kind == "pmos",
        is_hv=False,
    )


def _bussed_mos_device(
    gf,
    mos_core,
    tech,
    kind: str,
    length: float,
    wf: float,
    nf: int,
):
    raw = _ihp_mos_device(mos_core, tech, kind, length, wf, nf)
    name = _cell_name(f"{kind}_bussed", length, wf, nf)
    component = gf.Component(name)
    component.add_ref(raw)

    metal_columns = _layer_boxes(raw, "Metal1drawing")
    gate_fingers = _layer_boxes(raw, "GatPolydrawing")
    if len(metal_columns) != nf + 1:
        raise ValueError(
            f"{kind} nf={nf} exposes {len(metal_columns)} Metal1 columns; "
            f"expected {nf + 1}"
        )
    if len(gate_fingers) != nf:
        raise ValueError(
            f"{kind} nf={nf} exposes {len(gate_fingers)} gate fingers; expected {nf}"
        )

    source_columns = metal_columns[0::2]
    drain_columns = metal_columns[1::2]
    contact_cut_size = float(tech.cont_size)
    contact_pad_size = contact_cut_size + 2.0 * float(tech.gat_d)
    gate_bottom = min(box[1] for box in gate_fingers)
    gate_top = max(box[3] for box in gate_fingers)
    gate_bus_y = (
        gate_bottom
        + IHP_GATE_POLY_OVERLAP_UM
        - contact_pad_size / 2.0
    )
    raw_left = float(raw.dbbox().left)
    contact_x = min(
        raw_left
        - float(tech.cont_gate_dist)
        - contact_pad_size / 2.0,
        source_columns[0][0]
        - contact_pad_size / 2.0
        - IHP_BUS_CLEARANCE_UM,
    )
    _rectangle(
        component,
        "GatPolydrawing",
        contact_x - contact_pad_size / 2.0,
        gate_bus_y - contact_pad_size / 2.0,
        max(box[2] for box in gate_fingers),
        gate_bus_y + contact_pad_size / 2.0,
    )
    # Magic classifies contacts while importing each GDS cell, before flattening.
    _rectangle(
        component,
        "Metal1drawing",
        contact_x - contact_pad_size / 2.0,
        gate_bus_y - contact_pad_size / 2.0,
        contact_x + contact_pad_size / 2.0,
        gate_bus_y + contact_pad_size / 2.0,
    )
    _rectangle(
        component,
        "Contdrawing",
        contact_x - contact_cut_size / 2.0,
        gate_bus_y - contact_cut_size / 2.0,
        contact_x + contact_cut_size / 2.0,
        gate_bus_y + contact_cut_size / 2.0,
    )

    source_bus_y = (
        gate_bus_y
        - contact_pad_size / 2.0
        - IHP_ROUTE_WIDTH_UM / 2.0
        - IHP_BUS_CLEARANCE_UM
    )
    drain_bus_y = (
        gate_top + IHP_ROUTE_WIDTH_UM / 2.0 + IHP_BUS_CLEARANCE_UM
    )
    if kind == "nmos":
        source = _add_terminal_bus(
            component,
            source_columns,
            source_bus_y,
            connect_from_top=False,
        )
        drain = _add_terminal_bus(
            component,
            drain_columns,
            drain_bus_y,
            connect_from_top=True,
        )
    else:
        source = _add_terminal_bus(
            component,
            source_columns,
            drain_bus_y,
            connect_from_top=True,
        )
        drain = _add_terminal_bus(
            component,
            drain_columns,
            source_bus_y,
            connect_from_top=False,
        )
    component.add_port(
        name="S",
        center=source,
        width=IHP_ROUTE_WIDTH_UM,
        orientation=270,
        layer="Metal1pin",
        port_type="electrical",
    )
    component.add_port(
        name="D",
        center=drain,
        width=IHP_ROUTE_WIDTH_UM,
        orientation=90,
        layer="Metal1pin",
        port_type="electrical",
    )
    component.add_port(
        name="G",
        center=(contact_x, gate_bus_y),
        width=contact_pad_size,
        orientation=180,
        layer="Metal1pin",
        port_type="electrical",
    )
    return component


def _add_terminal_bus(
    component,
    columns: list[tuple[float, float, float, float]],
    bus_y: float,
    *,
    connect_from_top: bool,
) -> tuple[float, float]:
    centers = [((box[0] + box[2]) / 2.0) for box in columns]
    for left, bottom, right, top in columns:
        edge = top if connect_from_top else bottom
        _rectangle(
            component,
            "Metal1drawing",
            left,
            min(edge, bus_y - IHP_ROUTE_WIDTH_UM / 2.0),
            right,
            max(edge, bus_y + IHP_ROUTE_WIDTH_UM / 2.0),
        )
    _rectangle(
        component,
        "Metal1drawing",
        min(centers) - IHP_ROUTE_WIDTH_UM / 2.0,
        bus_y - IHP_ROUTE_WIDTH_UM / 2.0,
        max(centers) + IHP_ROUTE_WIDTH_UM / 2.0,
        bus_y + IHP_ROUTE_WIDTH_UM / 2.0,
    )
    return (min(centers) + max(centers)) / 2.0, bus_y


def _layer_boxes(component, layer: str) -> list[tuple[float, float, float, float]]:
    dbu = float(component.kcl.dbu)
    polygons = component.get_polygons(merge=False, by="name").get(layer, [])
    boxes = []
    for polygon in polygons:
        box = polygon.bbox()
        boxes.append(
            (
                float(box.left) * dbu,
                float(box.bottom) * dbu,
                float(box.right) * dbu,
                float(box.top) * dbu,
            )
        )
    return sorted(boxes, key=lambda box: (box[0], box[1], box[2], box[3]))


def _rectangle(
    component,
    layer: str,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> None:
    if right <= left or top <= bottom:
        raise ValueError(
            f"invalid {layer} rectangle ({left}, {bottom})..({right}, {top})"
        )
    component.add_polygon(
        [(left, bottom), (right, bottom), (right, top), (left, top)],
        layer=layer,
    )


def _add_metal1_metal2_via(
    component,
    tech,
    center: tuple[float, float],
) -> None:
    """Draw a same-cell Via1 stack so Magic recognizes it during GDS import."""
    x, y = center
    via_size = float(tech.via1_size)
    pad_size = via_size + 2.0 * float(tech.via1_enc_metal)
    for layer, size in (
        ("Metal1drawing", pad_size),
        ("Metal2drawing", pad_size),
        ("Via1drawing", via_size),
    ):
        _rectangle(
            component,
            layer,
            x - size / 2.0,
            y - size / 2.0,
            x + size / 2.0,
            y + size / 2.0,
        )


def _wire(
    component,
    start: tuple[float, float],
    stop: tuple[float, float],
    *,
    layer: str = "Metal1drawing",
) -> None:
    width = IHP_ROUTE_WIDTH_UM
    x1, y1 = start
    x2, y2 = stop
    if abs(y2 - y1) > 1.0e-12:
        _rectangle(
            component,
            layer,
            x1 - width / 2.0,
            min(y1, y2) - width / 2.0,
            x1 + width / 2.0,
            max(y1, y2) + width / 2.0,
        )
    if abs(x2 - x1) > 1.0e-12:
        _rectangle(
            component,
            layer,
            min(x1, x2) - width / 2.0,
            y2 - width / 2.0,
            max(x1, x2) + width / 2.0,
            y2 + width / 2.0,
        )


def _add_external_ports(component, ports: dict[str, tuple[float, float]]) -> None:
    for name, center in ports.items():
        component.add_port(
            name=name,
            center=center,
            width=0.3,
            orientation=0,
            layer="Metal1pin",
            port_type="electrical",
        )
        component.add_label(text=name, position=center, layer="Metal1pin")


def _validate_external_port_isolation(
    component, kdb, port_names: tuple[str, ...]
) -> None:
    polygons = component.get_polygons(merge=True, by="name").get(
        "Metal1drawing", []
    )
    if not polygons:
        raise ValueError("PCell has no Metal1 geometry")

    owners: dict[int, str] = {}
    for name in port_names:
        center = _point(component.ports[name])
        point = kdb.Point(
            round(center[0] / component.kcl.dbu),
            round(center[1] / component.kcl.dbu),
        )
        matches = [
            index
            for index, polygon in enumerate(polygons)
            if polygon.inside(point)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"external port {name} touches {len(matches)} Metal1 components"
            )
        component_index = matches[0]
        if component_index in owners:
            raise ValueError(
                f"external ports {owners[component_index]} and {name} are shorted "
                "on Metal1"
            )
        owners[component_index] = name


def _point(port) -> tuple[float, float]:
    center = port.center
    return float(center[0]), float(center[1])


def _cell_name(primitive: str, length: float, wf: float, nf: int) -> str:
    return f"{primitive}_l{length:.3f}_wf{wf:.3f}_nf{nf}".replace(".", "p")
