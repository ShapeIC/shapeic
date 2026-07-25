# shapeic-layout

`shapeic-layout` reads physical primitive LUTs and performs bounded trilinear
interpolation over transistor length, per-finger width, and integer finger
count. A query returns full nodal `G` and `C` matrices with a stable port order.

The LUT is separate from the electrical MOS LUT so electrical exploration can
run first. Surviving candidates can then query local primitive parasitics and
stamp `G + sC` into the circuit MNA for a layout-aware reevaluation.

See `lut_generation/README.md` for archive generation.

The first schema exposes one physical port per primitive net. It preserves
coupling capacitance and inter-net leakage conductance, but it cannot preserve
series resistance along a net: Kron reduction removes a resistor whose far end
has no distinct port. A later resistive-aware schema must expose separate
boundary and intrinsic-device ports (or retain the extracted RC graph). For
ordinary dielectric extraction the physical `G` matrix is therefore expected
to be zero and layout-aware DC gain matches the electrical result; the AC
response changes through the physical `C` matrix.
