# OTA pre-layout examples

The SKY130A and GF180MCU D examples run the same typed four-transistor OTA
exploration with PDK-specific electrical biases and transistor-level NGSpice
verification. They do not use a physical LUT or layout-aware analysis.

Select the locally installed PDK and pass the corresponding full electrical
LUTs:

```text
export PDK_ROOT=/path/to/pdks
export PDK=sky130A
cargo run --release -p shapeic-core --example ota_4t_sky130_pre_layout -- \
  shapeic-lut/lut_generation/generated/sky130A_1v8_nmos_5d.npz \
  shapeic-lut/lut_generation/generated/sky130A_1v8_pmos_5d.npz
```

```text
export PDK_ROOT=/path/to/pdks
export PDK=gf180mcuD
cargo run --release -p shapeic-core --example ota_4t_gf180_pre_layout -- \
  shapeic-lut/lut_generation/generated/gf180mcuD_3v3_nmos_5d.npz \
  shapeic-lut/lut_generation/generated/gf180mcuD_3v3_pmos_5d.npz
```

Each run writes the complete exploration CSV, the rendered transistor-level
netlist, NGSpice log, and comparison summary under a timestamped directory in
`target/`.
