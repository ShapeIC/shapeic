# Four-transistor physical verification

This example compares one Shapeic layout-aware operating point against
ngspice using a transistor-level PEX of the complete routed OTA:

```sh
export IHP_PDK_ROOT=/path/to/IHP-Open-PDK/ihp-sg13g2
export SHAPEIC_LAYOUT_PYTHON=shapeic-layout/lut_generation/.venv-ubuntu/bin/python

cargo run --release -p shapeic-core \
  --example ota_4t_physical_verification -- \
  shapeic-lut/lut_generation/generated/ihp_sg13g2_lv_nmos_5d.npz \
  shapeic-lut/lut_generation/generated/ihp_sg13g2_lv_pmos_5d.npz \
  shapeic-layout/lut_generation/generated/ihp_sg13g2_ota_physical.npz \
  shapeic-layout/lut_generation/configs/ihp_sg13g2_ota.toml
```

The Python helper receives the `L`, `Wf`, and `nf` selected by the Rust sizing
step, writes the full OTA GDS, runs Magic, normalizes the extracted bulk nodes,
and validates connectivity before ngspice is launched. NMOS source is
`IBIAS`, NMOS bulk is `VSS`, and PMOS source/bulk are `VDD`. Consequently the
differential-pair electrical query uses `VBS=-VS`.

The timestamped target directory retains the GDS even when Magic extraction or
topology validation fails. `pex/artifacts.json` records its absolute path as
soon as it is written. Successful runs also retain the raw and normalized PEX,
Magic TCL/logs, rendered ngspice netlist, Shapeic and ngspice AC sweeps, and
the scalar comparison CSV. The printed plotting command overlays gain and
phase from both sweeps.

The physical LUT must use layout policy
`symmetric-adjacent-with-edge-dummies-v3` and must be regenerated after the
multifinger terminal-bus correction. Its
matrices model local primitive parasitics; the reference PEX additionally
contains routing between primitives and the edge dummies, so a nonzero
difference is expected.
