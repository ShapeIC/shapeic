# Hierarchical four-transistor OTA

This example wraps `ota_4t` in an `ota_system` macro and runs the complete
one-pass hierarchical flow:

1. evaluate the parent across aligned representative compact seeds and the
   parent-owned `vout`/`vbias` grids;
2. derive the OTA gain requirement and automatically propagate only the
   surviving node-voltage values into the child's public inputs;
3. explore the OTA implementation;
4. project its accepted compact candidates into the parent;
5. evaluate the definitive parent and recover recursive sizing provenance.

The hierarchy flow is electrical-only. It intentionally does not accept a
physical LUT.

```sh
cargo run --release -p shapeic-core --example ota_4t_hierarchical -- \
  path/to/nmos-5d.npz path/to/pmos-5d.npz \
  --workers 8 --batch-size 64
```

Both LUTs must belong to the same supported PDK: IHP SG13G2, SKY130A, or
GF180MCU D. The example prints the parent-to-child derivation audit, execution
tree, parent and child AC metrics, and the selected primitive geometry and
operating-point columns.

Compact seeds contain only approximate correlated small-signal parameters such
as `gm` and `ro`. Interface voltages are not seed data: the parent owns their
exploration grids through `MacroPublicInputAlias`, while the definitive parent
evaluation receives the actual interface voltages projected by the explored
child.
