# Hierarchical four-transistor OTA

This example wraps `ota_4t` in an `ota_system` macro and runs the complete
one-pass hierarchical flow:

1. evaluate the parent with the child's nominal compact seed;
2. derive the OTA gain requirement and exact `vout`/`vbias` design values;
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
