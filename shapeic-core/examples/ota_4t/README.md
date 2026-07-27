# Four-transistor layout-aware OTA experiment

The example always runs the electrical sizing, DC gain, and an AC analysis with
the intrinsic and compact-model extrinsic capacitances of all four MOS devices:

```sh
cargo run --release -p shapeic-core --example ota_4t -- NMOS.npz PMOS.npz
```

Pass the separate physical primitive LUT as the third path to run bounded
queries over `(L, Wf, nf)` and add the extracted physical `G+sC` matrices to
the complete compact-model electrical MNA:

```sh
cargo run --release -p shapeic-core --example ota_4t -- \
  NMOS.npz PMOS.npz PHYSICAL.npz
```

Both AC analyses search from 1 Hz to 100 GHz using four coarse points per
decade and refine each downward crossing to 0.5% relative frequency tolerance.
In prune mode, a candidate stops after its first failed AC target. The
electrical baseline prepares the frequency-independent MNA once, instantiates a
numerical `G` matrix for each candidate, stamps the intrinsic MOS matrix plus
`cgsol`, `cgdol`, `cjs`, and `cjd` into a numerical `C` matrix, and solves
`(G+jwC)x=z` by complex LU at every requested frequency. The layout-aware path
starts from the base MNA, stamps the physical `G+sC` matrices followed by the
same complete compact-model capacitances, and obtains a symbolic transfer
function with the direct MNA solver before evaluating the adaptive frequencies.
The terminal prints aligned electrical and layout-aware tables with DC gain,
-3 dB bandwidth, unity-gain frequency, phase margin, and the number of evaluated
frequencies. Because both paths share the same compact-model base, the
difference table isolates the added physical matrix.

The separate electrical and physical verification examples retain their dense
20-points-per-decade sweeps. The adaptive electrical verification runs one
point through dense Shapeic, adaptive Shapeic, and NGSpice:

```sh
cargo run --release -p shapeic-core \
  --example ota_4t_electrical_adaptive_verification -- NMOS.npz PMOS.npz
```

The NMOS source is connected to `IBIAS`, while its bulk and the PCell substrate
tap are connected to `VSS`. Therefore each differential-pair LUT query uses
`VBS = VSS - VS = -VS`; the PMOS source and bulk remain tied to `VDD`.

The electrical numeric path stores capacitances directly in farads and forms
`G+j*2*pi*f*C` without symbolic substitution. The layout-aware path preserves
the original symbolic representation used by this example: it stamps
capacitances in farads and substitutes `s = j*2*pi*f`.

Candidates outside any physical axis, including `nf > 20`, remain visible and
are marked as excluded from the physical and difference tables. Their
electrical AC result remains available.

Charge-conserving physical capacitance matrices can leave a removable common
factor of `s` in the symbolic transfer function. If direct substitution at
`s=0` produces `0/0`, the reported DC gain uses the already evaluated 1 Hz
response as its low-frequency limit.

The example requires the nine independent intrinsic coefficients
`cgg,cgd,cgs`, `cdg,cdd,cds`, and `csg,csd,css`. Shapeic reconstructs the
seven bulk-related entries of the complete matrix from charge conservation. It
also requires the `nf=1..4` anchors for `cgsol`, `cgdol`, `cjs`, and `cjd`.
Both reduced LUTs and older files containing all sixteen intrinsic coefficients
can run the AC example.

The NPZ stores the raw ngspice convention. Before MNA stamping, Shapeic keeps
the diagonal terms and negates every mutual term, producing a charge-conserving
nodal matrix. The intrinsic matrix is then scaled by `nf`; the four extrinsic
capacitances already contain their evaluated total for the selected `nf` and
are not scaled again.
