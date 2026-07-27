# Shapeic physical LUT generation

This directory generates the separate physical LUT consumed by the
`shapeic-layout` crate. Each primitive is sampled over `(L, Wf, nf)` and stores
a reduced port conductance matrix and capacitance matrix. Values in TOML are in
micrometers; the NPZ axes use meters, siemens, and farads.

The fixed first experiment contains:

- `simplediffpair`: `DP, DN, GP, GN, S, B`
- `currentmirror`: `DOUT, DREF, S, B`
- symmetric adjacent devices with one edge dummy on each side
- integer query values for `nf`, with interpolation between sampled counts
- no extrapolation beyond the archive axes

Generate the deterministic smoke LUT without external EDA tools:

```sh
python3 generate.py configs/ihp_sg13g2_ota_smoke.toml --force
```

For real extraction, install the runtime dependencies and the pinned IHP PDK:

```sh
python3 -m pip install -r requirements-ihp.txt
python3 -m pip install --no-deps ihp-gdsfactory==2.0.0
```

The separate `--no-deps` step avoids the unresolved `gdsfactoryplus` /
`vlsirtools` dependency while retaining every package used by this generator.
Install Magic, set `IHP_PDK_ROOT`, and use
`configs/ihp_sg13g2_ota.toml`. The expected rcfile is
`${IHP_PDK_ROOT}/libs.tech/magic/ihp-sg13g2.magicrc`.

The PCell backend validates `Wf` as the per-finger width, then passes
`width=Wf*nf` to the IHP MOS geometry core because its `width` argument is total
device width. It uses fixed minimum-size `0.78 um` substrate and well taps.
Every source/drain diffusion column and every gate finger is connected to an
explicit logical bus outside the device core. Their positions remain valid as
`Wf` and `nf` change. GatPoly-to-Metal1 contacts are drawn directly in the bus
PCell so Magic sees poly, contact, and Metal1 together while importing GDS.
Before writing each GDS, the
backend also verifies that every logical port touches exactly one distinct
Metal1 component; accidental shorts or unconnected labels stop generation
before Magic runs.
Magic extracts a full RC graph, which is Kron-reduced to the fixed primitive
ports while retaining the first-order coefficient around `s=0`.
The Magic script flattens the generated hierarchy while preserving only the
top-level logical port labels. Signed capacitance correction terms emitted by
Magic are combined before the reduced nodal matrix is validated.
Networks containing only capacitors are reduced directly in `C`; mixed RC
networks retain the first-order reduction around `s=0`.

Before the complete 880-extraction run (440 points for each primitive),
validate the real backend at `nf=1` and `nf=50` for each primitive:

```sh
python3 generate.py configs/ihp_sg13g2_ota_magic_smoke.toml --force
```

Then generate the complete archive:

```sh
python3 generate.py configs/ihp_sg13g2_ota.toml --force
```

The resulting matrices contain only local primitive interconnect parasitics.
Inter-primitive routing and signoff extraction are intentionally outside this
first layout-aware stage.

`generate_ota_pex.py` is the separate verification path. It places and routes
one complete OTA from the same primitive PCells, preserves Magic's flattened
transistor-level PEX, maps NMOS substrate nodes to `VSS` and PMOS well nodes to
`VDD`, and rejects an extracted topology that does not match the six ports
`VOUT,VINP,VINN,IBIAS,VDD,VSS`. The Rust
`ota_4t_physical_verification` example invokes this helper automatically.
The complete OTA lifts its two drain nets to Metal2 with direct Via1 geometry;
this keeps the inter-primitive routes from crossing the local Metal1 gate and
source buses.
The complete multifinger terminal buses are identified by layout policy
`symmetric-adjacent-with-edge-dummies-v3`; the OTA examples reject older
archives so that data generated from partially connected fingers cannot be
used silently.

Because this version has one reduced port per logical net, it captures coupling
capacitance but not metal series resistance within that net. See the crate
README for the required port expansion in a future resistive-aware schema.
