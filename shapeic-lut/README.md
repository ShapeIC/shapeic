# shapeic-lut

`shapeic-lut` loads SSTADEx and Shapeic MOS lookup tables directly in Rust. It
exposes stored axes and parameter arrays, evaluates typed expressions, and
performs linear interpolation over `length`, `vbs`, `vgs`, and `vds`. Shapeic
NPZ v2 files add `finger_width` as a fifth dimension.

The legacy reader supports the SSTADEx `lookup_table.npy` object schema. New
files use independent `.npy` arrays and a JSON manifest inside the `.npz`, so
they do not require pickle. Python is only needed for offline LUT generation;
loading and querying remain native Rust operations.

```rust,no_run
use shapeic_lut::{LookupTable, MosExpression, OperatingPoint};

let table = LookupTable::open("nmos.npz")?;
let model = table.model("sg13_lv_nmos")?;
let point = OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6);

let drain_current = model.query_parameter(&point, "id")?;
let gmid = model.standard_expression(MosExpression::Gmid)?;
let gmid_value = model.query_expression(&point, &gmid)?;

# let _ = (drain_current, gmid_value);
# Ok::<(), shapeic_lut::LutError>(())
```

Expressions are evaluated at every surrounding LUT corner before their values
are interpolated. This matches the evaluation order used by `gmid` for derived
quantities such as `gm / id`.

Five-dimensional files use `LutPoint`, keeping geometry separate from the
four-coordinate `OperatingPoint`:

```rust,no_run
use shapeic_lut::{LookupTable, LutPoint, MosExpression, OperatingPoint};

let table = LookupTable::open("nmos-5d.npz")?;
let model = table.model("sg13_lv_nmos")?;
let point = LutPoint::new(
    OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6),
    0.5e-6,
);
let id = model.query_parameter_at(&point, "id")?;
let jd = model.standard_expression(MosExpression::CurrentDensity)?;
let jd_value = model.query_expression_at(&point, &jd)?;

# let _ = (id, jd_value);
# Ok::<(), shapeic_lut::LutError>(())
```

The existing `query_parameter`, `query_expression`, and `query_many` methods
remain specific to four-dimensional LUTs. Their `*_at` counterparts require a
`LutPoint` and interpolate 32 surrounding corners.

## Inverse current sizing

A five-dimensional LUT can select a finger width and integer finger count for
a requested total drain current. The smallest finger count that can reproduce
the current is used. If integer finger counts leave a gap between attainable
current ranges, the closest boundary is returned and exposed through
`current_error`.

```rust,no_run
use shapeic_lut::{Expr, LookupTable, OperatingPoint};

let table = LookupTable::open("nmos-5d.npz")?;
let model = table.model("sg13_lv_nmos")?;
let point = OperatingPoint::new(0.4e-6, 0.0, 0.6, 0.6);
let outputs = [Expr::parameter("gm"), Expr::parameter("gds")];
let sizing = model.size_for_current(&point, 250.0e-6, &outputs)?;

// Expression values describe one finger. Extensive values are scaled explicitly.
let total_gm = sizing.values[0] * f64::from(sizing.nf);
let total_gds = sizing.values[1] * f64::from(sizing.nf);
let extrinsic_capacitances = sizing.extrinsic_capacitances;
println!(
    "Wf={} nf={} W={} Id={} error={}",
    sizing.point.finger_width,
    sizing.nf,
    sizing.total_width,
    sizing.predicted_current,
    sizing.current_error,
);

# let _ = (total_gm, total_gds, extrinsic_capacitances);
# Ok::<(), shapeic_lut::LutError>(())
```

`LookupTable::open` reads and decompresses the archive once. Inverse sizing and
the requested expression evaluations operate on the same in-memory
`DeviceLut`; the `.npz` is not opened a second time.

New Shapeic LUTs store `cgsol`, `cgdol`, `cjs`, and `cjd` at
`nf = [1, 2, 3, 4]`. `size_for_current` evaluates separate affine branches for
odd and even finger counts and returns the total device values through
`extrinsic_capacitances`. Older LUTs without the sampled anchors remain
supported and return `None`. A direct five-dimensional query can use
`query_extrinsic_capacitances_at(point, nf)`.

## LUT generation

`lut_generation/` contains a standalone Python/ngspice generator and TOML
configurations for IHP SG13G2 LV NMOS and PMOS. The full configurations use
`ng=1` and a linear `finger_width` grid from `0.15 um` through `9.65 um`.
Generation instructions and smaller end-to-end configurations are documented
in `lut_generation/README.md`.

The generator is offline tooling. A normal Rust lookup, including inverse
current sizing, never launches ngspice.

## NGSpice verification

The public `shapeic_lut::verification` module compares values calculated by the
caller with transistor-level `.op` simulations. It does not access a LUT or
evaluate expressions internally:

```rust,no_run
use shapeic_lut::verification::{
    VerificationConfig, VerificationEngine, VerificationInput,
};
use shapeic_lut::OperatingPoint;

let input = VerificationInput::new(
    OperatingPoint::new(0.6e-6, 0.0, 0.6, 0.6),
    7.5e-6,
    1,
)
.reference("id", 250.0e-6)
.reference("gm", 1.5e-3)
.reference("gm_id", 6.0)
.reference("jd", 250.0e-6 / 7.5e-6);
let config = VerificationConfig::new("template.spice", "verification-output")
    .max_width_per_finger(10.0e-6);
let engine = VerificationEngine::new(config)?;
let report = engine.verify_one(&input)?;
println!("{}", report.render_table());

# let _ = report;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The output directory must not already exist. Each point retains its rendered
netlist, NGSpice log, and `wrdata` result; `summary.csv` contains one row per
metric. Its `reference_value` column is exactly the LUT-derived or manually
calculated value supplied by the caller, while `simulation_value` contains the
NGSpice result. Percentage error uses NGSpice as the reference and is empty
when its value is zero. `VerificationReport::render_table` presents the same
comparison as an aligned terminal table without making the engine write to
standard output.

Templates use strict `{{name}}` placeholders. The engine always supplies
`width`, `nf`, `length`, `vbs`, `vgs`, `vds`, and `results_path`;
technology-specific values come from `template_variable`. A template must run
one `.op`, enable `wr_vecnames` and `wr_singlescale`, and write exactly one
named data row with `wrdata`. Reference names must match its column names
exactly; extra simulated columns are ignored.

For multi-device verification, declare per-candidate placeholders with
`VerificationConfig::dynamic_template_variable` and assign each one through
`VerificationInput::dynamic_template_variable`. Every input must provide
exactly the declared set; reserved, missing, unexpected, or static/dynamic
conflicting names are rejected before NGSpice runs.

`width` is the explicit total width, `nf` is the positive integer number of
fingers, and `length` comes from `OperatingPoint`. The engine renders `width`
and `nf` directly without dividing or reinterpreting them. When
`max_width_per_finger` is configured, inputs must satisfy the strict relation
`width / nf < maximum`; no technology-specific limit is applied by default.
The IHP templates write `ng={{nf}}` because the `sg13_lv_*` wrapper maps its
external `ng` parameter to the OSDI model's `nf` parameter.
They also expose `jd = id / width`, in `A/m` when current and width use SI
units; this is numerically equivalent to `uA/um`.

The IHP SG13G2 NMOS and PMOS templates are under `examples/verification`. Run
the complete local example with:

```text
cargo run --release -p shapeic-lut --example ihp-sg13g2_verification
```
