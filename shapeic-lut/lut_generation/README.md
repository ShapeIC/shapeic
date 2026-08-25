# Shapeic LUT generation

This directory contains the offline Python/ngspice generator for five-dimensional
MOS lookup tables. Python is not required to load or query the generated files.

Create a virtual environment and install the only Python dependency:

```text
python3 -m venv .venv
.venv/bin/pip install -r shapeic-lut/lut_generation/requirements.txt
```

Set `PDK_ROOT` to the directory containing the installed PDKs and select one
with `PDK`, then run one of the checked configurations:

```text
export PDK_ROOT=/path/to/pdks
export PDK=ihp-sg13g2
.venv/bin/python shapeic-lut/lut_generation/generate.py \
  shapeic-lut/lut_generation/configs/ihp_sg13g2_lv_nmos.toml
```

The full IHP configurations characterize `length`, `vbs`, `vgs`, `vds`, and
`finger_width`. Width uses a linear grid from `0.15 um` through `9.65 um` with a
`0.5 um` step. Normal electrical parameters use `ng=1`. The `_smoke.toml`
configurations contain two values per axis and are intended for end-to-end
checks.

Generation refuses to replace an existing output unless `--force` is supplied.
`workers` controls independent ngspice processes; its default is one. A complete
file is moved into place only after all blocks have been simulated and checked
for missing or non-finite values.

The checked IHP configurations save nine independent intrinsic
charge-derivative coefficients: `cgg,cgd,cgs`, `cdg,cdd,cds`, and
`csg,csd,css`. They are interpolated together with `gm` and `gds` in the final
inverse-sizing query. Rust reconstructs the seven bulk-related coefficients
from charge conservation, applies ngspice's mutual-term sign convention, and
validates the resulting complete 4x4 nodal matrix.

The typed `capacitance_convention` setting normalizes native simulator outputs
before they are written. IHP uses `compact_mutual`; SKY130 and GF180 use
`signed_nodal`, whose six mutual derivatives are sign-converted into ShapeIC's
canonical compact-mutual representation. Existing LUT files remain readable.

The four extrinsic capacitances `cgsol`, `cgdol`, `cjs`, and `cjd` are sampled
at `nf = [1, 2, 3, 4]`. For each sample, the netlist uses `w = finger_width * nf`
and `ng = nf`, so the five-dimensional width coordinate remains the width of one
finger. The `nf=1` arrays keep their base names; the other samples use names such
as `cgsol_nf2`, `cgsol_nf3`, and `cgsol_nf4`. All arrays remain five-dimensional
and the archive remains format version 2.

At query time, Rust uses the `nf=1,3` samples for the odd-finger affine branch
and `nf=2,4` for the even-finger branch. This follows the IHP wrapper's separate
source/drain geometry formulas for odd and even finger counts. The setting
`capacitance_nf_samples` is currently restricted to exactly `[1, 2, 3, 4]`.

Validate one configured operating point independently with:

```text
.venv/bin/python shapeic-lut/lut_generation/probe.py CONFIG \
  --length 8e-7 --finger-width 2.15e-6 \
  --vbs 0 --vgs 0.65 --vds 0.65
```

The probe automatically evaluates `nf=1,2,3,4`. It compares the reconstructed
OP capacitance matrix against individual G/D/S/B AC excitations at 1 MHz and
10 MHz, checks charge conservation and `Id/gm/gds` scaling, and exits with
failure when a 1% matrix/scaling gate is exceeded. Netlists, logs, raw files,
parameter bindings, matrices, and `report.json` remain under
`target/shapeic-electrical-probe/<timestamp>` on both success and failure.
