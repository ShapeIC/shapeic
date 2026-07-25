# Shapeic LUT generation

This directory contains the offline Python/ngspice generator for five-dimensional
MOS lookup tables. Python is not required to load or query the generated files.

Create a virtual environment and install the only Python dependency:

```text
python3 -m venv .venv
.venv/bin/pip install -r shapeic-lut/lut_generation/requirements.txt
```

Set `PDK_ROOT` to the directory containing the `ihp-sg13g2` PDK directory, then
run one of the checked configurations:

```text
export PDK_ROOT=/path/to/IHP-Open-PDK
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

The checked IHP configurations save the complete intrinsic charge-derivative
matrix in terminal order `G,D,S,B`. This is required by the layout-aware OTA
example; the matrix is interpolated together with `gm` and `gds` in the final
inverse-sizing query rather than through sixteen independent queries.
The archive keeps ngspice's raw mutual-term convention; the Rust reader applies
the nodal signs and validates charge conservation after interpolation.

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
