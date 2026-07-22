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
`0.5 um` step and `ng=1`. The `_smoke.toml` configurations contain two values per
axis and are intended for end-to-end checks.

Generation refuses to replace an existing output unless `--force` is supplied.
`workers` controls independent ngspice processes; its default is one. A complete
file is moved into place only after all blocks have been simulated and checked
for missing or non-finite values.
