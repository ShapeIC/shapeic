.PHONY: ota_4t_sky130_pre_layout
ota_4t_sky130_pre_layout:
	PDK=sky130A cargo run --release -p shapeic-core --example ota_4t_sky130_pre_layout -- \
	    shapeic-lut/lut_generation/generated/sky130A_1v8_nmos_5d.npz \
	    shapeic-lut/lut_generation/generated/sky130A_1v8_pmos_5d.npz

.PHONY: ota_4t_v3_sky130_pre_layout
ota_4t_v3_sky130_pre_layout:
	PDK=sky130A cargo run --release -p shapeic-core --example ota_4t_v3 -- \
	    shapeic-lut/lut_generation/generated/sky130A_1v8_nmos_5d.npz \
	    shapeic-lut/lut_generation/generated/sky130A_1v8_pmos_5d.npz

.PHONY: ota_4t_v3_sky130
ota_4t_v3_sky130:
	PDK=sky130A cargo run --release -p shapeic-core --example ota_4t_v3 -- \
	    shapeic-lut/lut_generation/generated/sky130A_1v8_nmos_5d.npz \
	    shapeic-lut/lut_generation/generated/sky130A_1v8_pmos_5d.npz \
	    shapeic-layout/lut_generation/generated/sky130A_ota_physical_cellkit_v2.npz





.PHONY: ota_4t_gf180_pre_layout
ota_4t_gf180_pre_layout:
	PDK=gf180mcuD cargo run --release -p shapeic-core --example ota_4t_gf180_pre_layout -- \
	    shapeic-lut/lut_generation/generated/gf180mcuD_3v3_nmos_5d.npz \
	    shapeic-lut/lut_generation/generated/gf180mcuD_3v3_pmos_5d.npz

.PHONY: ota_4t_v3_gf180_pre_layout
ota_4t_v3_gf180_pre_layout:
	PDK=gf180mcuD cargo run --release -p shapeic-core --example ota_4t_v3 -- \
	    shapeic-lut/lut_generation/generated/gf180mcuD_3v3_nmos_5d.npz \
	    shapeic-lut/lut_generation/generated/gf180mcuD_3v3_pmos_5d.npz

.PHONY: ota_4t_v3_gf180
ota_4t_v3_gf180:
	PDK=gf180mcuD cargo run --release -p shapeic-core --example ota_4t_v3 -- \
	    shapeic-lut/lut_generation/generated/gf180mcuD_3v3_nmos_5d.npz \
	    shapeic-lut/lut_generation/generated/gf180mcuD_3v3_pmos_5d.npz \
	    shapeic-layout/lut_generation/generated/gf180mcuD_ota_physical_cellkit_v2.npz
