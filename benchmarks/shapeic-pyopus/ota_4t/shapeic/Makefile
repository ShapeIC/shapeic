ROOT := ../../../../

run-ota:
	cargo run \
		--manifest-path $(ROOT)/Cargo.toml \
		--release \
		-p shapeic-core \
		--example ota_4t_v3 \
		-- \
		$(ROOT)/shapeic-lut/lut_generation/generated/ihp_sg13g2_lv_nmos_5d.npz \
		$(ROOT)/shapeic-lut/lut_generation/generated/ihp_sg13g2_lv_pmos_5d.npz \
		--electrical-workers 8 \
		--electrical-batch-size 64 \
		> output.log 2>&1

	cp $(ROOT)/shapeic-core/examples/ota_4t_v3/results.csv .
