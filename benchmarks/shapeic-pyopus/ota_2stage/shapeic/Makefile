ROOT := ../../../../

run-ota:
	cargo run \
		--manifest-path $(ROOT)/Cargo.toml \
		--release \
		-p shapeic-core \
		--example ota_2stage \
		-- \
		$(ROOT)/shapeic-lut/lut_generation/generated/ihp_sg13g2_lv_nmos_5d.npz \
		$(ROOT)/shapeic-lut/lut_generation/generated/ihp_sg13g2_lv_pmos_5d.npz \
		--workers 16 --batch-size 64 \
		> output.log 2>&1

	cp $(ROOT)/shapeic-core/examples/ota_2stage/results.csv .
	cp $(ROOT)/shapeic-core/examples/ota_2stage/summary.json .
