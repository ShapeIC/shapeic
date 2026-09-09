.PHONY: all
all: run-pyopus run-shapeic plot-design-space

.PHONY: run-pyopus
run-pyopus:
	cd pyopus && ../../ota_4t/pyopus/.venv-pyopus/bin/python ota_2stage_opt.py

.PHONY: run-shapeic
run-shapeic:
	$(MAKE) -C shapeic

.PHONY: plot-design-space
plot-design-space:
	python plot.py
