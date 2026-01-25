# Makefile for Shroud :: WebGPU Spectral Radiative Transfer & Sensor Simulation Engine

.PHONY: all build check test test-wgsl test-rhai compliance bench clean

CARGO ?= cargo
PYTHON ?= python3

all: check test compliance

build:
	$(CARGO) build --workspace --release

check:
	$(CARGO) check --workspace

test:
	$(CARGO) test --workspace

test-wgsl:
	$(CARGO) run -p shroud-cli -- validate shaders/

test-rhai:
	$(CARGO) run -p shroud-cli -- test-rhai scripts/

compliance:
	$(PYTHON) tools/check_compliance.py

bench:
	$(CARGO) run -p shroud-cli --release -- bench

clean:
	$(CARGO) clean
	rm -rf output/ bench_results/
