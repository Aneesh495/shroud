# Makefile for Shroud :: WebGPU Spectral Radiative Transfer & Sensor Simulation Engine

.PHONY: all build check test test-wgsl test-rhai test-c test-python compliance bench clean

CARGO ?= cargo
PYTHON ?= python3
CC ?= cc

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHROUD_LIB := target/debug/libshroud_c.dylib
    RPATH_FLAG := -Wl,-rpath,@executable_path/../target/debug
else
    SHROUD_LIB := target/debug/libshroud_c.so
    RPATH_FLAG := -Wl,-rpath,'$$ORIGIN/../target/debug'
endif

all: check test test-c test-python compliance

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

test-c:
	$(CARGO) build -p shroud-c
	mkdir -p output
	$(CC) -O2 -Wall -Wextra tests/c_abi_test.c $(SHROUD_LIB) $(RPATH_FLAG) -lm -o output/c_abi_test
	./output/c_abi_test

test-python:
	$(CARGO) build -p shroud-c
	$(PYTHON) -m unittest discover -s tests/python -p "test_*.py" -v

compliance:
	$(PYTHON) tools/check_compliance.py

bench:
	$(CARGO) run -p shroud-cli --release -- bench

clean:
	$(CARGO) clean
	rm -rf output/ bench_results/
