# Verification & Benchmarks

All verification metrics below were measured directly on this repository (`Apple M2`, `macOS`, `Metal` WGPU backend, `naga 24.0`, `rhai 1.20`).

## Measured Execution Metrics (`shroud bench`)

| Stage | Workload | Observed Timing |
| --- | --- | --- |
| Naga AST + Type Checker | 386 WGSL compute modules parsed & validated | 604.35 ms |
| Naga Full Corpus (`validate shaders/`) | 391 WGSL compute modules parsed & validated | 606.40 ms |
| Rhai Physics DSL (`bench`) | 5,000 Henyey-Greenstein phase evaluations | 29.25 ms |
| Rhai Full Scenario Suite (`test-rhai scripts/`) | 400 independent `.rhai` simulation scripts | 2.63 s |
| WGPU Headless Metal Dispatch (`bench`) | 65,536 work-items compiled, dispatched, & read back | 16.83 ms |

## Multi-Language Verification Suites

| Suite | Command | Scope & Observed Result |
| --- | --- | --- |
| Rust Unit & Integration Tests | `make test` (`cargo test --workspace`) | 28 tests passed (`16` core, `4` rhai, `3` validator, `2` Metal GPU dispatch, `3` workspace integration over all 391 WGSL + 400 Rhai scripts) |
| Headless GPU Physics Cross-Check | `cargo test -p shroud-wgpu --test gpu_dispatch_tests` | 2 passed (`rng.wgsl` distribution & `phase_henyey_greenstein.wgsl` vs Rust CPU reference `< 1e-4` on Apple M2 Metal) |
| C99 ABI Integration Suite | `make test-c` | Compiled cleanly with `-Wall -Wextra`, verified version, HG, Rayleigh, LWIR, 905 nm LiDAR inverse-square, Morton3D, WGSL Naga, & Rhai eval |
| Python Analytical Verification | `make test-python` | 5 passed (`unittest` analytical quadrature + `ctypes` FFI cross-check against `libshroud_c`) |
| Static Compliance Gate | `make compliance` (`python3 tools/check_compliance.py`) | 860 scanned files, zero em-dashes, zero prohibited buzzwords |

## Codebase Breakdown (`tools/count_loc.py`)

| Language | Raw Lines | Share |
| --- | --- | --- |
| WGSL (`shaders/`) | 30,513 | 39.3% |
| Rhai (`scripts/`) | 30,273 | 39.0% |
| Python (`tools/`, `tests/python/`) | 13,848 | 17.7% |
| Rust (`crates/`) | 3,152 | 3.7% |
| TOML (`Cargo.toml`) | 125 | 0.2% |
| C / C Header (`tests/`, `include/`) | 104 | 0.1% |
