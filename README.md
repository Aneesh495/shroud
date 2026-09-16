# Shroud

**WebGPU Spectral Radiative Transfer, Multi-Sensor Physics, and Embedded Rhai Simulation Engine**

Shroud is a headless WebGPU compute and physical simulation runtime built around **391 standalone WGSL compute kernels (30,513 lines)** and **400 sandboxed Rhai scenario scripts (30,273 lines)**, backed by a Rust host engine (`wgpu 24.0`, `naga 24.0`, `rhai 1.20`), a C99 shared library ABI, and Python analytical verification suites.

Instead of treating atmospheric degradation and optical sensor pipelines as post-processing screen effects, Shroud solves spectral radiative transfer, null-collision participating media transport, wave optics diffraction, and coherent LiDAR/radar/LWIR detector physics directly on the GPU.

---

## Architecture Overview

```text
  +-------------------------------+       +-----------------------------------+
  |     Rhai Scenario DSL         |       |   Multi-Language Verification     |
  |  400 scripts / 30,273 lines   |       |  Rust tests | C ABI | Python FFI  |
  +---------------+---------------+       +-----------------+-----------------+
                  |                                         |
                  v                                         v
  +---------------------------------------------------------------------------+
  |                       Rust Host Orchestrator (`crates/`)                  |
  |                                                                           |
  |  shroud-core       : CODATA 2022 constants, std140/430 math, Planck LWIR, |
  |                      double Henyey-Greenstein, ToF/FMCW/Brown-Conrady     |
  |  shroud-validator  : Naga 24.0 WGSL AST parser, typechecker, & reflection |
  |  shroud-rhai       : Sandboxed Rhai 1.20 VM (50M ops ceiling) + DSL types |
  |  shroud-wgpu       : Headless Vulkan/Metal/DX12 compute dispatch harness  |
  |  shroud-c          : C99 shared ABI (`include/shroud.h`)                  |
  |  shroud-cli        : `validate`, `test-rhai`, `bench`, `verify`, `info`   |
  +-------------------------------------+-------------------------------------+
                                        |
                                        v
  +---------------------------------------------------------------------------+
  |                  WGSL Compute Kernel Corpus (`shaders/`)                  |
  |                      391 modules / 30,513 raw lines                       |
  |                                                                           |
  |  [Acceleration]  LBVH Morton, GPU Radix Sort, Karras Builder, BVH4/BVH8   |
  |  [Radiative]     Woodcock Delta Tracking, Ratio Tracking, Hero Spectral   |
  |  [Resampling]    ReSTIR DI & GI, SVGF A-Trous Denoising, Spherical Harm.  |
  |  [BSDF/Materials]GGX, Disney BSDF, Dipole/Walk BSSRDF, 40-Mat Sellmeier   |
  |  [Sensors]       905/1550nm ToF & FMCW LiDAR, SPAD, 8-14um LWIR, 4D MIMO  |
  |  [Wave & Astro]  Zernike PSF, Stokes-Mueller, TMM, AO, SOI Photonics, SAR |
  +---------------------------------------------------------------------------+
```

---

## Core Capabilities

- **Participating Media & Volumetric RTE**: Woodcock null-collision delta tracking, variance-reduced ratio tracking, spectral hero wavelength tracking, Rayleigh molecular dipolar scattering, double Henyey-Greenstein forward-backward lobes, and 1976 US Standard Atmosphere density stratification.
- **Physical Sensor Pipelines**:
  - **Pulsed ToF & Full-Waveform LiDAR**: APD gain/excess noise statistics, CFD sub-bin peak detection, multi-Gaussian waveform decomposition, and Wiener IRF deconvolution.
  - **Coherent FMCW LiDAR & 77 GHz Radar**: Signed complex I/Q intermediate beat frequency recovery preserving range-Doppler decoupling when optical 1550 nm Doppler shifts exceed range beat frequencies, plus 4D MIMO virtual arrays and CA/OS-CFAR detection.
  - **Thermal LWIR & Optical Cameras**: Midpoint quadrature integration of Planck's blackbody law across 8 to 14 um, 31-band CIE 1931 colorimetry, and Brown-Conrady radial/tangential lens distortion.
- **GPU Acceleration & Resampling**: 30-bit Morton LBVH construction, parallel radix histogram/scan/scatter, Karras binary tree assembly, wide `BVH4`/`BVH8` culling, ReSTIR DI/GI spatiotemporal reservoir resampling, and SVGF wavelet denoising.
- **Wave Optics, Photonics & Scientific Extensions**: Zernike wavefront aberrations, Fresnel CZT diffraction, Abeles 2x2 thin-film transfer matrices, Stokes-Mueller polarization transport, Shack-Hartmann adaptive optics, SOI microring/MZI photonics, SNSPD/TES cryogenics, hyperspectral FCLS unmixing, acoustic sonar, and InSAR geodesy.
- **Sandboxed Rhai Orchestration**: 400 scenario, calibration, qualification, and trajectory scripts verified against live physics bindings and Naga shader compilation.

---

## Repository Structure

```text
shroud/
├── crates/
│   ├── shroud-core/        # Physical constants, aligned structs, spectral & sensor math
│   ├── shroud-validator/   # Naga 24.0 WGSL parser, validator, and binding reflection
│   ├── shroud-rhai/        # Sandboxed Rhai 1.20 scripting engine and domain DSL
│   ├── shroud-wgpu/        # Headless WGPU 24.0 device, pipeline, and buffer harness
│   ├── shroud-c/           # C99 FFI cdylib exports
│   └── shroud-cli/         # Unified CLI (`validate`, `test-rhai`, `bench`, `verify`)
├── include/
│   └── shroud.h            # Public C99/C++ ABI header
├── shaders/                # 391 validated WGSL compute kernels across 37 subsystems
├── scripts/                # 400 executable Rhai scenarios across 15 test categories
├── tests/
│   ├── c_abi_test.c        # C99 shared library integration test suite
│   └── python/             # Analytical physics & ctypes FFI verification suite
├── tools/                  # Compliance gate, LOC counter, and deterministic generators
├── docs/                   # Architecture, physics derivations, sensor & API references
└── Makefile                # Single-command workspace build, test, and verification targets
```

---

## Quickstart

### Prerequisites
- Rust toolchain (2021 edition, Cargo)
- Python 3 (for analytical tests and compliance checks)
- C compiler (`cc` / `clang` / `gcc`, for C ABI integration tests)
- Vulkan, Metal, or DX12 compatible GPU driver (for headless `shroud bench` and WGPU dispatch tests)

### Build & Run Full Verification

```bash
# 1. Check workspace, run all Rust unit/GPU/integration tests, C ABI tests, Python suite, and compliance checks
make all

# 2. Validate every WGSL kernel in shaders/ through Naga
cargo run --release -p shroud-cli -- validate shaders/

# 3. Execute all 400 Rhai scenario scripts
cargo run --release -p shroud-cli -- test-rhai scripts/

# 4. Run Naga + Rhai + headless WGPU Metal/Vulkan GPU benchmark
cargo run --release -p shroud-cli -- bench

# 5. Inspect repository SLOC breakdown
python3 tools/count_loc.py
```

---

## Observed Verification Results

Measured locally on Apple M2 (macOS, Metal WGPU backend):

- **Naga WGSL Corpus**: `391 passed, 0 failed` in `606.40 ms`
- **Rhai Scenario Corpus**: `400 passed, 0 failed` in `2.63 s`
- **Headless WGPU Dispatch**: `65,536` compute work-items compiled, executed, and read back in `16.83 ms`, plus cross-checked `rng.wgsl` and `phase_henyey_greenstein.wgsl` matching CPU reference within `1e-4`
- **Rust Test Workspace (`make test`)**: `28 passed, 0 failed` across unit, GPU dispatch, and full-corpus integration suites
- **C ABI (`make test-c`)**: Compiled cleanly with `-Wall -Wextra` and passed all FFI checks
- **Python Analytical (`make test-python`)**: 5 closed-form quadrature & `ctypes` FFI cross-validation tests passed

---

## Documentation

- [System Architecture & Memory Layouts](docs/ARCHITECTURE.md)
- [Radiative Transfer & Participating Media](docs/RADIATIVE_TRANSFER.md)
- [Sensor Physics & Calibration Equations](docs/SENSOR_MODELS.md)
- [Rhai Scripting DSL Reference](docs/RHAI_API.md)
- [WGSL Compute Kernel Catalog](docs/WGSL_KERNELS.md)
- [Verification & Benchmark Metrics](docs/BENCHMARKS.md)

## License

Licensed under the [ISC License](LICENSE).
