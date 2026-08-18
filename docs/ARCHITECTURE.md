# Shroud Engine Architecture

Shroud is structured as a layered simulation runtime where WebGPU Shading Language (WGSL) compute kernels perform parallel physical transport and sensor synthesis, an embedded Rhai scripting runtime orchestrates multi-epoch scenarios and physical invariant checks, and a Rust + C ABI workspace handles memory-aligned buffer management, Naga AST validation, and headless GPU execution.

## System Topology

```text
+-----------------------------------------------------------------------------+
|                        Scenario & Verification Layer                        |
|   scripts/ (400 Rhai modules)              tests/ (Rust, WGPU, C, Python)  |
|   - Weather & aerosol sweeps               - Naga WGSL AST verification     |
|   - Ego trajectories & Doppler             - Headless Metal/Vulkan dispatch |
|   - Material & sensor calibration          - Analytical cross-validation    |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                         Host Runtime & CLI (Rust)                           |
|   shroud-cli        : CLI runner (`validate`, `test-rhai`, `bench`, `info`) |
|   shroud-rhai       : Sandboxed Rhai engine with 50M ops quota & DSL types  |
|   shroud-validator  : Naga 24.0 WGSL parser, type checker, & reflection     |
|   shroud-wgpu       : WGPU 24.0 headless device, storage buffers, readback  |
|   shroud-c          : C99 shared library ABI (`include/shroud.h`)           |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                      Core Physical Primitives (`shroud-core`)               |
|   - CODATA 2022 constants (c, h, k_B, sigma)                                |
|   - std140/std430 aligned Vec2, Vec3, Vec4, Mat4, Ray, Aabb, GpuBvhNode     |
|   - 31-band CIE 1931 colorimetry & 8-14 um Planck LWIR quadrature           |
|   - Double Henyey-Greenstein, Rayleigh, & US Standard Atmosphere 1976       |
|   - ToF LiDAR, signed I/Q FMCW radar/LiDAR, & Brown-Conrady distortion      |
+--------------------------------------+--------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                  WebGPU Compute Kernel Corpus (`shaders/`)                  |
|   391 standalone WGSL kernels across 37 physical domains (30,513 lines)     |
|   - BVH / Wide-BVH8 / Radix Sort        - ReSTIR DI / GI & SVGF Denoise     |
|   - Volumetric RTE & Woodcock Tracking  - Wave Optics, Stokes-Mueller, TMM  |
|   - LiDAR / FMCW / LWIR / SPAD / MIMO   - Hyperspectral, Sonar, InSAR, SLAM |
+-----------------------------------------------------------------------------+
```

## Multi-Stage GPU Execution Pipeline

```mermaid
flowchart LR
    A["Rhai Scenario DSL (.rhai)"] --> B["Host Scene & Media Parameters (shroud-core)"]
    B --> C["Naga AST Validation & Reflection (shroud-validator)"]
    C --> D["WGPU Compute Pipeline & Bind Groups (shroud-wgpu)"]
    D --> E["BVH Traversal & Volumetric RTE Kernels (.wgsl)"]
    E --> F["Sensor Physics & Noise Synthesis (.wgsl)"]
    F --> G["Staging Buffer Copy & Host Readback"]
```

## Memory Alignment & Zero-Copy ABI Invariants

WGSL uniform and storage buffers follow strict `std140` / `std430` alignment constraints where 3-component vectors align to 16 bytes (`vec4<f32>` or padded `vec3<f32>`). Every GPU-shared Rust structure in `crates/shroud-core` enforces `#[repr(C, align(16))]` alongside `bytemuck::Pod` and `bytemuck::Zeroable`:

| Host Rust Type | Alignment | Size (bytes) | Matching WGSL Layout |
| --- | --- | --- | --- |
| `Vec3` | 16 bytes | 16 | `vec3<f32>` + 4-byte tail padding |
| `Vec4` | 16 bytes | 16 | `vec4<f32>` |
| `Mat4` | 16 bytes | 64 | `mat4x4<f32>` (4 column `Vec4`s) |
| `Ray` | 16 bytes | 48 | `origin`, `dir`, `t_min`, `t_max`, `_pad` |
| `Aabb` | 16 bytes | 32 | `min: vec3` (padded), `max: vec3` (padded) |
| `GpuBvhNode` | 16 bytes | 48 | `aabb_min`, `aabb_max`, child indices, count |
| `MediaProperties` | 16 bytes | 16 | `sigma_a`, `sigma_s`, `g`, `_pad` |

## Static Shader Verification & Reflection

Before any WGSL module enters a WGPU device pipeline, `shroud-validator` parses the text through `naga::front::wgsl::parse_str` and runs `naga::valid::Validator` with full capabilities enabled. `shroud_validator::reflection::reflect_wgsl` extracts entry points, storage access modes (`read` vs `read_write`), and `@group(G) @binding(B)` slots so host harnesses never drift out of sync with shader declarations.

## Sandboxed Scripting & C ABI Interoperability

1. **Rhai Runtime (`shroud-rhai`)**: Configures an operation ceiling of 50,000,000 instructions, recursion depth limit of 128 frames, and expression depth limit of 64. All custom Rust domain types (`ScriptVec3`, `ScriptRay`, `ScriptLidar`, `ScriptFmcw`, `ScriptScenario`) implement `Clone + Send + Sync + 'static` under Rhai's `sync` feature.
2. **C99 Shared Library (`shroud-c`)**: Exposes flat `extern "C"` entry points declared in `include/shroud.h` for calling atmospheric phase evaluations, LWIR blackbody integrals, 30-bit Morton bit-interleaving, Naga WGSL validation, and Rhai script evaluation from C, C++, or Python `ctypes`.
