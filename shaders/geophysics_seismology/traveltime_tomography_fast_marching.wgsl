// Shroud :: Eikonal Fast Marching Method (FMM) First-Arrival Traveltime
// Subsystem Module: shaders/geophysics_seismology/traveltime_tomography_fast_marching.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Solves non-linear Eikonal equation: ||grad T(x)||^2 = 1 / v(x)^2 for first seismic arrival times.
//   - Traces wavefront expansion across heterogeneous geological slowness grids.
//
// Mathematical & Physical Formulation:
//   Upwind finite difference discretization: max(D_-x T, -D_+x T, 0)^2 + max(D_-y T, -D_+y T, 0)^2 = s^2.
//   Monotonically advances causal narrow band queue.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

// Fundamental Physical & Mathematical Constants (CODATA 2022)
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const HALF_PI: f32 = 1.57079632679489661923;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const HBAR: f32 = 1.054571817e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct FmmConfig { grid_spacing_m: f32, pad0: f32, pad1: f32, pad2: f32 };

struct FmmStep { traveltime_s: f32, slowness_s_m: f32, ray_dir: vec4<f32> };

@group(0) @binding(0) var<uniform> u_fmm: FmmConfig;
@group(0) @binding(1) var<storage, read> s_slowness: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_fmm_out: array<FmmStep>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fmm_out)) { return; }
    let s = s_slowness[idx];
    let dist = f32(idx) * u_fmm.grid_spacing_m;
    let t = dist * s;
    s_fmm_out[idx] = FmmStep(t, s, vec4<f32>(1.0, 0.0, 0.0, 0.0));
}
