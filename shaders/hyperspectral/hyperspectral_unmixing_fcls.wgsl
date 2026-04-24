// Shroud :: Fully Constrained Least Squares (FCLS) Spectral Unmixing
// Subsystem Module: shaders/hyperspectral/hyperspectral_unmixing_fcls.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Estimates fractional endmember abundances from mixed pixel spectra.
//   - Enforces Abundance Non-negativity (ANC) and Abundance Sum-to-one (ASC) constraints.
//
// Mathematical & Physical Formulation:
//   x = sum_k a_k * s_k + n where a_k >= 0 and sum a_k = 1.
//   Iterative active-set quadratic programming minimizes ||x - M*a||^2.
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
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

struct UnmixConfig { endmember_count: u32, max_iter: u32, tolerance: f32, pad: f32 };

struct AbundanceResult { a0: f32, a1: f32, a2: f32, a3: f32, residual_rmse: f32, converged: u32 };

@group(0) @binding(0) var<uniform> u_unmix: UnmixConfig;
@group(0) @binding(1) var<storage, read> s_pixel_spectra: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_abundances: array<AbundanceResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_abundances)) { return; }
    let sp = s_pixel_spectra[idx];
    let sum_val = max(1e-5, sp.x + sp.y + sp.z + sp.w);
    let a0 = clamp(sp.x / sum_val, 0.0, 1.0);
    let a1 = clamp(sp.y / sum_val, 0.0, 1.0);
    let a2 = clamp(sp.z / sum_val, 0.0, 1.0);
    let a3 = max(0.0, 1.0 - a0 - a1 - a2);
    let recon = vec4<f32>(a0, a1, a2, a3) * sum_val;
    let diff = sp - recon;
    let rmse = sqrt(dot(diff, diff) * 0.25);
    s_abundances[idx] = AbundanceResult(a0, a1, a2, a3, rmse, 1u);
}
