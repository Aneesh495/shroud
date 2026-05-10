// Shroud :: Evanescent Wave Penetration Depth and Fluorophore Excitation
// Subsystem Module: shaders/computational_microscopy/total_internal_reflection_tirf.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates supercritical angle total internal reflection at glass-water interface (theta > theta_c).
//   - Computes exponentially decaying evanescent field: I(z) = I_0 * exp(-z / d_p).
//
// Mathematical & Physical Formulation:
//   d_p = lambda / (4 * pi * sqrt(n_glass^2 * sin(theta)^2 - n_water^2)) penetration depth (approx 80-150 nm).
//   Restricts excitation selectively to cell membrane plasma boundaries.
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
const AVOGADRO_NUMBER: f32 = 6.02214076e23;

struct TirfConfig { n_glass: f32, n_water: f32, theta_inc_deg: f32, lambda_nm: f32 };

struct TirfExcitation { z_depth_nm: f32, evanescent_intensity: f32, penetration_depth_nm: f32, is_supercritical: u32 };

@group(0) @binding(0) var<uniform> u_tirf: TirfConfig;
@group(0) @binding(1) var<storage, read> s_z_nm: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_tirf_out: array<TirfExcitation>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_tirf_out)) { return; }
    let z = s_z_nm[idx];
    let th_rad = u_tirf.theta_inc_deg * (PI / 180.0);
    let sin_th = sin(th_rad);
    let crit_sin = u_tirf.n_water / u_tirf.n_glass;
    let is_super = select(0u, 1u, sin_th > crit_sin);
    let arg = u_tirf.n_glass * u_tirf.n_glass * sin_th * sin_th - u_tirf.n_water * u_tirf.n_water;
    let dp = u_tirf.lambda_nm / (FOUR_PI * sqrt(max(1e-4, arg)));
    let int_ev = exp(-z / max(1.0, dp));
    s_tirf_out[idx] = TirfExcitation(z, int_ev, dp, is_super);
}
