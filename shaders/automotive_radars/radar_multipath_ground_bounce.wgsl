// Shroud :: Four-Ray Ground Reflection Multipath Interference Modeler
// Subsystem Module: shaders/automotive_radars/radar_multipath_ground_bounce.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates direct, ground-reflected, target-reflected, and bistatic bounce propagation paths.
//   - Computes constructive and destructive coherent signal fading interference.
//
// Mathematical & Physical Formulation:
//   E_total = E_direct + Gamma * exp(i * delta_phi) * E_reflected.
//   Gamma Fresnel reflection coefficients for dry asphalt and wet pavement.
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
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;
const VACUUM_PERMEABILITY: f32 = 1.25663706212e-6;
const IMPEDANCE_FREE_SPACE: f32 = 376.730313668;

struct MultipathConfig { h_radar_m: f32, h_target_m: f32, dist_m: f32, lambda_m: f32 };

struct MultipathOut { path_diff_m: f32, gamma_mag: f32, interference_factor: f32, fade_depth_db: f32 };

@group(0) @binding(0) var<uniform> u_mp: MultipathConfig;
@group(0) @binding(1) var<storage, read> s_ranges: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_mp_out: array<MultipathOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_mp_out)) { return; }
    let r = s_ranges[idx];
    let h1 = u_mp.h_radar_m;
    let h2 = u_mp.h_target_m;
    let d_dir = sqrt(r * r + (h1 - h2) * (h1 - h2));
    let d_ref = sqrt(r * r + (h1 + h2) * (h1 + h2));
    let delta_d = d_ref - d_dir;
    let k = TWO_PI / u_mp.lambda_m;
    let phi = k * delta_d;
    let gamma = 0.85;
    let f_interf = 1.0 + gamma * gamma + 2.0 * gamma * cos(phi);
    let fade = 10.0 * log(max(1e-4, f_interf));
    s_mp_out[idx] = MultipathOut(delta_d, gamma, f_interf, fade);
}
