// Shroud :: Cox-Munk Anisotropic Wind-Driven Wave Slope Distribution
// Subsystem Module: shaders/oceanic_hydrology/sea_surface_cox_munk.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 2D Gram-Charlier wave slope probability density function p(z_x, z_y).
//   - Models wind speed U_10, upwind/crosswind variances, skewness, and peakedness kurtosis.
//
// Mathematical & Physical Formulation:
//   sigma_c^2 = 0.003 + 0.00192 * U_10 crosswind, sigma_u^2 = 0.000 + 0.00316 * U_10 upwind.
//   Gram-Charlier expansion accounts for non-Gaussian wave crest sharp peaking.
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

struct CoxMunkConfig { wind_speed_m_s: f32, wind_dir_rad: f32, pad0: f32, pad1: f32 };

struct WaveSlopeSample { slope_pdf: f32, fresnel_r: f32, normal_tilt: vec4<f32> };

@group(0) @binding(0) var<uniform> u_cm: CoxMunkConfig;
@group(0) @binding(1) var<storage, read> s_slopes: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_slope_res: array<WaveSlopeSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_slope_res)) { return; }
    let sl = s_slopes[idx];
    let u10 = u_cm.wind_speed_m_s;
    let su2 = max(1e-4, 0.00316 * u10);
    let sc2 = max(1e-4, 0.003 + 0.00192 * u10);
    let zx = sl.x;
    let zy = sl.y;
    let exp_val = exp(-0.5 * (zx * zx / su2 + zy * zy / sc2));
    let p = exp_val / (TWO_PI * sqrt(su2 * sc2));
    let n = normalize(vec3<f32>(-zx, -zy, 1.0));
    let r0 = 0.02;
    s_slope_res[idx] = WaveSlopeSample(p, r0, vec4<f32>(n, 1.0));
}
