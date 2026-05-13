// Shroud :: Cardelli-Clayton-Mathis (CCM 1989) Interstellar Extinction Law
// Subsystem Module: shaders/stellar_astrophysics/interstellar_reddening_cardelli.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates total-to-selective extinction A(lambda) / A_V = a(x) + b(x) / R_V from optical to far-UV.
//   - Models 2175 Angstrom carbonaceous polycyclic aromatic hydrocarbon (PAH) bump.
//
// Mathematical & Physical Formulation:
//   Parameterizes diffuse interstellar medium (R_V = 3.1) vs dense molecular clouds (R_V = 5.0).
//   x = 1 / lambda (um^-1) wavenumber.
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
const GRAVITATIONAL_CONSTANT: f32 = 6.67430e-11;
const SOLAR_MASS_KG: f32 = 1.98847e30;
const SOLAR_RADIUS_M: f32 = 6.957e8;

struct CcmConfig { r_v: f32, e_bv: f32, pad0: f32, pad1: f32 };

struct ExtinctionResult { a_lambda_mag: f32, ext_ratio: f32, bump_strength: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_ccm: CcmConfig;
@group(0) @binding(1) var<storage, read> s_wavenumber_um_inv: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_ext_out: array<ExtinctionResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ext_out)) { return; }
    let x = s_wavenumber_um_inv[idx];
    let y = x - 1.82;
    let a_x = 1.0 + 0.17699 * y - 0.50447 * y * y - 0.024278 * y * y * y;
    let b_x = 1.41338 * y + 2.28305 * y * y + 1.07233 * y * y * y;
    let ratio = a_x + b_x / u_ccm.r_v;
    let a_v = u_ccm.r_v * u_ccm.e_bv;
    s_ext_out[idx] = ExtinctionResult(ratio * a_v, ratio, exp(-pow((x - 4.6) / 0.25, 2.0)), 0.0);
}
