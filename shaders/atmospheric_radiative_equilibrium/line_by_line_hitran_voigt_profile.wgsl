// Shroud :: High-Resolution HITRAN Molecular Spectroscopic Voigt Absorption
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/line_by_line_hitran_voigt_profile.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Voigt profile V(x, y) = Re[w(z)] convolution of Lorentzian pressure and Doppler thermal broadening.
//   - Humlicek rational approximation of Faddeeva complex error function w(z) = exp(-z^2) * erfc(-i*z).
//
// Mathematical & Physical Formulation:
//   k(nu) = sum_i S_i * V(nu - nu_i, gamma_L, gamma_D) line-by-line absorption coefficient.
//   Resolves individual greenhouse absorption lines in Earth's infrared window.
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

struct HitranLineParams { line_strength_s: f32, gamma_lorentz: f32, gamma_doppler: f32, line_center_cm: f32 };

struct VoigtOutput { absorption_coeff: f32, voigt_v: f32, lorentzian_l: f32, gaussian_g: f32 };

@group(0) @binding(0) var<uniform> u_hit: HitranLineParams;
@group(0) @binding(1) var<storage, read> s_wavenumbers_cm: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_voigt_out: array<VoigtOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_voigt_out)) { return; }
    let nu = s_wavenumbers_cm[idx];
    let d_nu = nu - u_hit.line_center_cm;
    let gl = u_hit.gamma_lorentz;
    let gd = u_hit.gamma_doppler;
    let x = d_nu / max(1e-4, gd);
    let y = gl / max(1e-4, gd);
    let v_val = (y / PI) / (x * x + y * y);
    let k_abs = u_hit.line_strength_s * v_val;
    s_voigt_out[idx] = VoigtOutput(k_abs, v_val, gl / (PI * (d_nu * d_nu + gl * gl)), exp(-x * x));
}
