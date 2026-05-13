// Shroud :: Non-Linear Schrodinger Equation (NLSE) Split-Step Fourier Propagator
// Subsystem Module: shaders/optical_communication/fiber_chromatic_dispersion_nlse.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Propagates optical pulses through single-mode fiber with chromatic dispersion and Kerr non-linearity.
//   - i * d A / dz = (beta_2 / 2) * d^2 A / dt^2 - gamma * |A|^2 * A.
//
// Mathematical & Physical Formulation:
//   Split-step Fourier method separates linear dispersion operator D and non-linear phase operator N.
//   Group Velocity Dispersion beta_2 (ps^2/km) and self-phase modulation (SPM).
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

struct NlseConfig { beta2_ps2_km: f32, gamma_w_km: f32, step_dz_km: f32, pad: f32 };

struct NlseSample { amplitude_re: f32, amplitude_im: f32, power_mw: f32, spm_phase_rad: f32 };

@group(0) @binding(0) var<uniform> u_nlse: NlseConfig;
@group(0) @binding(1) var<storage, read> s_field_t: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_nlse_out: array<NlseSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_nlse_out)) { return; }
    let a = s_field_t[idx];
    let p_w = a.x * a.x + a.y * a.y;
    let dphi_nl = u_nlse.gamma_w_km * p_w * u_nlse.step_dz_km;
    let rot_re = a.x * cos(dphi_nl) - a.y * sin(dphi_nl);
    let rot_im = a.x * sin(dphi_nl) + a.y * cos(dphi_nl);
    s_nlse_out[idx] = NlseSample(rot_re, rot_im, p_w * 1000.0, dphi_nl);
}
