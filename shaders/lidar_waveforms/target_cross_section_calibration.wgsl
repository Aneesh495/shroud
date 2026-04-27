// Shroud :: Differential Backscattering Cross Section and Target Reflectance
// Subsystem Module: shaders/lidar_waveforms/target_cross_section_calibration.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Converts returned echo power integrals into physical cross section sigma (m^2).
//   - Solves radar equation: P_rx = (P_tx * D_rx^2) / (4 * pi * R^4 * beta_tx^2) * eta_sys * eta_atm^2 * sigma.
//
// Mathematical & Physical Formulation:
//   sigma = (4 * pi * R^4 / C_cal) * Integral P_rx(t) dt.
//   Estimates target surface Lambertian diffuse reflectance rho = sigma / (4 * A_footprint).
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

struct CalibConstants { c_cal: f32, beam_div_rad: f32, eta_atm: f32, pad: f32 };

struct CrossSectionResult { sigma_m2: f32, reflectance: f32, range_m: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_cal: CalibConstants;
@group(0) @binding(1) var<storage, read> s_echo_energies: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_sigma_out: array<CrossSectionResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sigma_out)) { return; }
    let e = s_echo_energies[idx];
    let energy = e.x;
    let r = e.y;
    let r4 = r * r * r * r;
    let sigma = (FOUR_PI * r4 * energy) / max(1e-3, u_cal.c_cal * u_cal.eta_atm * u_cal.eta_atm);
    let a_fp = PI * pow(r * u_cal.beam_div_rad * 0.5, 2.0);
    let rho = clamp(sigma / max(1e-4, 4.0 * a_fp), 0.0, 1.0);
    s_sigma_out[idx] = CrossSectionResult(sigma, rho, r, 1u);
}
