// Shroud :: Coupled-Mode Theory Asymmetric Directional Coupler Power Exchange
// Subsystem Module: shaders/photonic_integrated_circuits/directional_coupler_supermodes.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Solves coupled-mode differential equations between parallel evanescently coupled waveguides.
//   * Computes beating length L_pi = pi / (2 * kappa), crossover power, and phase mismatch Delta_beta.
//
// Mathematical Formulation:
//   P_cross(z) = (kappa^2 / gamma^2) * sin^2(gamma * z) where gamma = sqrt(kappa^2 + (Delta_beta / 2)^2)
//   Achieves 100% power transfer when propagation constants match (Delta_beta = 0).
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct CouplerConfig { kappa_um_inv: f32, delta_beta_um_inv: f32, coupling_len_um: f32, pad: f32 };

struct CouplerPowerOut { z_um: f32, bar_power: f32, cross_power: f32, beat_length_um: f32 };

@group(0) @binding(0) var<uniform> u_dc: CouplerConfig;
@group(0) @binding(1) var<storage, read> s_z_steps: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_dc_out: array<CouplerPowerOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_dc_out)) { return; }
    let z = s_z_steps[idx];
    let k = u_dc.kappa_um_inv;
    let half_db = 0.5 * u_dc.delta_beta_um_inv;
    let gamma = sqrt(k * k + half_db * half_db);
    let sin_gz = sin(gamma * z);
    let p_cross = ((k * k) / max(1e-8, gamma * gamma)) * sin_gz * sin_gz;
    let p_bar = 1.0 - p_cross;
    let l_pi = PI / max(1e-6, 2.0 * gamma);
    s_dc_out[idx] = CouplerPowerOut(z, p_bar, p_cross, l_pi);
}
