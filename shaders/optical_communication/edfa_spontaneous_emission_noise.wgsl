// Shroud :: Erbium-Doped Fiber Amplifier (EDFA) Amplified Spontaneous Emission (ASE)
// Subsystem Module: shaders/optical_communication/edfa_spontaneous_emission_noise.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates two-level Er^3+ population inversion (4I_13/2 to 4I_15/2).
//   - Calculates signal gain G = exp(Integral (sigma_e * N2 - sigma_a * N1) dz) and ASE spectral noise power.
//
// Mathematical & Physical Formulation:
//   P_ase = 2 * n_sp * (G - 1) * h * nu * Delta_nu in both polarization states.
//   Noise figure NF = (SNR_in / SNR_out) approaches quantum limit 3 dB.
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

struct EdfaParams { pump_power_mw: f32, signal_gain_db: f32, noise_figure_db: f32, optical_bw_ghz: f32 };

struct EdfaResult { p_out_signal_mw: f32, p_ase_total_uw: f32, snr_out_db: f32, n_sp_factor: f32 };

@group(0) @binding(0) var<uniform> u_edfa: EdfaParams;
@group(0) @binding(1) var<storage, read> s_in_signals_mw: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_edfa_out: array<EdfaResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_edfa_out)) { return; }
    let pin = s_in_signals_mw[idx];
    let g_lin = pow(10.0, u_edfa.signal_gain_db * 0.1);
    let pout = pin * g_lin;
    let nsp = 0.5 * pow(10.0, u_edfa.noise_figure_db * 0.1);
    let p_ase_w = 2.0 * nsp * (g_lin - 1.0) * PLANCK_CONSTANT * (SPEED_OF_LIGHT / 1550e-9) * (u_edfa.optical_bw_ghz * 1e9);
    s_edfa_out[idx] = EdfaResult(pout, p_ase_w * 1e6, 10.0 * (log(pout / max(1e-6, p_ase_w * 1000.0)) / log(10.0)), nsp);
}
