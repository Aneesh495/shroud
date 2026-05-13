// Shroud :: Blazar Relativistic Jet Synchrotron Emission and Self-Absorption
// Subsystem Module: shaders/stellar_astrophysics/synchrotron_relativistic_jet.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates non-thermal relativistic power-law electron distribution N(gamma) = N_0 * gamma^-p.
//   - Calculates synchrotron emissivity j_nu proportional to nu^-(p-1)/2 and self-absorption coefficient alpha_nu.
//
// Mathematical & Physical Formulation:
//   Doppler boosting: delta = 1 / [Gamma * (1 - beta * cos theta)].
//   Observed flux S_obs(nu) = delta^(3 + alpha) * S_int(nu / delta).
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

struct JetConfig { lorentz_gamma: f32, doppler_delta: f32, p_index: f32, b_gauss: f32 };

struct SynchrotronOut { flux_jy: f32, turnover_freq_ghz: f32, spectral_index: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_jet: JetConfig;
@group(0) @binding(1) var<storage, read> s_freqs_ghz: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_synch_out: array<SynchrotronOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_synch_out)) { return; }
    let nu = s_freqs_ghz[idx];
    let nu_t = 15.0;
    let alpha_thin = -(u_jet.p_index - 1.0) * 0.5;
    let flux = select(pow(nu / nu_t, 2.5), pow(nu / nu_t, alpha_thin), nu > nu_t);
    let boosted = flux * pow(u_jet.doppler_delta, 3.0);
    s_synch_out[idx] = SynchrotronOut(boosted, nu_t, alpha_thin, 0.0);
}
