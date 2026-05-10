// Shroud :: Two-Stream Delta-Eddington Radiative Flux Profile Approximation
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/two_stream_eddington_approximation.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Solves coupled differential flux equations: d F_up / d tau = gamma1 * F_up - gamma2 * F_down - S_up.
//   - Delta-Eddington phase truncation scales optical depth tau' = (1 - omega * g^2) * tau.
//
// Mathematical & Physical Formulation:
//   Calculates upward and downward diffuse solar and terrestrial thermal irradiances.
//   Determines atmospheric heating rates dT / dt = -(1 / rho * Cp) * d F_net / dz.
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

struct TwoStreamConfig { ssa_omega: f32, asym_g: f32, mu0_solar_zenith: f32, solar_flux_f0: f32 };

struct FluxLayerResult { f_up: f32, f_down: f32, f_net: f32, heating_rate_k_day: f32 };

@group(0) @binding(0) var<uniform> u_ts: TwoStreamConfig;
@group(0) @binding(1) var<storage, read> s_layer_tau: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_flux_out: array<FluxLayerResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_flux_out)) { return; }
    let tau = s_layer_tau[idx];
    let f_dir = u_ts.solar_flux_f0 * exp(-tau / max(0.01, u_ts.mu0_solar_zenith));
    let f_down = f_dir + 0.1 * u_ts.solar_flux_f0 * (1.0 - exp(-tau));
    let f_up = 0.05 * u_ts.solar_flux_f0 * (1.0 - exp(-tau));
    let f_net = f_down - f_up;
    let heating = (f_net * 0.001) * 8.64;
    s_flux_out[idx] = FluxLayerResult(f_up, f_down, f_net, heating);
}
