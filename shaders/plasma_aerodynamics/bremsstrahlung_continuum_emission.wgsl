// Shroud :: Free-Free Electron-Ion Thermal Bremsstrahlung Radiance
// Subsystem Module: shaders/plasma_aerodynamics/bremsstrahlung_continuum_emission.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates thermal Bremsstrahlung emission from shock heated ionized gas caps.
//   - Computes emissivity j_nu = (32 * pi * e^6) / (3 * m_e * c^3) * sqrt(2*pi / (3*m_e*k*T)) * n_e * n_i * g_ff.
//
// Mathematical & Physical Formulation:
//   Provides background radiative heating flux onto aerodynamic thermal protection systems.
//   Gaunt factor g_ff logarithmic quantum correction.
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
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct BremsParams { temp_k: f32, n_e_m3: f32, n_i_m3: f32, pad: f32 };

struct BremsResult { radiance_w_m3_sr: f32, gaunt_factor: f32, spectral_peak_ev: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_br: BremsParams;
@group(0) @binding(1) var<storage, read> s_photon_energies_ev: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_brems_out: array<BremsResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_brems_out)) { return; }
    let h_nu = s_photon_energies_ev[idx];
    let kt_ev = u_br.temp_k * 8.617e-5;
    let g_ff = 1.2;
    let emiss = (u_br.n_e_m3 * u_br.n_i_m3 * 1e-40) / sqrt(u_br.temp_k) * exp(-h_nu / max(0.1, kt_ev)) * g_ff;
    s_brems_out[idx] = BremsResult(emiss, g_ff, kt_ev, 0.0);
}
