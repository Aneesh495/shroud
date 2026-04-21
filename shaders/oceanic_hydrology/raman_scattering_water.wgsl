// Shroud :: Inelastic Water Molecular Vibrational Raman Scattering Simulator
// Subsystem Module: shaders/oceanic_hydrology/raman_scattering_water.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates inelastic O-H vibrational stretch Raman scattering in clear ocean water.
//   - Evaluates 3350 cm^-1 wavenumber Stokes shift converting blue photons to green wavelengths.
//
// Mathematical & Physical Formulation:
//   1 / lambda_emission = 1 / lambda_excitation - Delta_nu_raman.
//   Raman cross-section sigma_r(lambda) proportional to lambda^-4.3.
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

struct RamanConfig { ex_lambda_nm: f32, temp_c: f32, salinity_psu: f32, pad: f32 };

struct RamanOutput { em_lambda_nm: f32, cross_section: f32, phase_val: f32, intensity: f32 };

@group(0) @binding(0) var<uniform> u_rm: RamanConfig;
@group(0) @binding(1) var<storage, read> s_in_photons: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_rm_out: array<RamanOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rm_out)) { return; }
    let in_p = s_in_photons[idx];
    let nu_ex = 1e7 / max(100.0, u_rm.ex_lambda_nm);
    let nu_em = nu_ex - 3350.0;
    let em_lambda = 1e7 / max(100.0, nu_em);
    let sigma = 2.6e-30 * pow(488.0 / u_rm.ex_lambda_nm, 4.3);
    let cos_t = in_p.w;
    let phase = 0.75 * (1.0 + cos_t * cos_t) * INV_PI;
    s_rm_out[idx] = RamanOutput(em_lambda, sigma, phase, in_p.y * sigma * phase);
}
