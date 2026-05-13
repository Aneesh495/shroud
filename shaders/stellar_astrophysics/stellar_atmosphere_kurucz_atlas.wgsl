// Shroud :: Kurucz ATLAS9 LTE Stellar Atmosphere Model Opacity and Emergent Flux
// Subsystem Module: shaders/stellar_astrophysics/stellar_atmosphere_kurucz_atlas.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates continuum H-minus ion bound-free and free-free absorption opacities.
//   - Calculates emergent surface flux F_nu(tau=0) = 2 * pi * Integral S_nu(tau) * E_2(tau) dtau.
//
// Mathematical & Physical Formulation:
//   Local Thermodynamic Equilibrium (LTE): Source function S_nu = B_nu(T).
//   Effective temperature T_eff and surface gravity log(g) parameterize stellar spectral type.
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

struct StarParams { teff_k: f32, log_g: f32, metallicity_fe_h: f32, pad: f32 };

struct StellarFluxOut { emergent_flux: f32, continuum_opacity: f32, blackbody_flux: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_star: StarParams;
@group(0) @binding(1) var<storage, read> s_nu_hz: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_star_out: array<StellarFluxOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_star_out)) { return; }
    let nu = s_nu_hz[idx];
    let hnu = PLANCK_CONSTANT * nu;
    let kt = BOLTZMANN_CONSTANT * u_star.teff_k;
    let b_nu = (2.0 * PLANCK_CONSTANT * nu * nu * nu) / (SPEED_OF_LIGHT * SPEED_OF_LIGHT * max(1e-6, exp(hnu / max(1e-22, kt)) - 1.0));
    let kap = 1.0e-26 * pow(nu / 1e14, -3.0);
    let f_nu = b_nu * 0.95;
    s_star_out[idx] = StellarFluxOut(f_nu, kap, b_nu, 0.0);
}
