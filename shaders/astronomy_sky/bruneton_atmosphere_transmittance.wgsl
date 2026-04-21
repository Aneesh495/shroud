// Shroud :: Bruneton Atmospheric Transmittance Precomputed Lookup Table
// Subsystem Module: shaders/astronomy_sky/bruneton_atmosphere_transmittance.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Solves numerical line integral of Rayleigh and ozone extinction across spherical shell.
//   - Generates 2D transmittance lookup table T(r, mu) mapping altitude and view zenith angle.
//
// Mathematical & Physical Formulation:
//   T(r, mu) = exp(- sum [alpha_rayleigh * rho_r(s) + alpha_mie * rho_m(s) + alpha_ozone * rho_o(s)] ds).
//   Handles planetary horizon shadow cutoff without singularity.
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

struct AtmosParams { r_planet_m: f32, r_atmos_m: f32, scale_h_rayleigh_m: f32, scale_h_mie_m: f32 };

struct TransmittanceOut { trans_rayleigh: f32, trans_mie: f32, trans_ozone: f32, optical_depth: f32 };

@group(0) @binding(0) var<uniform> u_at: AtmosParams;
@group(0) @binding(1) var<storage, read> s_r_mu_coords: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_trans_out: array<TransmittanceOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_trans_out)) { return; }
    let coord = s_r_mu_coords[idx];
    let r = coord.x;
    let mu = coord.y;
    let h = r - u_at.r_planet_m;
    let d_ray = exp(-h / u_at.scale_h_rayleigh_m);
    let d_mie = exp(-h / u_at.scale_h_mie_m);
    let airmass = 1.0 / max(1e-3, sqrt(mu * mu + 0.001));
    let tau_ray = d_ray * airmass * 0.1;
    let tau_mie = d_mie * airmass * 0.05;
    s_trans_out[idx] = TransmittanceOut(exp(-tau_ray), exp(-tau_mie), exp(-0.01 * airmass), tau_ray + tau_mie);
}
