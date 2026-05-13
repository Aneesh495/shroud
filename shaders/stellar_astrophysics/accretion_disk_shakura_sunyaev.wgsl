// Shroud :: Shakura-Sunyaev Geometrically Thin Optically Thick Accretion Disk
// Subsystem Module: shaders/stellar_astrophysics/accretion_disk_shakura_sunyaev.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates radial effective temperature profile: T_eff(r) = [3 * G * M * M_dot / (8*pi * sigma * r^3) * (1 - sqrt(R_in / r))]^0.25.
//   - Viscous dissipation generates multi-color blackbody spectral energy distribution (SED).
//
// Mathematical & Physical Formulation:
//   Accretion alpha-prescription: Kinematic viscosity nu = alpha * c_s * H.
//   Peak emission in soft X-rays for stellar-mass black holes and UV for supermassive AGN.
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

struct DiskConfig { m_blackhole_msun: f32, mdot_eddington: f32, r_isco_rg: f32, pad: f32 };

struct DiskAnnulus { r_rg: f32, teff_k: f32, flux_w_m2: f32, orbital_vel_c: f32 };

@group(0) @binding(0) var<uniform> u_disk: DiskConfig;
@group(0) @binding(1) var<storage, read> s_radii_rg: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_annuli: array<DiskAnnulus>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_annuli)) { return; }
    let r = s_radii_rg[idx];
    let r_in = u_disk.r_isco_rg;
    let fac = max(0.0, 1.0 - sqrt(r_in / r));
    let t_peak = 1.0e7 * pow(u_disk.mdot_eddington / max(1.0, u_disk.m_blackhole_msun), 0.25);
    let t_eff = t_peak * pow(r, -0.75) * pow(fac, 0.25);
    let flux = STEFAN_BOLTZMANN * pow(t_eff, 4.0);
    let v_c = 1.0 / sqrt(2.0 * r);
    s_annuli[idx] = DiskAnnulus(r, t_eff, flux, v_c);
}
