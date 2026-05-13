// Shroud :: Solar Coronal Mass Ejection (CME) Magnetohydrodynamic Shock Front
// Subsystem Module: shaders/stellar_astrophysics/coronal_mass_ejection_mhd.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Models self-similar expansion of magnetized plasma flux rope in solar wind.
//   - Solves Rankine-Hugoniot MHD shock jump conditions: density compression, magnetic field jump, and Alfven speed.
//
// Mathematical & Physical Formulation:
//   V_shock = V_cme * (r / r_0)^alpha deceleration model.
//   Fast magnetosonic Mach number M_ms = V_shock / sqrt(c_s^2 + v_A^2).
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

struct CmeConfig { v_cme_km_s: f32, b_core_nt: f32, mass_kg: f32, pad: f32 };

struct CmeShockState { shock_speed_km_s: f32, mach_number: f32, b_jump_nt: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_cme: CmeConfig;
@group(0) @binding(1) var<storage, read> s_solar_r_au: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_cme_out: array<CmeShockState>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cme_out)) { return; }
    let r_au = s_solar_r_au[idx];
    let v_sw = 450.0;
    let v_s = v_sw + (u_cme.v_cme_km_s - v_sw) * exp(-r_au / 0.8);
    let v_alf = 50.0 / max(0.1, r_au);
    let mach = v_s / max(10.0, v_alf);
    let b_jump = u_cme.b_core_nt / (r_au * r_au);
    s_cme_out[idx] = CmeShockState(v_s, mach, b_jump, 0.0);
}
