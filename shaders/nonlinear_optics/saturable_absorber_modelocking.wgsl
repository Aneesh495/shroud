// Shroud :: SESAM Saturable Absorber Passive Mode-Locking Transmission
// Subsystem Module: shaders/nonlinear_optics/saturable_absorber_modelocking.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Models intensity-dependent nonlinear absorption q(F) = q_0 / (1 + F / F_sat) + q_ns.
//   * Simulates ultrashort laser pulse narrowing and Q-switched mode-locking stability criteria.
//
// Mathematical Formulation:
//   T(F) = exp(-(q_ns + q_0 / (1 + F / F_sat)))
//   Stability against Q-switching requires E_p^2 > E_sat_L * E_sat_A * Delta_R.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct SesamConfig { mod_depth_q0: f32, nonsat_loss_qns: f32, f_sat_uj_cm2: f32, pad: f32 };

struct SesamResult { pulse_fluence: f32, transmittance: f32, loss_reduction: f32, stable_cw_ml: u32 };

@group(0) @binding(0) var<uniform> u_ses: SesamConfig;
@group(0) @binding(1) var<storage, read> s_fluences_uj_cm2: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_ses_out: array<SesamResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ses_out)) { return; }
    let flu = s_fluences_uj_cm2[idx];
    let s_ratio = flu / max(1e-3, u_ses.f_sat_uj_cm2);
    let q_sat = u_ses.mod_depth_q0 / (1.0 + s_ratio);
    let total_q = u_ses.nonsat_loss_qns + q_sat;
    let trans = exp(-total_q);
    let stable = select(0u, 1u, s_ratio > 2.5);
    s_ses_out[idx] = SesamResult(flu, trans, u_ses.mod_depth_q0 - q_sat, stable);
}
