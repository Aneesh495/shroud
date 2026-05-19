// Shroud :: Degenerate Four-Wave Mixing (FWM) Chi^(3) Parametric Gain
// Subsystem Module: shaders/nonlinear_optics/four_wave_mixing_degenerate.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Simulates phase-conjugate signal and idler wave amplification from two counter-propagating pumps.
//   * Computes phase-matching mismatch Delta_k = beta_2 * (omega_s - omega_p)^2 + 2 * gamma * P_pump.
//
// Mathematical Formulation:
//   g_fwm = sqrt((gamma * P_0)^2 - (kappa / 2)^2)
//   Amplification G = 1 + [(gamma * P_0 / g) * sinh(g * L)]^2.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct FwmConfig { gamma_w_km: f32, pump_power_w: f32, fiber_len_km: f32, beta2_ps2_km: f32 };

struct FwmResult { detuning_thz: f32, parametric_gain_db: f32, phase_mismatch: f32, conversion_ratio: f32 };

@group(0) @binding(0) var<uniform> u_fwm: FwmConfig;
@group(0) @binding(1) var<storage, read> s_detunings_thz: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_fwm_out: array<FwmResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fwm_out)) { return; }
    let df_thz = s_detunings_thz[idx];
    let dw = TWO_PI * df_thz;
    let gp = u_fwm.gamma_w_km * u_fwm.pump_power_w;
    let kappa = u_fwm.beta2_ps2_km * dw * dw + 2.0 * gp;
    let arg = gp * gp - 0.25 * kappa * kappa;
    let g = sqrt(max(0.0, arg));
    let gl = min(10.0, g * u_fwm.fiber_len_km);
    let gain_lin = 1.0 + pow((gp / max(1e-4, g)) * sinh(gl), 2.0);
    let gain_db = 10.0 * (log(max(1.0, gain_lin)) / log(10.0));
    s_fwm_out[idx] = FwmResult(df_thz, gain_db, kappa, gain_lin - 1.0);
}
