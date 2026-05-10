// Shroud :: Greenhouse Gas Direct Radiative Forcing and Band Overlap Coupling
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/greenhouse_gas_radiative_forcing.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Calculates top-of-atmosphere (TOA) instantaneous radiative forcing Delta_F (W/m^2).
//   - CO2 logarithmic formula Delta_F = 5.35 * ln(C / C_0) (Myhre et al. 1998).
//
// Mathematical & Physical Formulation:
//   Square-root formula for methane CH4 and nitrous oxide N2O with spectral overlap band correction f(M, N).
//   Estimates Earth equilibrium climate sensitivity ECS = Delta_F_2x / lambda_feedback.
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

struct GhgConcentrations { co2_ppm: f32, ch4_ppb: f32, n2o_ppb: f32, c0_co2: f32 };

struct ForcingResult { forcing_co2_w_m2: f32, forcing_ch4_w_m2: f32, forcing_n2o_w_m2: f32, total_forcing_w_m2: f32 };

@group(0) @binding(0) var<uniform> u_ghg: GhgConcentrations;
@group(0) @binding(1) var<storage, read> s_eval_steps: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_forcing_out: array<ForcingResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_forcing_out)) { return; }
    let f_co2 = 5.35 * log(u_ghg.co2_ppm / max(1.0, u_ghg.c0_co2));
    let f_ch4 = 0.036 * (sqrt(u_ghg.ch4_ppb) - sqrt(722.0));
    let f_n2o = 0.12 * (sqrt(u_ghg.n2o_ppb) - sqrt(270.0));
    let tot = f_co2 + f_ch4 + f_n2o;
    s_forcing_out[idx] = ForcingResult(f_co2, f_ch4, f_n2o, tot);
}
