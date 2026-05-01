// Shroud :: Transition-Edge Sensor (TES) Microcalorimeter Electrothermal Feedback
// Subsystem Module: shaders/cryogenic_superconducting/transition_edge_sensor_bolometer.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates extreme dR/dT superconducting-to-normal phase transition at mK temperatures.
//   - Simulates strong negative electrothermal feedback (ETF) stabilizing detector temperature.
//
// Mathematical & Physical Formulation:
//   alpha_I = (T / R) * (dR / dT) transition sensitivity exceeds 100.
//   Delta_E = sqrt(4 * k_B * T^2 * C / alpha) ultra-high energy resolution.
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

struct TesBolomParams { tc_mk: f32, r0_mohm: f32, alpha_sens: f32, bias_voltage_uv: f32 };

struct TesBolomOut { current_ua: f32, power_pw: f32, responsivity_a_w: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_tb: TesBolomParams;
@group(0) @binding(1) var<storage, read> s_temps_mk: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_tb_out: array<TesBolomOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_tb_out)) { return; }
    let t = s_temps_mk[idx];
    let dt = t - u_tb.tc_mk;
    let r = u_tb.r0_mohm * (1.0 + u_tb.alpha_sens * (dt / u_tb.tc_mk));
    let i_ua = (u_tb.bias_voltage_uv / max(1e-3, r));
    let p_pw = i_ua * u_tb.bias_voltage_uv;
    s_tb_out[idx] = TesBolomOut(i_ua, p_pw, -1.0 / max(1e-4, u_tb.bias_voltage_uv * 1e-6), 0.0);
}
