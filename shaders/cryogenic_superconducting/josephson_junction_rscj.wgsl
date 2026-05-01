// Shroud :: Resistively and Capacitively Shunted Junction (RCSJ) Phase Dynamics
// Subsystem Module: shaders/cryogenic_superconducting/josephson_junction_rscj.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Solves non-linear second-order pendulum equation for superconducting Josephson phase phi.
//   - beta_c * d^2 phi / dt^2 + d phi / dt + sin(phi) = I / I_c.
//
// Mathematical & Physical Formulation:
//   I_super = I_c * sin(phi) DC Josephson effect.
//   V = (hbar / 2e) * d phi / dt AC Josephson voltage-frequency relation.
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

struct RscjConfig { beta_c: f32, i_ratio: f32, pad0: f32, pad1: f32 };

struct RscjStep { phi_rad: f32, dphi_dt: f32, voltage_uv: f32, is_running_state: u32 };

@group(0) @binding(0) var<uniform> u_rc: RscjConfig;
@group(0) @binding(1) var<storage, read> s_phases: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_rc_out: array<RscjStep>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rc_out)) { return; }
    let st = s_phases[idx];
    let phi = st.x;
    let dphi = st.y;
    let accel = (u_rc.i_ratio - dphi - sin(phi)) / max(1e-3, u_rc.beta_c);
    let new_dphi = dphi + accel * 0.05;
    let v_uv = new_dphi * 2.0678;
    s_rc_out[idx] = RscjStep(phi + new_dphi * 0.05, new_dphi, v_uv, select(0u, 1u, abs(new_dphi) > 0.1));
}
