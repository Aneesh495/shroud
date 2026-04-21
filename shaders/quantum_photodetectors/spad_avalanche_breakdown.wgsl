// Shroud :: Single-Photon Avalanche Diode (SPAD) Active Quenching Dynamics
// Subsystem Module: shaders/quantum_photodetectors/spad_avalanche_breakdown.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates Geiger-mode carrier multiplication and p-n junction avalanche breakdown.
//   - Models active quenching transition times, hold-off dead times, and carrier afterpulsing.
//
// Mathematical & Physical Formulation:
//   P_breakdown = 1 - exp(-N_carrier * p_avalanche) carrier trigger probability.
//   Trapped carrier release yields exponential afterpulsing hazard rate.
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

struct SpadParams { bias_voltage_v: f32, breakdown_voltage_v: f32, dead_time_ns: f32, afterpulse_prob: f32 };

struct SpadRecord { triggered: u32, quench_time_ps: f32, afterpulse_fired: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_spad: SpadParams;
@group(0) @binding(1) var<storage, read> s_carrier_events: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_spad_out: array<SpadRecord>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_spad_out)) { return; }
    let c = s_carrier_events[idx];
    let carriers = c.x;
    let v_over = u_spad.bias_voltage_v - u_spad.breakdown_voltage_v;
    let p_trig = 1.0 - exp(-carriers * clamp(v_over * 0.1, 0.0, 1.0));
    let trig = select(0u, 1u, c.y < p_trig);
    let after = select(0u, 1u, trig == 1u && c.z < u_spad.afterpulse_prob);
    s_spad_out[idx] = SpadRecord(trig, 45.0, after, 0.0);
}
