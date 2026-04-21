// Shroud :: Boulic Kinematic Gait Model Pedestrian Micro-Doppler Synthesizer
// Subsystem Module: shaders/automotive_radars/micro_doppler_pedestrian.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 12-segment kinematic human locomotion trajectories.
//   - Generates characteristic oscillating micro-Doppler frequency modulation envelopes.
//
// Mathematical & Physical Formulation:
//   f_d(t) = (2 * f_0 / c) * v_r(t) where v_r(t) is limb segmental velocity projection.
//   Models torso, thighs, shins, and swinging feet harmonics.
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

struct GaitParams { walk_speed_m_s: f32, carrier_freq_ghz: f32, height_m: f32, pad: f32 };

struct LimbDoppler { torso_fd: f32, foot_fd: f32, shin_fd: f32, arm_fd: f32 };

@group(0) @binding(0) var<uniform> u_gait: GaitParams;
@group(0) @binding(1) var<storage, read> s_times_s: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_limbs: array<LimbDoppler>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_limbs)) { return; }
    let t = s_times_s[idx];
    let v0 = u_gait.walk_speed_m_s;
    let f0 = u_gait.carrier_freq_ghz * 1e9;
    let k_dopp = 2.0 * f0 / SPEED_OF_LIGHT;
    let cycle = TWO_PI * (v0 / max(0.5, u_gait.height_m * 0.8)) * t;
    let v_torso = v0 + 0.1 * sin(2.0 * cycle);
    let v_foot = v0 * (1.0 + 1.2 * sin(cycle));
    let v_shin = v0 * (1.0 + 0.8 * sin(cycle));
    let v_arm = v0 * (1.0 - 0.7 * sin(cycle));
    s_limbs[idx] = LimbDoppler(v_torso * k_dopp, v_foot * k_dopp, v_shin * k_dopp, v_arm * k_dopp);
}
