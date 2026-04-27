// Shroud :: ADCP 4-Beam Janus Configuration Ocean Current Velocity Profiler
// Subsystem Module: shaders/acoustic_sonar/acoustic_doppler_current_profiler.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 4-beam Janus geometry to measure 3D ocean water current velocity vectors [u, v, w].
//   - Inverts Doppler frequency shifts f_d = (2 * f_0 / c) * v_radial along tilted transducer beams.
//
// Mathematical & Physical Formulation:
//   u = (v1 - v2) / (2 * sin(theta_beam)), v = (v3 - v4) / (2 * sin(theta_beam)), w = (v1 + v2 + v3 + v4) / (4 * cos(theta_beam)).
//   Error velocity e = (v1 + v2) - (v3 + v4) verifies flow homogeneity.
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

struct AdcpConfig { beam_angle_rad: f32, f0_hz: f32, c_sound: f32, pad: f32 };

struct VelocityVector { u_east_m_s: f32, v_north_m_s: f32, w_up_m_s: f32, error_velocity_m_s: f32 };

@group(0) @binding(0) var<uniform> u_adcp: AdcpConfig;
@group(0) @binding(1) var<storage, read> s_beam_doppler: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_currents: array<VelocityVector>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_currents)) { return; }
    let fd = s_beam_doppler[idx];
    let k_v = u_adcp.c_sound / (2.0 * u_adcp.f0_hz);
    let v1 = fd.x * k_v;
    let v2 = fd.y * k_v;
    let v3 = fd.z * k_v;
    let v4 = fd.w * k_v;
    let sin_b = sin(u_adcp.beam_angle_rad);
    let cos_b = cos(u_adcp.beam_angle_rad);
    let u = (v1 - v2) / (2.0 * sin_b);
    let v = (v3 - v4) / (2.0 * sin_b);
    let w = (v1 + v2 + v3 + v4) / (4.0 * cos_b);
    let err_v = (v1 + v2) - (v3 + v4);
    s_currents[idx] = VelocityVector(u, v, w, err_v);
}
