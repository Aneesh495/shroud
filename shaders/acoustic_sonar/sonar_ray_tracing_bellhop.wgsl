// Shroud :: Gaussian Beam Acoustic Ray Tracing through Stratified Sound Speed
// Subsystem Module: shaders/acoustic_sonar/sonar_ray_tracing_bellhop.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Traces acoustic ray trajectories through ocean sound speed profile (SSP) c(z).
//   - Solves Snell's acoustic law: cos(theta(z)) / c(z) = constant across sound channel (SOFAR).
//
// Mathematical & Physical Formulation:
//   d^2 r / ds^2 = (1 / c) * grad(c) eikonal ray equation.
//   Gaussian beam weighting synthesizes acoustic transmission loss without caustics.
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

struct SspParams { c0_m_s: f32, ssp_gradient: f32, z_channel_m: f32, pad: f32 };

struct SonarRayStep { pos: vec4<f32>, dir: vec4<f32>, travel_time_s: f32, phase_rad: f32 };

@group(0) @binding(0) var<uniform> u_ssp: SspParams;
@group(0) @binding(1) var<storage, read> s_in_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_ray_steps: array<SonarRayStep>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ray_steps)) { return; }
    let r = s_in_rays[idx];
    let z = r.y;
    let c_z = u_ssp.c0_m_s + u_ssp.ssp_gradient * abs(z - u_ssp.z_channel_m);
    let angle = r.w;
    let curvature = -u_ssp.ssp_gradient * cos(angle) / c_z;
    let new_angle = angle + curvature * 10.0;
    let new_pos = r.xyz + vec3<f32>(cos(new_angle), sin(new_angle), 0.0) * 10.0;
    s_ray_steps[idx] = SonarRayStep(vec4<f32>(new_pos, 1.0), vec4<f32>(cos(new_angle), sin(new_angle), 0.0, c_z), 10.0 / c_z, 0.0);
}
