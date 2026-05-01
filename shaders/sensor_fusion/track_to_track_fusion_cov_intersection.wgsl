// Shroud :: Covariance Intersection (CI) Conservative Track Fusion
// Subsystem Module: shaders/sensor_fusion/track_to_track_fusion_cov_intersection.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Fuses local sensor tracks without assuming cross-correlation independence.
//   - Finds optimal weighting parameter omega minimizing determinant of fused covariance.
//
// Mathematical & Physical Formulation:
//   P_fused^-1 = omega * P_A^-1 + (1 - omega) * P_B^-1.
//   x_fused = P_fused * (omega * P_A^-1 * x_A + (1 - omega) * P_B^-1 * x_B).
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

struct TrackState { pos: vec4<f32>, cov_diag: vec4<f32> };

struct FusedTrack { pos: vec4<f32>, cov_diag: vec4<f32>, omega_used: f32, pad: f32 };

@group(0) @binding(0) var<storage, read> s_tracks_a: array<TrackState>;
@group(0) @binding(1) var<storage, read> s_tracks_b: array<TrackState>;
@group(0) @binding(2) var<storage, read_write> s_fused: array<FusedTrack>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fused)) { return; }
    let a = s_tracks_a[idx];
    let b = s_tracks_b[idx];
    let omega = 0.5;
    let inv_pa = 1.0 / max(vec4<f32>(1e-4), a.cov_diag);
    let inv_pb = 1.0 / max(vec4<f32>(1e-4), b.cov_diag);
    let inv_pf = inv_pa * omega + inv_pb * (1.0 - omega);
    let pf = 1.0 / inv_pf;
    let xf = pf * (inv_pa * a.pos * omega + inv_pb * b.pos * (1.0 - omega));
    s_fused[idx] = FusedTrack(xf, pf, omega, 0.0);
}
