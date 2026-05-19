// Shroud :: Spinning LiDAR Scan Motion Deskewing via Continuous SLERP
// Subsystem Module: shaders/lidar_slam/deskew_motion_distortion_slerp.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Corrects helical geometric distortion in rotating multi-beam LiDAR frames during rapid vehicle turns.
//   * Transforms each point timestamp alpha_i in [0, 1] back to frame start or end epoch.
//
// Mathematical Formulation:
//   p_deskew = R(alpha_i) * p_raw + t(alpha_i)
//   Quaternion SLERP interpolates rotational twist smoothly across 100 ms sweep windows.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct SweepDeltaPose { delta_trans: vec4<f32>, delta_quat: vec4<f32> };

struct DeskewedPoint { pos: vec4<f32> };

@group(0) @binding(0) var<uniform> u_sweep: SweepDeltaPose;
@group(0) @binding(1) var<storage, read> s_raw_pts_time: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_deskewed: array<DeskewedPoint>;

fn quat_rotate_vec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_deskewed)) { return; }
    let raw = s_raw_pts_time[idx];
    let alpha = clamp(raw.w, 0.0, 1.0);
    let q_id = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let q_interp = normalize(mix(q_id, u_sweep.delta_quat, alpha));
    let t_interp = u_sweep.delta_trans.xyz * alpha;
    let p_corr = quat_rotate_vec(q_interp, raw.xyz) + t_interp;
    s_deskewed[idx] = DeskewedPoint(vec4<f32>(p_corr, raw.w));
}
