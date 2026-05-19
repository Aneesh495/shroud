// Shroud :: Point-to-Line Edge Correspondence Distance and Analytical Jacobian
// Subsystem Module: shaders/lidar_slam/point_to_edge_line_residual.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes perpendicular Euclidean distance from scan edge keypoint p to reference line (a, b).
//   * Evaluates normalized residual vector and Gauss-Newton SE(3) Jacobian projection.
//
// Mathematical Formulation:
//   d_edge = || (p - a) x (p - b) || / || a - b ||
//   Unit residual normal n_edge = ((p - a) x (p - b)) x (a - b) / ||...||.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct EdgeCorrespondence { p_curr: vec4<f32>, line_a: vec4<f32>, line_b: vec4<f32> };

struct EdgeResidual { distance_m: f32, jac_trans: vec4<f32>, jac_rot: vec4<f32> };

@group(0) @binding(0) var<storage, read> s_edge_pairs: array<EdgeCorrespondence>;
@group(0) @binding(1) var<storage, read_write> s_edge_residuals: array<EdgeResidual>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_edge_residuals)) { return; }
    let pair = s_edge_pairs[idx];
    let p = pair.p_curr.xyz;
    let a = pair.line_a.xyz;
    let b = pair.line_b.xyz;
    let pa = p - a;
    let pb = p - b;
    let ab = a - b;
    let area_vec = cross(pa, pb);
    let ab_len = max(1e-4, length(ab));
    let dist = length(area_vec) / ab_len;
    let grad = normalize(cross(area_vec, ab) + vec3<f32>(1e-6, 0.0, 0.0));
    let j_rot = cross(p, grad);
    s_edge_residuals[idx] = EdgeResidual(dist, vec4<f32>(grad, 1.0), vec4<f32>(j_rot, 0.0));
}
