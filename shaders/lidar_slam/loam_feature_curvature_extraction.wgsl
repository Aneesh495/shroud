// Shroud :: LOAM Smoothness Curvature Edge and Planar Feature Classifier
// Subsystem Module: shaders/lidar_slam/loam_feature_curvature_extraction.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes local ring neighborhood smoothness metric c across spinning LiDAR scanlines.
//   * Separates sharp geometric edge corners from locally flat planar road and wall facets.
//
// Mathematical Formulation:
//   c = || sum_{j in S, j != i} (X_i - X_j) || / (|S| * ||X_i||)
//   Edge features satisfy c > c_edge; planar features satisfy c < c_plane.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct LoamParams { edge_thresh: f32, plane_thresh: f32, half_window: u32, pad: f32 };

struct FeatureLabel { smoothness: f32, feature_type: u32, range_m: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_loam: LoamParams;
@group(0) @binding(1) var<storage, read> s_ring_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_features_out: array<FeatureLabel>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = arrayLength(&s_ring_points);
    if (idx >= n) { return; }
    let p_center = s_ring_points[idx].xyz;
    let r_center = length(p_center);
    let hw = u_loam.half_window;
    if (idx < hw || idx + hw >= n || r_center < 0.5) {
        s_features_out[idx] = FeatureLabel(0.0, 0u, r_center, 0u);
        return;
    }
    var diff_sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var j: u32 = 1u; j <= hw; j = j + 1u) {
        diff_sum = diff_sum + (p_center - s_ring_points[idx - j].xyz);
        diff_sum = diff_sum + (p_center - s_ring_points[idx + j].xyz);
    }
    let c = length(diff_sum) / (f32(2u * hw) * max(0.5, r_center));
    let f_type = select(select(0u, 1u, c < u_loam.plane_thresh), 2u, c > u_loam.edge_thresh);
    s_features_out[idx] = FeatureLabel(c, f_type, r_center, 1u);
}
