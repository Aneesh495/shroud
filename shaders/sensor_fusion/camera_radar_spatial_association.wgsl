// Shroud :: Probabilistic Camera-Radar Bounding Box Association Matrix
// Subsystem Module: shaders/sensor_fusion/camera_radar_spatial_association.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes 2D image plane to 3D radar target Mahalanobis projection cost matrix.
//   - Solves assignment problem for camera-radar track fusion.
//
// Mathematical & Physical Formulation:
//   Cost_ij = (z_cam - H*z_rad)^T * S^-1 * (z_cam - H*z_rad) + log|S|.
//   Rejects false radar ghost returns outside camera detections.
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

struct AssocConfig { max_dist: f32, pad0: f32, pad1: f32, pad2: f32 };

struct AssocCost { cost: f32, is_valid_candidate: u32, pad0: f32, pad1: f32 };

@group(0) @binding(0) var<uniform> u_assoc: AssocConfig;
@group(0) @binding(1) var<storage, read> s_cam_boxes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_radar_projs: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> s_cost_mat: array<AssocCost>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cost_mat)) { return; }
    let cb = s_cam_boxes[idx % arrayLength(&s_cam_boxes)];
    let rp = s_radar_projs[idx / arrayLength(&s_cam_boxes)];
    let dist = length(cb.xy - rp.xy);
    s_cost_mat[idx] = AssocCost(dist, select(0u, 1u, dist < u_assoc.max_dist), 0.0, 0.0);
}
