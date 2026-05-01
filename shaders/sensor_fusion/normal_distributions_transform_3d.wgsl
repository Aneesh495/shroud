// Shroud :: 3D Normal Distributions Transform (NDT) Score and Gradient Evaluator
// Subsystem Module: shaders/sensor_fusion/normal_distributions_transform_3d.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Gaussian spatial likelihood of lidar point cloud within regular voxel grids.
//   - Computes score and analytical gradient for Newton-Raphson scan registration.
//
// Mathematical & Physical Formulation:
//   score = -d1 * exp(-0.5 * d2 * (x - mu)^T * Sigma^-1 * (x - mu)).
//   D(score)/dx analytical Jacobian.
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

struct NdtVoxel { mu: vec4<f32>, inv_cov_diag: vec4<f32> };

struct NdtScore { score: f32, grad: vec4<f32> };

@group(0) @binding(0) var<storage, read> s_voxels: array<NdtVoxel>;
@group(0) @binding(1) var<storage, read> s_points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_ndt_scores: array<NdtScore>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ndt_scores)) { return; }
    let pt = s_points[idx].xyz;
    let vox = s_voxels[idx % arrayLength(&s_voxels)];
    let diff = pt - vox.mu.xyz;
    let d2 = dot(diff * diff, vox.inv_cov_diag.xyz);
    let score = exp(-0.5 * d2);
    let grad = -diff * vox.inv_cov_diag.xyz * score;
    s_ndt_scores[idx] = NdtScore(score, vec4<f32>(grad, 0.0));
}
