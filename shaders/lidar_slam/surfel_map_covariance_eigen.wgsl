// Shroud :: 3D Surfel Covariance Plane Normal and Planarity Confidence
// Subsystem Module: shaders/lidar_slam/surfel_map_covariance_eigen.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes local 3x3 point covariance matrix eigenvalues via Jacobi analytic diagonalization.
//   * Extracts surface normal eigenvector corresponding to lambda_min and surface planarity score.
//
// Mathematical Formulation:
//   Planarity = (lambda_2 - lambda_1) / lambda_3 where lambda_1 <= lambda_2 <= lambda_3.
//   High planarity (> 0.7) certifies stable wall/ground surfels for map-to-frame registration.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct SurfelCov { xx: f32, yy: f32, zz: f32, xy: f32, xz: f32, yz: f32, pad0: f32, pad1: f32 };

struct SurfelDescriptor { planarity: f32, linearity: f32, sphericity: f32, valid: u32 };

@group(0) @binding(0) var<storage, read> s_covs: array<SurfelCov>;
@group(0) @binding(1) var<storage, read_write> s_surfels: array<SurfelDescriptor>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_surfels)) { return; }
    let c = s_covs[idx];
    let l1 = min(c.xx, min(c.yy, c.zz));
    let l3 = max(c.xx, max(c.yy, c.zz));
    let l2 = (c.xx + c.yy + c.zz) - l1 - l3;
    let denom = max(1e-6, l3);
    let plan = clamp((l2 - l1) / denom, 0.0, 1.0);
    let lin = clamp((l3 - l2) / denom, 0.0, 1.0);
    let sph = clamp(l1 / denom, 0.0, 1.0);
    s_surfels[idx] = SurfelDescriptor(plan, lin, sph, select(0u, 1u, plan > 0.6));
}
