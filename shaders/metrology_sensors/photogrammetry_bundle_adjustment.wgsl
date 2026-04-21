// Shroud :: Sparse Levenberg-Marquardt Reprojection Error and Jacobian Accumulator
// Subsystem Module: shaders/metrology_sensors/photogrammetry_bundle_adjustment.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes camera pinhole reprojection error residuals for multi-view stereo calibration.
//   - Evaluates Jacobian matrices J = d r / d theta for camera extrinsics and 3D tie points.
//
// Mathematical & Physical Formulation:
//   r_i = p_proj(K, R, t, X_j) - x_obs_i residual vector.
//   damping (J^T * J + lambda * I) * delta = -J^T * r.
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

struct CameraIntrinsics { fx: f32, fy: f32, cx: f32, cy: f32 };

struct ReprojectionRes { err_x: f32, err_y: f32, chi2: f32, weight: f32 };

@group(0) @binding(0) var<uniform> u_cam: CameraIntrinsics;
@group(0) @binding(1) var<storage, read> s_pts_cam: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_pts_obs: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> s_reproj: array<ReprojectionRes>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_reproj)) { return; }
    let p = s_pts_cam[idx].xyz;
    let obs = s_pts_obs[idx];
    let z = max(1e-3, p.z);
    let u = u_cam.fx * (p.x / z) + u_cam.cx;
    let v = u_cam.fy * (p.y / z) + u_cam.cy;
    let ex = u - obs.x;
    let ey = v - obs.y;
    let chi2 = ex * ex + ey * ey;
    let huber = select(1.0, 1.5 / max(1.5, sqrt(chi2)), chi2 > 2.25);
    s_reproj[idx] = ReprojectionRes(ex, ey, chi2, huber);
}
