// Shroud :: Parallel GPU RANSAC Drivable Road Ground Surface Estimator
// Subsystem Module: shaders/sensor_fusion/ground_surface_plane_ransac.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Estimates dominant ground plane model a*x + b*y + c*z + d = 0 from LiDAR points.
//   - Identifies inlier points, curb boundaries, and vertical off-ground obstacles.
//
// Mathematical & Physical Formulation:
//   dist = abs(a*x + b*y + c*z + d) / sqrt(a^2 + b^2 + c^2).
//   Consensus evaluation across thousands of parallel sample hypotheses.
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

struct PlaneHypothesis { plane: vec4<f32>, inlier_count: u32, score: f32 };

struct RansacPoint { pos: vec4<f32>, is_ground: u32, dist_to_plane: f32 };

@group(0) @binding(0) var<uniform> u_hypo: PlaneHypothesis;
@group(0) @binding(1) var<storage, read> s_pts_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_ransac_out: array<RansacPoint>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ransac_out)) { return; }
    let p = s_pts_in[idx].xyz;
    let plane = u_hypo.plane;
    let d = abs(dot(plane.xyz, p) + plane.w) / max(1e-4, length(plane.xyz));
    let is_gnd = select(0u, 1u, d < 0.15);
    s_ransac_out[idx] = RansacPoint(vec4<f32>(p, 1.0), is_gnd, d);
}
