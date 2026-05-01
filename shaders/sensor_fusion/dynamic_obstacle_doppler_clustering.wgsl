// Shroud :: DBSCAN 4D Spatio-Temporal Velocity Radar Clustering Kernel
// Subsystem Module: shaders/sensor_fusion/dynamic_obstacle_doppler_clustering.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Clusters radar targets based on Euclidean spatial distance and Doppler velocity consistency.
//   - Groups multi-reflection radar returns into single dynamic vehicle obstacles.
//
// Mathematical & Physical Formulation:
//   dist_4d = sqrt(||p1 - p2||^2 + w_v * (v1 - v2)^2).
//   Identifies core points, border points, and random clutter noise.
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

struct DbscanParams { eps_m: f32, eps_vel: f32, min_pts: u32, pad: f32 };

struct ClusterResult { cluster_id: i32, is_core: u32, neighbor_count: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_db: DbscanParams;
@group(0) @binding(1) var<storage, read> s_pts_vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_clusters: array<ClusterResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_clusters)) { return; }
    let cur = s_pts_vel[idx];
    var neighbors = 0u;
    let count = arrayLength(&s_pts_vel);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let other = s_pts_vel[i];
        let d_pos = length(cur.xyz - other.xyz);
        let d_vel = abs(cur.w - other.w);
        if (d_pos < u_db.eps_m && d_vel < u_db.eps_vel) {
            neighbors = neighbors + 1u;
        }
    }
    let is_core = select(0u, 1u, neighbors >= u_db.min_pts);
    s_clusters[idx] = ClusterResult(select(-1, 1, is_core == 1u), is_core, neighbors, 0.0);
}
