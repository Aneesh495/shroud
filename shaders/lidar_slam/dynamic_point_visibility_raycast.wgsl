// Shroud :: Visibility-Based Dynamic Object Removal in Static LiDAR Maps
// Subsystem Module: shaders/lidar_slam/dynamic_point_visibility_raycast.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Identifies transient dynamic vehicles and pedestrians by comparing scan rays against voxel map.
//   * Flags stored map points pierced by free-space transmission rays from subsequent epochs.
//
// Mathematical Formulation:
//   If ray passes within solid angle cone delta_theta past map point R_map and hits R_obs > R_map + margin, R_map is dynamic.
//   Eliminates ghost trails during urban HD map generation.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct VisibilityConfig { margin_m: f32, angular_tol_rad: f32, pad0: f32, pad1: f32 };

struct MapQueryPair { r_map_m: f32, r_obs_m: f32, delta_angle_rad: f32, pad: f32 };

struct DynamicFlag { is_dynamic: u32, occlusion_depth_m: f32, confidence: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_vis: VisibilityConfig;
@group(0) @binding(1) var<storage, read> s_queries: array<MapQueryPair>;
@group(0) @binding(2) var<storage, read_write> s_flags: array<DynamicFlag>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_flags)) { return; }
    let q = s_queries[idx];
    let pierced = (q.delta_angle_rad < u_vis.angular_tol_rad) && (q.r_obs_m > q.r_map_m + u_vis.margin_m);
    let depth = max(0.0, q.r_obs_m - q.r_map_m);
    let conf = select(0.0, clamp(depth / 2.0, 0.2, 1.0), pierced);
    s_flags[idx] = DynamicFlag(select(0u, 1u, pierced), depth, conf, 0.0);
}
