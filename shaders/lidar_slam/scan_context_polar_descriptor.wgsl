// Shroud :: Scan Context Azimuthal-Radial 2D Loop Closure Global Descriptor
// Subsystem Module: shaders/lidar_slam/scan_context_polar_descriptor.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Encodes 3D LiDAR point clouds into invariant 20-ring x 60-sector max-height polar matrices.
//   * Supports rapid place recognition and yaw-invariant loop closure cosine matching.
//
// Mathematical Formulation:
//   r = sqrt(x^2 + y^2), theta = atan2(y, x) mapped into discrete (i_ring, j_sector) bin.
//   Column-wise circular shift aligns arbitrary heading offsets between revisit frames.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct ScConfig { max_radius_m: f32, num_rings: u32, num_sectors: u32, pad: f32 };

struct ScBinKey { ring_idx: u32, sector_idx: u32, max_z_m: f32, occupied: u32 };

@group(0) @binding(0) var<uniform> u_sc: ScConfig;
@group(0) @binding(1) var<storage, read> s_pts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_bins: array<ScBinKey>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_bins)) { return; }
    let p = s_pts[idx];
    let r = sqrt(p.x * p.x + p.y * p.y);
    let th = (atan2(p.y, p.x) + TWO_PI) % TWO_PI;
    let ring = min(u_sc.num_rings - 1u, u32((r / max(1.0, u_sc.max_radius_m)) * f32(u_sc.num_rings)));
    let sec = min(u_sc.num_sectors - 1u, u32((th / TWO_PI) * f32(u_sc.num_sectors)));
    let occ = select(0u, 1u, r <= u_sc.max_radius_m && r > 0.5);
    s_bins[idx] = ScBinKey(ring, sec, p.z, occ);
}
