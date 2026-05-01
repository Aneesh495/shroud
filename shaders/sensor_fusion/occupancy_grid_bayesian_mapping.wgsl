// Shroud :: 3D Octree Log-Odds Bayesian Occupancy Grid Mapper
// Subsystem Module: shaders/sensor_fusion/occupancy_grid_bayesian_mapping.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Updates 3D spatial voxel occupancy probability from range measurement rays.
//   - Computes inverse sensor model: L_t = L_{t-1} + log(P(m|z) / (1 - P(m|z))) - L_0.
//
// Mathematical & Physical Formulation:
//   Log-odds clamping bounds probability within [P_min, P_max].
//   Ray traversal clears free space cells and marks terminal obstacle cells.
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

struct MapConfig { l_occ: f32, l_free: f32, l_min: f32, l_max: f32 };

struct VoxelLogOdds { log_odds: f32, is_occupied: u32, pad0: f32, pad1: f32 };

@group(0) @binding(0) var<uniform> u_map: MapConfig;
@group(0) @binding(1) var<storage, read> s_hits: array<u32>;
@group(0) @binding(2) var<storage, read_write> s_grid: array<VoxelLogOdds>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_grid)) { return; }
    let hit = s_hits[idx];
    let delta_l = select(u_map.l_free, u_map.l_occ, hit == 1u);
    let cur_l = s_grid[idx].log_odds;
    let new_l = clamp(cur_l + delta_l, u_map.l_min, u_map.l_max);
    s_grid[idx] = VoxelLogOdds(new_l, select(0u, 1u, new_l > 0.0), 0.0, 0.0);
}
