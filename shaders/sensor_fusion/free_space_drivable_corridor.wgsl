// Shroud :: Polar Radial Ray Boundary Search for Drivable Free Space
// Subsystem Module: shaders/sensor_fusion/free_space_drivable_corridor.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates angular sector radial distances to closest obstacle boundary.
//   - Outputs 360-degree polar polygon of safe navigable driving corridor.
//
// Mathematical & Physical Formulation:
//   r(theta) = min_i { r_i | angle(p_i) in [theta - d_th/2, theta + d_th/2] and z_i > z_ground }.
//   Smooths radial contours for trajectory motion planning.
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

struct CorridorConfig { sector_count: u32, max_range_m: f32, pad0: f32, pad1: f32 };

struct SectorDistance { angle_rad: f32, free_range_m: f32, has_obstacle: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_corr: CorridorConfig;
@group(0) @binding(1) var<storage, read> s_sector_min_r: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_corridor_out: array<SectorDistance>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_corridor_out)) { return; }
    let min_r = s_sector_min_r[idx];
    let th = (f32(idx) / f32(u_corr.sector_count)) * TWO_PI;
    let clamped_r = min(min_r, u_corr.max_range_m);
    s_corridor_out[idx] = SectorDistance(th, clamped_r, select(0u, 1u, clamped_r < u_corr.max_range_m), 0.0);
}
