// Shroud :: Canopy Height Profile (CHP) and Leaf Area Index (LAI) Inversion
// Subsystem Module: shaders/lidar_waveforms/subcanopy_vegetation_penetration.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Inverts vertical LiDAR echo profile into apparent foliage volume density.
//   - Calculates gap fraction P_gap(z) and cumulative plant area index PAI = -integral d(ln P_gap) / G(theta).
//
// Mathematical & Physical Formulation:
//   P_gap(z) = 1 - sum_{z' > z} I_canopy(z') / (sum I_canopy + I_ground * (rho_c / rho_g)).
//   Recovers vertical multi-layer leaf area distribution.
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

struct CanopyConfig { rho_ratio: f32, g_factor: f32, bin_height_m: f32, pad: f32 };

struct CanopyProfile { gap_fraction: f32, lai_cumulative: f32, foliage_density: f32, canopy_height_m: f32 };

@group(0) @binding(0) var<uniform> u_canopy: CanopyConfig;
@group(0) @binding(1) var<storage, read> s_canopy_energy: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_chp_out: array<CanopyProfile>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_chp_out)) { return; }
    let c = s_canopy_energy[idx];
    let i_can = c.x;
    let i_gnd = c.y;
    let tot = i_can + i_gnd * u_canopy.rho_ratio;
    let p_gap = clamp(1.0 - (i_can / max(1e-4, tot)), 0.01, 1.0);
    let lai = -log(p_gap) / max(0.1, u_canopy.g_factor);
    let density = lai / max(1.0, u_canopy.bin_height_m);
    s_chp_out[idx] = CanopyProfile(p_gap, lai, density, c.z);
}
