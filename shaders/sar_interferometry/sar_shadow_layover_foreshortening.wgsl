// Shroud :: Terrain Radiometric Terrain Correction (RTC) Layover & Shadow Mask
// Subsystem Module: shaders/sar_interferometry/sar_shadow_layover_foreshortening.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes local radar incidence angle theta_i relative to DEM terrain surface normal.
//   * Classifies steep foreshortening, geometric layover fold-over, and radar shadow occlusion.
//
// Mathematical Formulation:
//   cos(theta_i) = dot(look_vec, terrain_normal)
//   Shadow occurs when local grazing angle falls behind topographic slope.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct SarGeomConfig { radar_look_vec: vec4<f32>, nominal_inc_rad: f32, pad0: f32, pad1: f32, pad2: f32 };

struct TerrainMaskOut { local_inc_rad: f32, rtc_gamma_area: f32, is_layover: u32, is_shadow: u32 };

@group(0) @binding(0) var<uniform> u_geom: SarGeomConfig;
@group(0) @binding(1) var<storage, read> s_dem_normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_mask_out: array<TerrainMaskOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_mask_out)) { return; }
    let n = normalize(s_dem_normals[idx].xyz);
    let l = normalize(-u_geom.radar_look_vec.xyz);
    let cos_inc = clamp(dot(n, l), -1.0, 1.0);
    let local_inc = acos(clamp(cos_inc, 0.0, 1.0));
    let is_shad = select(0u, 1u, cos_inc <= 0.02);
    let is_lay = select(0u, 1u, local_inc < 0.08 && cos_inc > 0.02);
    let rtc_area = sin(u_geom.nominal_inc_rad) / max(0.05, abs(cos_inc));
    s_mask_out[idx] = TerrainMaskOut(local_inc, rtc_area, is_lay, is_shad);
}
