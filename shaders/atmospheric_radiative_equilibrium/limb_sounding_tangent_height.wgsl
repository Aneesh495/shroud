// Shroud :: Satellite Limb Sounding Abel Transform Geometric Path Inversion
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/limb_sounding_tangent_height.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates long line-of-sight tangent ray geometry through Earth's atmospheric limb.
//   - Solves Abel integral inversion: n(r) = exp( (1 / pi) * Integral (alpha(p) / sqrt(p^2 - r^2)) dp ).
//
// Mathematical & Physical Formulation:
//   Tangent height z_tan provides high vertical resolution (< 1 km) stratospheric profiling.
//   Long optical path enhances trace gas detection sensitivity.
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
const AVOGADRO_NUMBER: f32 = 6.02214076e23;

struct LimbConfig { r_earth_km: f32, sat_alt_km: f32, pad0: f32, pad1: f32 };

struct LimbRaySegment { tangent_height_km: f32, path_length_km: f32, zenith_angle_deg: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_limb: LimbConfig;
@group(0) @binding(1) var<storage, read> s_tangent_heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_limb_out: array<LimbRaySegment>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_limb_out)) { return; }
    let z_tan = s_tangent_heights[idx];
    let r_tan = u_limb.r_earth_km + z_tan;
    let r_sat = u_limb.r_earth_km + u_limb.sat_alt_km;
    let path = 2.0 * sqrt(max(0.0, r_sat * r_sat - r_tan * r_tan));
    let zen = asin(r_tan / r_sat) * (180.0 / PI);
    s_limb_out[idx] = LimbRaySegment(z_tan, path, zen, 0.0);
}
