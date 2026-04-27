// Shroud :: Airborne Bathymetric Green LiDAR Water Interface Refraction
// Subsystem Module: shaders/lidar_waveforms/bathy_lidar_water_surface_refraction.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Traces 532 nm pulsed laser refraction across wavy sea surface into seabed.
//   - Applies Snell's law, subsurface speed of light c_w = c / 1.333, and optical water attenuation.
//
// Mathematical & Physical Formulation:
//   theta_w = asin(sin(theta_a) / 1.333).
//   Depth d = 0.5 * (t_bottom - t_surface) * (c / n_water) * cos(theta_w).
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

struct BathyParams { n_water: f32, diffuse_atten_k: f32, pad0: f32, pad1: f32 };

struct BathyResult { water_depth_m: f32, bottom_reflectance: f32, refracted_angle_rad: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_bathy: BathyParams;
@group(0) @binding(1) var<storage, read> s_bathy_times: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_bathy_out: array<BathyResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_bathy_out)) { return; }
    let b = s_bathy_times[idx];
    let t_surf = b.x;
    let t_bot = b.y;
    let inc_rad = b.z;
    let sin_w = sin(inc_rad) / u_bathy.n_water;
    let th_w = asin(clamp(sin_w, 0.0, 0.95));
    let dt_s = max(0.0, t_bot - t_surf) * 1e-9;
    let c_w = SPEED_OF_LIGHT / u_bathy.n_water;
    let depth = 0.5 * dt_s * c_w * cos(th_w);
    let atten = exp(-2.0 * u_bathy.diffuse_atten_k * depth);
    s_bathy_out[idx] = BathyResult(depth, b.w / max(1e-4, atten), th_w, select(0u, 1u, depth > 0.1));
}
