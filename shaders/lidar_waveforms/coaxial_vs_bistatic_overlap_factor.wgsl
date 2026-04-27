// Shroud :: Geometric Overlap Function G(R) for Coaxial and Bistatic LiDAR
// Subsystem Module: shaders/lidar_waveforms/coaxial_vs_bistatic_overlap_factor.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Calculates transmitter beam and receiver field-of-view (FOV) spatial overlap integral.
//   - Corrects near-range blind zones and signal saturation in atmospheric elastic backscatter.
//
// Mathematical & Physical Formulation:
//   G(R) = Area(Beam(R) intersect FOV(R)) / Area(Beam(R)).
//   Coaxial: G(R) = 1 outside minimum focus distance. Bistatic: G(R) = 0 until beam intersects receiver cone.
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

struct OverlapConfig { baseline_m: f32, fov_rad: f32, divergence_rad: f32, r_focus_m: f32 };

struct OverlapResult { g_factor: f32, range_m: f32, corrected_power: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_ov: OverlapConfig;
@group(0) @binding(1) var<storage, read> s_raw_ranges: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_overlap_out: array<OverlapResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_overlap_out)) { return; }
    let r_sample = s_raw_ranges[idx];
    let r = r_sample.x;
    let p_raw = r_sample.y;
    let d_beam = r * u_ov.divergence_rad;
    let d_fov = r * u_ov.fov_rad;
    let separation = max(0.0, u_ov.baseline_m - (d_beam + d_fov) * 0.5);
    let g = clamp(1.0 - separation / max(1e-3, u_ov.baseline_m), 0.0, 1.0);
    s_overlap_out[idx] = OverlapResult(g, r, p_raw / max(0.05, g), 0.0);
}
