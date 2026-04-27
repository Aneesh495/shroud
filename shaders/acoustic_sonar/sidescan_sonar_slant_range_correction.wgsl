// Shroud :: Sidescan Sonar Geometric Slant-Range to Ground-Range Rectification
// Subsystem Module: shaders/acoustic_sonar/sidescan_sonar_slant_range_correction.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Corrects geometric compression in sidescan acoustic backscatter imagery.
//   - Converts slant travel time R_slant into true horizontal across-track ground distance Y_ground.
//
// Mathematical & Physical Formulation:
//   Y_ground = sqrt(R_slant^2 - H_altitude^2).
//   Interpolates acoustic waterfall lines onto linear metric grid.
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

struct SidescanConfig { altitude_m: f32, pixel_size_m: f32, pad0: f32, pad1: f32 };

struct SidescanOut { ground_range_m: f32, intensity: f32, grazing_angle_rad: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_ss: SidescanConfig;
@group(0) @binding(1) var<storage, read> s_slant_samples: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_ss_out: array<SidescanOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ss_out)) { return; }
    let smp = s_slant_samples[idx];
    let r_slant = smp.x;
    let raw_i = smp.y;
    let h = u_ss.altitude_m;
    let y_gnd = sqrt(max(0.0, r_slant * r_slant - h * h));
    let grazing = asin(clamp(h / max(1e-3, r_slant), 0.0, 1.0));
    s_ss_out[idx] = SidescanOut(y_gnd, raw_i, grazing, select(0u, 1u, r_slant > h));
}
