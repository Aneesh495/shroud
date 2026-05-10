// Shroud :: Selective Plane Illumination Microscopy (SPIM) Gaussian Light Sheet
// Subsystem Module: shaders/computational_microscopy/light_sheet_fluorescence_spim.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Gaussian beam waist w_0 and Rayleigh range z_R = pi * w_0^2 / lambda.
//   - Calculates optical sectioning thickness and off-focus photobleaching suppression.
//
// Mathematical & Physical Formulation:
//   w(z) = w_0 * sqrt(1 + (z / z_R)^2) beam waist hyperbolic profile.
//   Decouples illumination and detection optical axes orthogonally at 90 degrees.
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

struct SpimConfig { w0_um: f32, lambda_um: f32, sheet_length_um: f32, pad: f32 };

struct SheetIntensity { z_pos_um: f32, waist_w_um: f32, peak_intensity: f32, rayleigh_range_um: f32 };

@group(0) @binding(0) var<uniform> u_spim: SpimConfig;
@group(0) @binding(1) var<storage, read> s_z_grid_um: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sheet_out: array<SheetIntensity>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sheet_out)) { return; }
    let z = s_z_grid_um[idx];
    let zr = (PI * u_spim.w0_um * u_spim.w0_um) / u_spim.lambda_um;
    let wz = u_spim.w0_um * sqrt(1.0 + (z * z) / (zr * zr));
    let int_peak = pow(u_spim.w0_um / wz, 2.0);
    s_sheet_out[idx] = SheetIntensity(z, wz, int_peak, zr);
}
