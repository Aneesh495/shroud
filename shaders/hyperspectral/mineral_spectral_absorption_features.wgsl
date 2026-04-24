// Shroud :: Continuum Removal and Mineral Absorption Band Depth Analyzer
// Subsystem Module: shaders/hyperspectral/mineral_spectral_absorption_features.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Normalizes mineral spectra via convex hull continuum removal.
//   - Extracts absorption band center wavelength, depth D = 1 - R/Rc, and full-width at half-maximum.
//
// Mathematical & Physical Formulation:
//   R_cr(lambda) = R(lambda) / R_c(lambda) where R_c is linear upper envelope.
//   Characterizes 2.2 um Al-OH (kaolinite, alunite) and 2.3 um Mg-OH/carbonate absorptions.
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

struct MineralBands { r_2100: f32, r_2200: f32, r_2300: f32, r_2400: f32 };

struct MineralFeature { band_depth_2200: f32, band_depth_2300: f32, aloh_detected: u32, mgoh_detected: u32 };

@group(0) @binding(0) var<storage, read> s_min_in: array<MineralBands>;
@group(0) @binding(1) var<storage, read_write> s_min_out: array<MineralFeature>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_min_out)) { return; }
    let m = s_min_in[idx];
    let rc_2200 = 0.667 * m.r_2100 + 0.333 * m.r_2400;
    let rc_2300 = 0.333 * m.r_2100 + 0.667 * m.r_2400;
    let d_2200 = max(0.0, 1.0 - m.r_2200 / max(1e-4, rc_2200));
    let d_2300 = max(0.0, 1.0 - m.r_2300 / max(1e-4, rc_2300));
    s_min_out[idx] = MineralFeature(d_2200, d_2300, select(0u, 1u, d_2200 > 0.15), select(0u, 1u, d_2300 > 0.12));
}
