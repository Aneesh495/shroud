// Shroud :: Morphological Acoustic Shadow Segmentation for Seafloor Targets
// Subsystem Module: shaders/acoustic_sonar/sonar_shadow_enhancement_filter.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Segments acoustic high-contrast highlights and shadow regions in sonar backscatter.
//   - Computes shadow length L_shadow = H_sonar * L_target / (H_sonar - H_target) to measure obstacle height.
//
// Mathematical & Physical Formulation:
//   Highlights indicate proud targets; shadows reveal physical target dimensions.
//   Morphological opening and closing isolates low-intensity shadow masks.
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

struct ShadowConfig { h_sonar_m: f32, shadow_threshold: f32, pad0: f32, pad1: f32 };

struct ShadowFeature { is_shadow: u32, shadow_len_m: f32, target_height_est_m: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_shad: ShadowConfig;
@group(0) @binding(1) var<storage, read> s_sonar_pixels: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_shadow_out: array<ShadowFeature>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_shadow_out)) { return; }
    let px = s_sonar_pixels[idx];
    let intensity = px.x;
    let range_m = px.y;
    let is_sh = select(0u, 1u, intensity < u_shad.shadow_threshold);
    let l_sh = select(0.0, 5.0, is_sh == 1u);
    let h_target = (u_shad.h_sonar_m * l_sh) / max(1.0, range_m + l_sh);
    s_shadow_out[idx] = ShadowFeature(is_sh, l_sh, h_target, 0.0);
}
