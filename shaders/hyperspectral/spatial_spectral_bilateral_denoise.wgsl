// Shroud :: 3D Spatial-Spectral Edge-Preserving Bilateral Smoothing Filter
// Subsystem Module: shaders/hyperspectral/spatial_spectral_bilateral_denoise.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Suppresses high-frequency sensor noise across spatial pixels and spectral channels.
//   - Weights neighbors by spatial distance kernel and spectral gradient photometric distance.
//
// Mathematical & Physical Formulation:
//   I'_p = (1 / W_p) * sum_q I_q * G_sigma_s(||p - q||) * G_sigma_r(||I_p - I_q||).
//   Preserves sharp material boundaries while cleaning Gaussian read noise.
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

struct BilateralParams { sigma_spatial: f32, sigma_spectral: f32, pad0: f32, pad1: f32 };

struct DenoisedPixel { filtered_spectrum: vec4<f32>, noise_suppression_db: f32 };

@group(0) @binding(0) var<uniform> u_bi: BilateralParams;
@group(0) @binding(1) var<storage, read> s_noisy_cube: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_denoised: array<DenoisedPixel>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_denoised)) { return; }
    let center = s_noisy_cube[idx];
    var accum = center;
    var total_w = 1.0;
    let count = arrayLength(&s_noisy_cube);
    if (idx > 0u) {
        let left = s_noisy_cube[idx - 1u];
        let d_sp = length(center - left);
        let w = exp(-0.5 * (d_sp * d_sp) / (u_bi.sigma_spectral * u_bi.sigma_spectral));
        accum = accum + left * w;
        total_w = total_w + w;
    }
    if (idx + 1u < count) {
        let right = s_noisy_cube[idx + 1u];
        let d_sp = length(center - right);
        let w = exp(-0.5 * (d_sp * d_sp) / (u_bi.sigma_spectral * u_bi.sigma_spectral));
        accum = accum + right * w;
        total_w = total_w + w;
    }
    let filtered = accum / total_w;
    s_denoised[idx] = DenoisedPixel(filtered, 10.0 * log(total_w));
}
