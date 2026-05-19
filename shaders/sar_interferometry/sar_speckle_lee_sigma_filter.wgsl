// Shroud :: Refined Lee Multiplicative Speckle Noise Adaptive Filter
// Subsystem Module: shaders/sar_interferometry/sar_speckle_lee_sigma_filter.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Reduces coherent multiplicative SAR speckle noise while preserving point targets and linear edges.
//   * Computes local coefficient of variation sigma_v and MMSE linear adaptive blending weight.
//
// Mathematical Formulation:
//   I_hat = I_mean + k * (I_center - I_mean)
//   k = var_x / (var_x + sigma_noise^2) adaptive local MMSE gain.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct LeeConfig { looks_n: f32, pad0: f32, pad1: f32, pad2: f32 };

struct LeeFilteredPixel { filtered_intensity: f32, mmse_weight_k: f32, local_mean: f32, enl_gain: f32 };

@group(0) @binding(0) var<uniform> u_lee: LeeConfig;
@group(0) @binding(1) var<storage, read> s_sar_amp: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_lee_out: array<LeeFilteredPixel>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let count = arrayLength(&s_sar_amp);
    if (idx >= count) { return; }
    let center = s_sar_amp[idx];
    let prev = s_sar_amp[select(0u, idx - 1u, idx > 0u)];
    let next = s_sar_amp[min(idx + 1u, count - 1u)];
    let mean = (prev + center + next) / 3.0;
    let var_tot = ((prev - mean) * (prev - mean) + (center - mean) * (center - mean) + (next - mean) * (next - mean)) / 3.0;
    let sigma_n2 = (mean * mean) / max(1.0, u_lee.looks_n);
    let k = clamp((var_tot - sigma_n2) / max(1e-5, var_tot), 0.0, 1.0);
    let out_i = mean + k * (center - mean);
    s_lee_out[idx] = LeeFilteredPixel(out_i, k, mean, 1.0 / max(0.05, 1.0 - k));
}
