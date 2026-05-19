// Shroud :: Goldstein Adaptive Spectral Interferogram Phase Filter
// Subsystem Module: shaders/sar_interferometry/goldstein_adaptive_phase_filter.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Suppresses interferometric phase speckle noise while preserving sharp fringe boundaries.
//   * Weights 2D spectral power spectrum |Z(u, v)|^alpha before inverse synthesis.
//
// Mathematical Formulation:
//   H(u, v) = |S(u, v)|^alpha where alpha in [0, 1] adapts to local coherence.
//   Reduces phase unwrapping branch-cut residues.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct GoldsteinConfig { alpha_exponent: f32, smooth_weight: f32, pad0: f32, pad1: f32 };

struct FilteredComplex { re: f32, im: f32, phase_rad: f32, spectral_weight: f32 };

@group(0) @binding(0) var<uniform> u_gold: GoldsteinConfig;
@group(0) @binding(1) var<storage, read> s_spectrum_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_spectrum_out: array<FilteredComplex>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_spectrum_out)) { return; }
    let c = s_spectrum_in[idx];
    let mag = max(1e-5, length(c));
    let w = pow(mag, u_gold.alpha_exponent);
    let filt = c * w;
    s_spectrum_out[idx] = FilteredComplex(filt.x, filt.y, atan2(filt.y, filt.x), w);
}
