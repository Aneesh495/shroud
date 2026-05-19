// Shroud :: InSAR Complex Interferometric Coherence and Phase Noise Estimator
// Subsystem Module: shaders/sar_interferometry/insar_complex_coherence.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes complex Hermitian inner product across master and slave Single Look Complex (SLC) SAR pairs.
//   * Estimates interferometric phase phi_int and coherence magnitude gamma in [0, 1].
//
// Mathematical Formulation:
//   gamma = |sum M * conj(S)| / sqrt(sum |M|^2 * sum |S|^2)
//   phi_int = arg(sum M * conj(S)) tracks millimeter-scale surface deformation.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct SlcPair { master_re: f32, master_im: f32, slave_re: f32, slave_im: f32 };

struct CoherenceResult { interferometric_phase: f32, coherence_gamma: f32, intensity_db: f32, pad: f32 };

@group(0) @binding(0) var<storage, read> s_slc_pairs: array<SlcPair>;
@group(0) @binding(1) var<storage, read_write> s_coh_out: array<CoherenceResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_coh_out)) { return; }
    let p = s_slc_pairs[idx];
    let cross_re = p.master_re * p.slave_re + p.master_im * p.slave_im;
    let cross_im = p.master_im * p.slave_re - p.master_re * p.slave_im;
    let m2 = p.master_re * p.master_re + p.master_im * p.master_im;
    let s2 = p.slave_re * p.slave_re + p.slave_im * p.slave_im;
    let cross_mag = sqrt(cross_re * cross_re + cross_im * cross_im);
    let gamma = clamp(cross_mag / max(1e-6, sqrt(m2 * s2)), 0.0, 1.0);
    let phi = atan2(cross_im, cross_re);
    let db = 10.0 * (log(max(1e-6, m2)) / log(10.0));
    s_coh_out[idx] = CoherenceResult(phi, gamma, db, 0.0);
}
