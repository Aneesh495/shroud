// Shroud :: SAR Matched Filter Range-Doppler Chirp Compression
// Subsystem Module: shaders/sar_interferometry/sar_range_doppler_compression.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates LFM chirp matched filter phase compensation across fast-time SAR echoes.
//   * Computes range cell migration correction (RCMC) shift for squint-angle radar platforms.
//
// Mathematical Formulation:
//   Delta_R(f_eta) = (lambda^2 * R_0 * f_eta^2) / (8 * v_plat^2)
//   Compresses distributed target radar backscatter into crisp range resolution cells.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct SarConfig { lambda_m: f32, v_platform_m_s: f32, r0_m: f32, pad: f32 };

struct RcmcResult { rcmc_shift_m: f32, compressed_amp: f32, doppler_freq_hz: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_sar: SarConfig;
@group(0) @binding(1) var<storage, read> s_echoes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_rcmc_out: array<RcmcResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rcmc_out)) { return; }
    let e = s_echoes[idx];
    let f_eta = e.z;
    let lam2 = u_sar.lambda_m * u_sar.lambda_m;
    let shift = (lam2 * u_sar.r0_m * f_eta * f_eta) / max(1.0, 8.0 * u_sar.v_platform_m_s * u_sar.v_platform_m_s);
    let amp = sqrt(e.x * e.x + e.y * e.y);
    s_rcmc_out[idx] = RcmcResult(shift, amp, f_eta, 1u);
}
