// Shroud :: Wiener Frequency-Domain System Instrument Response Deconvolution
// Subsystem Module: shaders/lidar_waveforms/waveform_wiener_deconvolution.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Deconvolves laser pulse shape and photodetector impulse response function (IRF).
//   - Reconstructs true delta-reflection target differential cross-section profile.
//
// Mathematical & Physical Formulation:
//   H_w(omega) = S*(omega) / (|S(omega)|^2 + NSR) Wiener inverse filter.
//   Suppresses high-frequency noise amplification outside signal passband.
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

struct WienerConfig { nsr_constant: f32, pad0: f32, pad1: f32, pad2: f32 };

struct DeconvolvedSample { re: f32, im: f32, cross_section: f32, snr_gain_db: f32 };

@group(0) @binding(0) var<uniform> u_wn: WienerConfig;
@group(0) @binding(1) var<storage, read> s_fft_wf: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> s_fft_irf: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> s_deconv: array<DeconvolvedSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_deconv)) { return; }
    let y = s_fft_wf[idx];
    let h = s_fft_irf[idx % arrayLength(&s_fft_irf)];
    let h_mag2 = h.x * h.x + h.y * h.y;
    let den = max(1e-6, h_mag2 + u_wn.nsr_constant);
    let w_re = h.x / den;
    let w_im = -h.y / den;
    let x_re = y.x * w_re - y.y * w_im;
    let x_im = y.x * w_im + y.y * w_re;
    let sigma = sqrt(x_re * x_re + x_im * x_im);
    s_deconv[idx] = DeconvolvedSample(x_re, x_im, sigma, 10.0 * log(h_mag2 / max(1e-6, u_wn.nsr_constant)));
}
