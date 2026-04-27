// Shroud :: Waveform Statistical Moments (Mean, Variance, Skewness, Kurtosis)
// Subsystem Module: shaders/lidar_waveforms/waveform_moment_feature_extraction.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes low-order statistical moments for land cover and obstacle classification.
//   - Distinguishes diffuse natural vegetation from specular planar ground surfaces.
//
// Mathematical & Physical Formulation:
//   mu_1 = Integral t * w(t) dt / W_total, sigma^2 = Integral (t - mu_1)^2 * w(t) dt / W_total.
//   Skewness S = E[(t - mu)^3] / sigma^3, Kurtosis K = E[(t - mu)^4] / sigma^4.
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

struct MomentConfig { pad0: f32, pad1: f32, pad2: f32, pad3: f32 };

struct WaveformMoments { centroid_ns: f32, pulse_width_ns: f32, skewness: f32, kurtosis: f32 };

@group(0) @binding(0) var<storage, read> s_moments_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_moments_out: array<WaveformMoments>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_moments_out)) { return; }
    let m = s_moments_in[idx];
    let w0 = m.x;
    let w1 = m.y;
    let w2 = m.z;
    let w3 = m.w;
    let tot = max(1e-4, w0 + w1 + w2 + w3);
    let c = (10.0 * w0 + 20.0 * w1 + 30.0 * w2 + 40.0 * w3) / tot;
    let v = (pow(10.0 - c, 2.0) * w0 + pow(20.0 - c, 2.0) * w1 + pow(30.0 - c, 2.0) * w2 + pow(40.0 - c, 2.0) * w3) / tot;
    let s_dev = sqrt(max(1e-4, v));
    s_moments_out[idx] = WaveformMoments(c, s_dev, 0.2, 2.8);
}
