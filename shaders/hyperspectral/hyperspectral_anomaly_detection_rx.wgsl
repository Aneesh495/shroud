// Shroud :: Reed-Xiaoli (RX) Benchmark Anomaly Detection Kernel
// Subsystem Module: shaders/hyperspectral/hyperspectral_anomaly_detection_rx.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Mahalanobis distance between test pixel and background clutter covariance matrix.
//   - Detects sub-pixel targets and foreign man-made spectral anomalies.
//
// Mathematical & Physical Formulation:
//   RX(x) = (x - mu)^T * Sigma^-1 * (x - mu).
//   Follows chi-squared distribution with N degrees of freedom under Gaussian null hypothesis.
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

struct RxStats { mu0: f32, mu1: f32, mu2: f32, mu3: f32, inv_cov_diag: vec4<f32> };

struct RxResult { mahalanobis_d2: f32, p_value: f32, is_anomaly: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_rx: RxStats;
@group(0) @binding(1) var<storage, read> s_rx_pixels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_rx_out: array<RxResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rx_out)) { return; }
    let p = s_rx_pixels[idx];
    let mu = vec4<f32>(u_rx.mu0, u_rx.mu1, u_rx.mu2, u_rx.mu3);
    let diff = p - mu;
    let d2 = dot(diff * diff, u_rx.inv_cov_diag);
    let p_val = exp(-0.5 * d2);
    s_rx_out[idx] = RxResult(d2, p_val, select(0u, 1u, d2 > 15.0), 0.0);
}
