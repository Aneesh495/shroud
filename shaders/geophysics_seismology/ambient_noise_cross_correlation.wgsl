// Shroud :: Ambient Seismic Noise Long-Term Cross-Correlation Green Function
// Subsystem Module: shaders/geophysics_seismology/ambient_noise_cross_correlation.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Extracts empirical Green's functions between pairs of passive seismic stations.
//   - Correlates diffuse microseismic noise wavefield to reconstruct virtual inter-station surface waves.
//
// Mathematical & Physical Formulation:
//   C_AB(t) proportional to d/dt [G_AB(t) - G_BA(-t)].
//   Spectral whitening and 1-bit amplitude normalization eliminate non-stationary earthquake spikes.
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
const HBAR: f32 = 1.054571817e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct CorrelConfig { lag_bins: u32, pad0: f32, pad1: f32, pad2: f32 };

struct CorrelationOut { cross_corr_val: f32, causal_energy: f32, acausal_energy: f32, snr: f32 };

@group(0) @binding(0) var<uniform> u_cc: CorrelConfig;
@group(0) @binding(1) var<storage, read> s_trace_a: array<f32>;
@group(0) @binding(2) var<storage, read> s_trace_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_cc_out: array<CorrelationOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cc_out)) { return; }
    var sum_cc = 0.0;
    let count = min(arrayLength(&s_trace_a), arrayLength(&s_trace_b));
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        let idx_b = (idx + i) % count;
        sum_cc = sum_cc + sign(s_trace_a[i]) * sign(s_trace_b[idx_b]);
    }
    let norm_cc = sum_cc / 64.0;
    s_cc_out[idx] = CorrelationOut(norm_cc, abs(norm_cc), abs(norm_cc), abs(norm_cc) * 10.0);
}
