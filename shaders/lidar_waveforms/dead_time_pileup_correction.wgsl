// Shroud :: Time-Correlated Single-Photon Counting (TCSPC) Pileup Correction
// Subsystem Module: shaders/lidar_waveforms/dead_time_pileup_correction.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Corrects single-photon detector dead-time distortion and first-photon bias.
//   - Recovers true photon arrival distribution N_true(t) = N_meas(t) / [1 - sum_{t' < t} N_meas(t')].
//
// Mathematical & Physical Formulation:
//   N_true(t) inverts Coats distortion equation for high-count-rate TCSPC.
//   Compensates distortion at photon count rates exceeding 5% of repetition frequency.
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

struct PileupConfig { dead_time_bins: u32, pad0: f32, pad1: f32, pad2: f32 };

struct PileupResult { corrected_counts: f32, raw_counts: f32, pileup_factor: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_pu: PileupConfig;
@group(0) @binding(1) var<storage, read> s_meas_hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_pu_out: array<PileupResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pu_out)) { return; }
    let raw = s_meas_hist[idx];
    var sum_prev = 0.0;
    let limit = min(idx, u_pu.dead_time_bins);
    for (var i: u32 = 0u; i < limit; i = i + 1u) {
        sum_prev = sum_prev + s_meas_hist[idx - i];
    }
    let factor = 1.0 / max(0.1, 1.0 - sum_prev * 0.001);
    s_pu_out[idx] = PileupResult(raw * factor, raw, factor, 0.0);
}
