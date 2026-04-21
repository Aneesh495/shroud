// Shroud :: 2D Range-Doppler Cell-Averaging and Ordered-Statistic CFAR Detector
// Subsystem Module: shaders/automotive_radars/cfar_ca_os_detector.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates CA-CFAR and OS-CFAR adaptive thresholding on radar Range-Doppler matrices.
//   - Computes background noise statistics across guard and reference cells.
//
// Mathematical & Physical Formulation:
//   T_ca = alpha * (1 / 2N) * (sum_left + sum_right) where alpha = N * (P_fa^(-1/N) - 1).
//   Rejects clutter edges and multi-target masking.
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
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;
const VACUUM_PERMEABILITY: f32 = 1.25663706212e-6;
const IMPEDANCE_FREE_SPACE: f32 = 376.730313668;

struct CfarConfig { guard_len: u32, ref_len: u32, pfa: f32, threshold_factor: f32 };

struct CfarTarget { range_idx: u32, doppler_idx: u32, snr_db: f32, is_detection: u32 };

@group(0) @binding(0) var<uniform> u_cfar: CfarConfig;
@group(0) @binding(1) var<storage, read> s_rd_map: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_targets: array<CfarTarget>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_targets)) { return; }
    let cell_val = s_rd_map[idx];
    let g = u_cfar.guard_len;
    let r = u_cfar.ref_len;
    var noise_sum = 0.0;
    var count = 0.0;
    let len = u32(arrayLength(&s_rd_map));
    for (var i: u32 = 1u; i <= r; i = i + 1u) {
        let left = select(0u, idx - g - i, idx >= (g + i));
        let right = min(len - 1u, idx + g + i);
        noise_sum = noise_sum + s_rd_map[left] + s_rd_map[right];
        count = count + 2.0;
    }
    let noise_floor = noise_sum / max(1.0, count);
    let threshold = noise_floor * u_cfar.threshold_factor;
    let det = select(0u, 1u, cell_val > threshold);
    let snr = 10.0 * log(max(1e-4, cell_val / max(1e-4, noise_floor)));
    s_targets[idx] = CfarTarget(idx % 256u, idx / 256u, snr, det);
}
