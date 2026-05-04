// Shroud :: Glauber-Sudarshan P-Representation and Poisson Photon Distribution
// Subsystem Module: shaders/quantum_optics/coherent_state_glauber_p_representation.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates coherent state |alpha> photon number statistics P(n) = exp(-|alpha|^2) * |alpha|^(2*n) / n!.
//   - Fano factor F = Var(n) / <n> = 1.0 represents shot noise limit.
//
// Mathematical & Physical Formulation:
//   P-function P(alpha') = delta(alpha' - alpha) delta function representation.
//   Models laser output well above threshold.
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

struct CoherentConfig { alpha_mag: f32, pad0: f32, pad1: f32, pad2: f32 };

struct PoissonStat { photon_n: u32, prob_n: f32, mean_photons: f32, fano_factor: f32 };

@group(0) @binding(0) var<uniform> u_coh: CoherentConfig;
@group(0) @binding(1) var<storage, read> s_n_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> s_stat_out: array<PoissonStat>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_stat_out)) { return; }
    let n = s_n_values[idx];
    let n_mean = u_coh.alpha_mag * u_coh.alpha_mag;
    var p = exp(-n_mean);
    for (var i: u32 = 1u; i <= min(n, 20u); i = i + 1u) {
        p = p * (n_mean / f32(i));
    }
    s_stat_out[idx] = PoissonStat(n, p, n_mean, 1.0);
}
