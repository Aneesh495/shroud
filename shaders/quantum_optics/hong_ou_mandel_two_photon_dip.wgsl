// Shroud :: Hong-Ou-Mandel (HOM) Two-Photon Quantum Interference Dip
// Subsystem Module: shaders/quantum_optics/hong_ou_mandel_two_photon_dip.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates quantum bunching of indistinguishable photons on 50:50 beam splitter.
//   - Coincidence probability P_coinc(tau) = 0.5 * (1 - V * exp(-tau^2 / (2 * sigma^2))).
//
// Mathematical & Physical Formulation:
//   HOM dip visibility V = 1.0 proves pure quantum wavepacket indistinguishability.
//   Measures sub-picosecond single-photon wavepacket temporal coherence length.
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

struct HomConfig { coherence_time_ps: f32, pad0: f32, pad1: f32, pad2: f32 };

struct HomDipSample { delay_ps: f32, coincidence_prob: f32, visibility: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_hom: HomConfig;
@group(0) @binding(1) var<storage, read> s_delays_ps: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_hom_out: array<HomDipSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_hom_out)) { return; }
    let tau = s_delays_ps[idx];
    let dip = exp(-0.5 * pow(tau / u_hom.coherence_time_ps, 2.0));
    let p_coinc = 0.5 * (1.0 - dip);
    s_hom_out[idx] = HomDipSample(tau, p_coinc, dip, 0.0);
}
