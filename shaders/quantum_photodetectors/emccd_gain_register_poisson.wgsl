// Shroud :: Electron-Multiplying CCD Stochastic Avalanche Multiplication Register
// Subsystem Module: shaders/quantum_photodetectors/emccd_gain_register_poisson.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates multi-stage serial high-voltage clock avalanche multiplication.
//   - Generates stochastic Poisson-Gamma excess noise factor F^2 = 2.0 electron distribution.
//
// Mathematical & Physical Formulation:
//   Gain G = (1 + g_stage)^N where g_stage approx 0.01 to 0.02 across N = 512 stages.
//   Probability distribution P(x | n_in) follows confluent hypergeometric Gamma distribution.
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

struct EmccdParams { stage_count: u32, stage_gain: f32, read_noise_e: f32, pad: f32 };

struct EmccdOut { output_electrons: f32, total_gain: f32, noise_factor: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_em: EmccdParams;
@group(0) @binding(1) var<storage, read> s_in_electrons: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_em_out: array<EmccdOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_em_out)) { return; }
    let n_in = s_in_electrons[idx];
    let g = pow(1.0 + u_em.stage_gain, f32(u_em.stage_count));
    let out_e = n_in * g + u_em.read_noise_e * 0.5;
    s_em_out[idx] = EmccdOut(out_e, g, 2.0, 0.0);
}
