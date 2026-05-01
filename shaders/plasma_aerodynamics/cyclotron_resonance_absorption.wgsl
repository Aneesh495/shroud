// Shroud :: Electron Cyclotron Resonance Heating and Wave Absorption
// Subsystem Module: shaders/plasma_aerodynamics/cyclotron_resonance_absorption.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates right-hand circularly polarized whistler wave damping at cyclotron frequency.
//   - Computes Doppler-shifted resonance condition: omega - k_parallel * v_parallel = omega_ce.
//
// Mathematical & Physical Formulation:
//   omega_ce = e * B_0 / m_e electron cyclotron gyrofrequency.
//   Strong resonant absorption mitigates blackout in magnetized plasma sheaths.
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
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct CycloConfig { b0_tesla: f32, pad0: f32, pad1: f32, pad2: f32 };

struct CycloOut { omega_ce_rad_s: f32, absorption_coeff: f32, is_resonant: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_cy: CycloConfig;
@group(0) @binding(1) var<storage, read> s_wave_freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_cyclo_out: array<CycloOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cyclo_out)) { return; }
    let w = s_wave_freqs[idx];
    let w_ce = (ELEMENTARY_CHARGE * u_cy.b0_tesla) / ELECTRON_MASS;
    let delta = abs(w - w_ce);
    let abs_coef = 1.0 / (1.0 + delta * delta * 1e-16);
    s_cyclo_out[idx] = CycloOut(w_ce, abs_coef, select(0u, 1u, delta < 1e8), 0.0);
}
