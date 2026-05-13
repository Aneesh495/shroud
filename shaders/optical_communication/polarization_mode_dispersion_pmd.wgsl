// Shroud :: First-Order Differential Group Delay (DGD) and PMD Vector Rotation
// Subsystem Module: shaders/optical_communication/polarization_mode_dispersion_pmd.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates fiber core circular asymmetry (birefringence) inducing differential delay between principal states of polarization (PSP).
//   - DGD Delta_tau = ||Omega_pmd|| causes pulse broadening and inter-symbol interference.
//
// Mathematical & Physical Formulation:
//   PMD accumulates stochastically as sqrt(L): <Delta_tau> = D_pmd * sqrt(L) where D_pmd approx 0.05 ps/sqrt(km).
//   2x2 Jones matrix propagation: J(omega) = exp(-i * 0.5 * omega * Delta_tau * sigma_3).
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
const GRAVITATIONAL_CONSTANT: f32 = 6.67430e-11;
const SOLAR_MASS_KG: f32 = 1.98847e30;
const SOLAR_RADIUS_M: f32 = 6.957e8;

struct PmdConfig { fiber_length_km: f32, d_pmd_ps_sqrt_km: f32, pad0: f32, pad1: f32 };

struct PmdResult { dgd_ps: f32, pulse_broadening_factor: f32, psp_angle_deg: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_pmd: PmdConfig;
@group(0) @binding(1) var<storage, read> s_pulse_durations_ps: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_pmd_out: array<PmdResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pmd_out)) { return; }
    let t0 = s_pulse_durations_ps[idx];
    let mean_dgd = u_pmd.d_pmd_ps_sqrt_km * sqrt(u_pmd.fiber_length_km);
    let broad = sqrt(1.0 + pow(mean_dgd / max(1.0, t0), 2.0));
    s_pmd_out[idx] = PmdResult(mean_dgd, broad, 45.0, 0.0);
}
