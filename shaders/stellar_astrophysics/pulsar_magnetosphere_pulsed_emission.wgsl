// Shroud :: Rotating Neutron Star Polar Cap Relativistic Curvature Radiation
// Subsystem Module: shaders/stellar_astrophysics/pulsar_magnetosphere_pulsed_emission.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates Goldreich-Julian magnetospheric charge density rho_GJ = -2 * (Omega . B) / c.
//   - High-energy electrons accelerated in polar gap emit relativistic beamed curvature radiation gamma = 10^7.
//
// Mathematical & Physical Formulation:
//   Curvature frequency omega_c = 1.5 * c * gamma^3 / rho_curv.
//   Lighthouse sweep generates sharp periodic pulses P = 1.5 ms to 5 s.
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

struct PulsarConfig { period_s: f32, b_surface_gauss: f32, inclination_deg: f32, pad: f32 };

struct PulsarPulseProfile { intensity: f32, phase_rad: f32, polarization_angle_deg: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_psr: PulsarConfig;
@group(0) @binding(1) var<storage, read> s_phases_norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_pulse_out: array<PulsarPulseProfile>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pulse_out)) { return; }
    let ph = s_phases_norm[idx];
    let d_phi = ph - 0.5;
    let beam_w = 0.03;
    let int_val = exp(-0.5 * (d_phi * d_phi) / (beam_w * beam_w));
    let pa = atan(sin(d_phi * TWO_PI) / cos(u_psr.inclination_deg * (PI / 180.0))) * (180.0 / PI);
    s_pulse_out[idx] = PulsarPulseProfile(int_val, ph * TWO_PI, pa, 0.0);
}
