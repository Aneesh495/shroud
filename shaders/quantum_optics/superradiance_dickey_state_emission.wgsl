// Shroud :: Dicke Superradiance Collective Spontaneous Emission Pulse
// Subsystem Module: shaders/quantum_optics/superradiance_dickey_state_emission.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates coherent collective spontaneous decay of N inverted two-level atoms within subwavelength volume.
//   - Peak emission intensity scales superlinearly as I_max proportional to N^2 (coherent dipole superposition).
//
// Mathematical & Physical Formulation:
//   Pulse delay time t_delay = tau_0 / N * ln(N), pulse duration tau_pulse = tau_0 / N.
//   Macroscopic quantum coherence emerges spontaneously from initial vacuum fluctuations.
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

struct DickeConfig { atom_count_n: f32, single_atom_tau_ns: f32, pad0: f32, pad1: f32 };

struct SuperradiantPulse { intensity: f32, delay_time_ns: f32, pulse_width_ns: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_dicke: DickeConfig;
@group(0) @binding(1) var<storage, read> s_time_grid_ns: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_pulse_out: array<SuperradiantPulse>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pulse_out)) { return; }
    let t = s_time_grid_ns[idx];
    let n = u_dicke.atom_count_n;
    let t0 = u_dicke.single_atom_tau_ns;
    let t_delay = (t0 / n) * log(max(2.0, n));
    let tau_p = t0 / n;
    let dt = (t - t_delay) / max(1e-4, tau_p);
    let sech_val = 1.0 / cosh(dt);
    let intensity = 0.25 * n * n * sech_val * sech_val;
    s_pulse_out[idx] = SuperradiantPulse(intensity, t_delay, tau_p, 0.0);
}
