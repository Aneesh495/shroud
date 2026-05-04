// Shroud :: Jaynes-Cummings Cavity QED Atom-Photon Coherent Rabi Oscillations
// Subsystem Module: shaders/quantum_optics/cavity_qed_jaynes_cummings_dynamics.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Solves H = hbar * omega_c * a^dagger * a + 0.5 * hbar * omega_a * sigma_z + hbar * g * (a^dagger * sigma_- + a * sigma_+).
//   - Simulates vacuum Rabi splitting 2*g and reversible spontaneous emission in high-Q optical resonators.
//
// Mathematical & Physical Formulation:
//   Rabi oscillation frequency Omega_n = sqrt(Delta^2 + 4 * g^2 * (n + 1)).
//   Strong coupling regime g >> (kappa, gamma).
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

struct CavityQedConfig { coupling_g_mhz: f32, detuning_delta_mhz: f32, photon_n: u32, pad: f32 };

struct CavityState { excited_prob: f32, ground_prob: f32, rabi_freq_mhz: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_cqed: CavityQedConfig;
@group(0) @binding(1) var<storage, read> s_times_ns: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_cqed_out: array<CavityState>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cqed_out)) { return; }
    let t_ns = s_times_ns[idx];
    let g = u_cqed.coupling_g_mhz;
    let n = f32(u_cqed.photon_n);
    let omega_rabi = sqrt(u_cqed.detuning_delta_mhz * u_cqed.detuning_delta_mhz + 4.0 * g * g * (n + 1.0));
    let t_s = t_ns * 1e-9;
    let arg = TWO_PI * omega_rabi * 1e6 * t_s * 0.5;
    let p_e = pow(cos(arg), 2.0);
    s_cqed_out[idx] = CavityState(p_e, 1.0 - p_e, omega_rabi, 0.0);
}
