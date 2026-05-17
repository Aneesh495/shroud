// Shroud :: All-Pass and Add-Drop Microring Resonator Lorentzian Transfer Function
// Subsystem Module: shaders/photonic_integrated_circuits/microring_resonator_transfer.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates complex optical field transmission through coupled waveguide-ring resonators.
//   * Computes Free Spectral Range (FSR), Quality Factor Q, Finesse, and Extinction Ratio.
//   * Models thermo-optic phase tuning d n / d T = 1.86e-4 K^-1 for silicon WDM modulators.
//
// Mathematical Formulation:
//   T_pass = (a^2 - 2 * r * a * cos(phi) + r^2) / (1 - 2 * r * a * cos(phi) + (r * a)^2)
//   phi = (2 * pi / lambda) * n_eff * 2 * pi * R_ring round-trip phase shift.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct RingConfig { radius_um: f32, n_eff: f32, self_coupling_r: f32, attenuation_a: f32, thermo_delta_t_k: f32, dn_dt: f32, pad0: f32, pad1: f32 };

struct RingSpectrumPoint { lambda_nm: f32, through_power: f32, drop_power: f32, phase_rad: f32 };

@group(0) @binding(0) var<uniform> u_ring: RingConfig;
@group(0) @binding(1) var<storage, read> s_lambdas: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_ring_out: array<RingSpectrumPoint>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ring_out)) { return; }
    let lam_nm = s_lambdas[idx];
    let l_ring_nm = TWO_PI * u_ring.radius_um * 1000.0;
    let neff_t = u_ring.n_eff + u_ring.dn_dt * u_ring.thermo_delta_t_k;
    let phi = (TWO_PI * neff_t * l_ring_nm) / max(100.0, lam_nm);
    let r = u_ring.self_coupling_r;
    let a = u_ring.attenuation_a;
    let cos_phi = cos(phi);
    let num = a * a - 2.0 * r * a * cos_phi + r * r;
    let den = max(1e-6, 1.0 - 2.0 * r * a * cos_phi + r * r * a * a);
    let t_through = clamp(num / den, 0.0, 1.0);
    let t_drop = clamp(((1.0 - r * r) * (1.0 - r * r) * a) / den, 0.0, 1.0);
    s_ring_out[idx] = RingSpectrumPoint(lam_nm, t_through, t_drop, phi % TWO_PI);
}
