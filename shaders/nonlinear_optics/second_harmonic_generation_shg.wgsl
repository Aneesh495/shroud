// Shroud :: Second-Harmonic Generation (SHG) Quasi-Phase-Matched Nonlinear Crystal
// Subsystem Module: shaders/nonlinear_optics/second_harmonic_generation_shg.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Couples fundamental pump wave omega and second-harmonic wave 2*omega in PPLN/KTP crystals.
//   * Evaluates phase-mismatch factor sinc^2(Delta_k * L / 2) and pump depletion efficiency.
//
// Mathematical Formulation:
//   d A_2 / dz = i * kappa * A_1^2 * exp(i * Delta_k * z)
//   eta_shg = tanh^2(sqrt(I_pump) * d_eff * L) under exact quasi-phase matching.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct ShgConfig { d_eff_pm_v: f32, crystal_len_mm: f32, delta_k_mm_inv: f32, n_omega: f32, n_2omega: f32, lambda_pump_nm: f32, pad0: f32, pad1: f32 };

struct ShgOutput { conversion_efficiency: f32, p_2omega_w: f32, phase_mismatch_rad: f32, coherence_len_um: f32 };

@group(0) @binding(0) var<uniform> u_shg: ShgConfig;
@group(0) @binding(1) var<storage, read> s_pump_irradiance: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_shg_out: array<ShgOutput>;

fn sinc_sq(x: f32) -> f32 {
    if (abs(x) < 1e-4) {
        return 1.0;
    }
    let s = sin(x) / x;
    return s * s;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_shg_out)) { return; }
    let i_pump = s_pump_irradiance[idx];
    let phi_mis = 0.5 * u_shg.delta_k_mm_inv * u_shg.crystal_len_mm;
    let pm_factor = sinc_sq(phi_mis);
    let drive = sqrt(max(0.0, i_pump)) * u_shg.d_eff_pm_v * u_shg.crystal_len_mm * 1e-4;
    let eff = clamp(pow(tanh(drive), 2.0) * pm_factor, 0.0, 0.99);
    let l_coh_um = (PI / max(1e-4, abs(u_shg.delta_k_mm_inv))) * 1000.0;
    s_shg_out[idx] = ShgOutput(eff, i_pump * eff, 2.0 * phi_mis, l_coh_um);
}
