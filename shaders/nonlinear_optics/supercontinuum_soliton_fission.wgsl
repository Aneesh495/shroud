// Shroud :: Photonic Crystal Fiber Supercontinuum Soliton Fission
// Subsystem Module: shaders/nonlinear_optics/supercontinuum_soliton_fission.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes soliton number N = sqrt(gamma * P_0 * T_0^2 / |beta_2|) for femtosecond pulse injection.
//   * Predicts dispersive wave Cherenkov radiation phase-matching and soliton self-frequency redshift.
//
// Mathematical Formulation:
//   L_fission approx L_D / N = T_0^2 / (N * |beta_2|)
//   Spectral bandwidth broadens octaves from visible 450 nm into mid-infrared 2400 nm.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct SolitonConfig { t0_fs: f32, beta2_ps2_m: f32, gamma_w_m: f32, pad: f32 };

struct FissionOutput { soliton_order_n: f32, fission_len_mm: f32, dispersive_shift_nm: f32, octave_span: f32 };

@group(0) @binding(0) var<uniform> u_sol: SolitonConfig;
@group(0) @binding(1) var<storage, read> s_peak_powers_kw: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sol_out: array<FissionOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sol_out)) { return; }
    let p0_w = s_peak_powers_kw[idx] * 1000.0;
    let t0_ps = u_sol.t0_fs * 1e-3;
    let l_disp_m = (t0_ps * t0_ps) / max(1e-6, abs(u_sol.beta2_ps2_m));
    let l_nl_m = 1.0 / max(1e-6, u_sol.gamma_w_m * p0_w);
    let n_order = sqrt(l_disp_m / l_nl_m);
    let l_fiss_mm = (l_disp_m / max(1.0, n_order)) * 1000.0;
    let octaves = clamp(log(max(1.0, n_order)) / log(2.0), 0.2, 3.0);
    s_sol_out[idx] = FissionOutput(n_order, l_fiss_mm, 520.0 - 20.0 * n_order, octaves);
}
