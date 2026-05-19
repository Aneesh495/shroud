// Shroud :: Stimulated Brillouin Scattering (SBS) Electrostrictive Threshold
// Subsystem Module: shaders/nonlinear_optics/stimulated_brillouin_scattering.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates backward Stokes wave amplification driven by acoustic phonon Bragg gratings.
//   * Computes SBS threshold power P_th approx 21 * A_eff / (g_B * L_eff) for narrow-linewidth LiDAR.
//
// Mathematical Formulation:
//   nu_B = 2 * n_eff * v_acoustic / lambda_pump (~11 GHz in silica at 1550 nm)
//   L_eff = (1 - exp(-alpha * L)) / alpha effective nonlinear interaction length.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct SbsConfig { g_brillouin_m_w: f32, a_eff_um2: f32, alpha_db_km: f32, length_km: f32, n_eff: f32, v_acoustic_m_s: f32, lambda_nm: f32, pad: f32 };

struct SbsOutput { l_eff_m: f32, p_threshold_mw: f32, brillouin_shift_ghz: f32, stokes_gain_db: f32 };

@group(0) @binding(0) var<uniform> u_sbs: SbsConfig;
@group(0) @binding(1) var<storage, read> s_launch_powers_mw: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sbs_out: array<SbsOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sbs_out)) { return; }
    let p_mw = s_launch_powers_mw[idx];
    let alpha_m = (u_sbs.alpha_db_km * 0.2302585) * 1e-3;
    let len_m = u_sbs.length_km * 1000.0;
    let l_eff = (1.0 - exp(-alpha_m * len_m)) / max(1e-6, alpha_m);
    let a_eff_m2 = u_sbs.a_eff_um2 * 1e-12;
    let p_th_w = (21.0 * a_eff_m2) / max(1e-15, u_sbs.g_brillouin_m_w * l_eff);
    let nu_b_ghz = (2.0 * u_sbs.n_eff * u_sbs.v_acoustic_m_s) / max(1e-7, u_sbs.lambda_nm * 1e-9) * 1e-9;
    let gain_db = 4.343 * (u_sbs.g_brillouin_m_w * (p_mw * 1e-3) * l_eff / max(1e-15, a_eff_m2));
    s_sbs_out[idx] = SbsOutput(l_eff, p_th_w * 1000.0, nu_b_ghz, gain_db);
}
