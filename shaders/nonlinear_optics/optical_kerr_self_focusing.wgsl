// Shroud :: Optical Kerr Effect n_2 Intensity-Dependent Self-Focusing Threshold
// Subsystem Module: shaders/nonlinear_optics/optical_kerr_self_focusing.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates nonlinear refractive index shift Delta_n = n_2 * I(r, t) in fused silica and sapphire.
//   * Computes critical collapse power P_cr = 3.77 * lambda^2 / (8 * pi * n_0 * n_2) and B-integral.
//
// Mathematical Formulation:
//   n(I) = n_0 + n_2 * I
//   B = (2 * pi / lambda) * Integral n_2 * I(z) dz.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;

struct KerrConfig { n0: f32, n2_cm2_w: f32, lambda_nm: f32, path_len_m: f32 };

struct KerrOutput { delta_n: f32, b_integral_rad: f32, p_crit_mw: f32, filament_risk: u32 };

@group(0) @binding(0) var<uniform> u_kerr: KerrConfig;
@group(0) @binding(1) var<storage, read> s_intensities_gw_cm2: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_kerr_out: array<KerrOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_kerr_out)) { return; }
    let i_w_cm2 = s_intensities_gw_cm2[idx] * 1e9;
    let dn = u_kerr.n2_cm2_w * i_w_cm2;
    let lam_m = u_kerr.lambda_nm * 1e-9;
    let b_int = (TWO_PI / max(1e-8, lam_m)) * dn * u_kerr.path_len_m;
    let n2_m2_w = u_kerr.n2_cm2_w * 1e-4;
    let p_cr_w = (3.77 * lam_m * lam_m) / max(1e-24, 8.0 * PI * u_kerr.n0 * n2_m2_w);
    let risk = select(0u, 1u, b_int > 3.14159);
    s_kerr_out[idx] = KerrOutput(dn, b_int, p_cr_w * 1e-6, risk);
}
