// Shroud :: Polarimetric SAR Cloude-Pottier H-Alpha Eigenvalue Decomposition
// Subsystem Module: shaders/sar_interferometry/polsar_cloude_pottier_decomp.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Decomposes 3x3 coherency matrix T3 into scattering entropy H, anisotropy A, and mean alpha angle.
//   * Discriminates single-bounce surface scattering, double-bounce urban dihedral, and volume forest canopy.
//
// Mathematical Formulation:
//   H = -sum_{i=1}^3 P_i * log_3(P_i) where P_i = lambda_i / sum lambda_j
//   alpha_bar = sum_{i=1}^3 P_i * alpha_i in [0, 90 deg].
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct T3Diag { l1: f32, l2: f32, l3: f32, a1_deg: f32, a2_deg: f32, a3_deg: f32, pad0: f32, pad1: f32 };

struct CloudePottierOut { entropy_h: f32, anisotropy_a: f32, alpha_bar_deg: f32, class_id: u32 };

@group(0) @binding(0) var<storage, read> s_t3_in: array<T3Diag>;
@group(0) @binding(1) var<storage, read_write> s_cp_out: array<CloudePottierOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cp_out)) { return; }
    let t = s_t3_in[idx];
    let sum_l = max(1e-6, t.l1 + t.l2 + t.l3);
    let p1 = clamp(t.l1 / sum_l, 1e-5, 1.0);
    let p2 = clamp(t.l2 / sum_l, 1e-5, 1.0);
    let p3 = clamp(t.l3 / sum_l, 1e-5, 1.0);
    let inv_ln3 = 1.0 / log(3.0);
    let h = -(p1 * log(p1) + p2 * log(p2) + p3 * log(p3)) * inv_ln3;
    let aniso = (t.l2 - t.l3) / max(1e-6, t.l2 + t.l3);
    let alpha_b = p1 * t.a1_deg + p2 * t.a2_deg + p3 * t.a3_deg;
    let cid = select(select(1u, 2u, alpha_b > 42.0), 3u, h > 0.7);
    s_cp_out[idx] = CloudePottierOut(h, aniso, alpha_b, cid);
}
