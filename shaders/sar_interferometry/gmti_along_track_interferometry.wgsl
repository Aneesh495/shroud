// Shroud :: Along-Track Interferometry (ATI) Ground Moving Target Indicator
// Subsystem Module: shaders/sar_interferometry/gmti_along_track_interferometry.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates phase difference between fore and aft dual-antenna phase centers separated along track.
//   * Detects slow-moving maritime vessels and highway vehicles buried inside stationary ground clutter.
//
// Mathematical Formulation:
//   phi_ati = (4 * pi * B_along / lambda) * (v_radial / v_platform)
//   Displaced phase center antenna (DPCA) clutter suppression cancels stationary ground echoes.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct AtiConfig { b_along_m: f32, lambda_m: f32, v_platform_m_s: f32, min_vel_m_s: f32 };

struct GmtiDetection { ati_phase_rad: f32, radial_vel_m_s: f32, dpca_residue: f32, is_moving_target: u32 };

@group(0) @binding(0) var<uniform> u_ati: AtiConfig;
@group(0) @binding(1) var<storage, read> s_fore_aft: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_gmti_out: array<GmtiDetection>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_gmti_out)) { return; }
    let fa = s_fore_aft[idx];
    let f_re = fa.x;
    let f_im = fa.y;
    let a_re = fa.z;
    let a_im = fa.w;
    let c_re = f_re * a_re + f_im * a_im;
    let c_im = f_im * a_re - f_re * a_im;
    let phi = atan2(c_im, c_re);
    let vr = (phi * u_ati.lambda_m * u_ati.v_platform_m_s) / max(1e-4, FOUR_PI * u_ati.b_along_m);
    let dpca = sqrt((f_re - a_re) * (f_re - a_re) + (f_im - a_im) * (f_im - a_im));
    s_gmti_out[idx] = GmtiDetection(phi, vr, dpca, select(0u, 1u, abs(vr) > u_ati.min_vel_m_s));
}
