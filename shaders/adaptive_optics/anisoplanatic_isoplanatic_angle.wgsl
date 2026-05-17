// Shroud :: Atmospheric Isoplanatic Angle and Angular Anisoplanatism Error
// Subsystem Module: shaders/adaptive_optics/anisoplanatic_isoplanatic_angle.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates optical phase decorrelation as a function of angular field separation theta.
//   * Computes Fried isoplanatic angle theta_0 and wavefront variance sigma_aniso^2 = (theta / theta_0)^(5/3).
//   * Quantifies off-axis Strehl ratio degradation in wide-field adaptive optics.
//
// Mathematical Formulation:
//   theta_0 = [2.914 * k^2 * Integral C_n^2(h) * h^(5/3) dh]^(-3/5)
//   S(theta) approx exp(-(theta / theta_0)^(5/3)) angular Strehl attenuation.
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

struct AnisoConfig { theta0_arcsec: f32, r0_m: f32, mean_turb_height_m: f32, pad: f32 };

struct AnisoResult { off_axis_arcsec: f32, phase_var_rad2: f32, strehl_factor: f32, corr_coeff: f32 };

@group(0) @binding(0) var<uniform> u_aniso: AnisoConfig;
@group(0) @binding(1) var<storage, read> s_field_angles_arcsec: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_aniso_out: array<AnisoResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_aniso_out)) { return; }
    let theta = s_field_angles_arcsec[idx];
    let ratio = abs(theta) / max(0.1, u_aniso.theta0_arcsec);
    let var_rad2 = pow(ratio, 5.0 / 3.0);
    let strehl = exp(-min(10.0, var_rad2));
    s_aniso_out[idx] = AnisoResult(theta, var_rad2, strehl, exp(-0.5 * var_rad2));
}
