// Shroud :: Modulated Four-Sided Pyramid Wavefront Sensor Optical Gain
// Subsystem Module: shaders/adaptive_optics/pyramid_wfs_optical_modulation.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Models dynamic circular tip-tilt modulation around the apex of a 4-facet glass pyramid prism.
//   * Computes normalized intensity contrast signals S_x and S_y across four re-imaged pupil quadrants.
//   * Estimates non-linear optical gain attenuation under strong atmospheric turbulence residuals.
//
// Mathematical Formulation:
//   S_x = [(I_1 + I_4) - (I_2 + I_3)] / (I_1 + I_2 + I_3 + I_4)
//   Linearity range increases monotonically with modulation radius r_mod (in units of lambda / D).
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

struct PyramidConfig { mod_radius_lambda_d: f32, wavelength_nm: f32, pupil_diam_m: f32, pad: f32 };

struct PupilQuadIntensities { q_nw: f32, q_ne: f32, q_sw: f32, q_se: f32 };

struct PyramidSignalOut { sx: f32, sy: f32, optical_gain: f32, total_intensity: f32 };

@group(0) @binding(0) var<uniform> u_pwfs: PyramidConfig;
@group(0) @binding(1) var<storage, read> s_pupils: array<PupilQuadIntensities>;
@group(0) @binding(2) var<storage, read_write> s_pwfs_out: array<PyramidSignalOut>;

fn compute_modulation_gain(r_mod: f32) -> f32 {
    return 1.0 / max(1.0, 0.5 * PI * r_mod);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pwfs_out)) { return; }
    let p = s_pupils[idx];
    let sum_i = max(1e-5, p.q_nw + p.q_ne + p.q_sw + p.q_se);
    let sx = ((p.q_ne + p.q_se) - (p.q_nw + p.q_sw)) / sum_i;
    let sy = ((p.q_nw + p.q_ne) - (p.q_sw + p.q_se)) / sum_i;
    let gain = compute_modulation_gain(u_pwfs.mod_radius_lambda_d);
    s_pwfs_out[idx] = PyramidSignalOut(sx, sy, gain, sum_i);
}
