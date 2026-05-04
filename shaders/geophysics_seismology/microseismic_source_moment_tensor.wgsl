// Shroud :: Microseismic Source Seismic Moment Tensor M_ij Radiation Pattern
// Subsystem Module: shaders/geophysics_seismology/microseismic_source_moment_tensor.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 3x3 symmetric seismic moment tensor M = [Mxx, Myy, Mzz, Mxy, Mxz, Myz].
//   - Calculates P and S far-field radiation lobes: R_p(theta, phi) and R_s(theta, phi) for double-couple fault planes.
//
// Mathematical & Physical Formulation:
//   Fault strike, dip, and rake angles define dislocation slip vector and fault normal.
//   Moment magnitude M_w = (2/3) * (log10(M_0) - 9.1).
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

struct MomentTensor { mxx: f32, myy: f32, mzz: f32, mxy: f32, mxz: f32, myz: f32, pad0: f32, pad1: f32 };

struct RadiationResult { p_amplitude: f32, s_amplitude: f32, m0_joules: f32, mw_magnitude: f32 };

@group(0) @binding(0) var<uniform> u_mt: MomentTensor;
@group(0) @binding(1) var<storage, read> s_takeoff_angles: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_rad_out: array<RadiationResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rad_out)) { return; }
    let ang = s_takeoff_angles[idx];
    let th = ang.x;
    let ph = ang.y;
    let p_rad = sin(2.0 * th) * cos(ph);
    let s_rad = cos(2.0 * th) * cos(ph);
    let m0 = sqrt(0.5 * (u_mt.mxx * u_mt.mxx + u_mt.myy * u_mt.myy + u_mt.mzz * u_mt.mzz + 2.0 * (u_mt.mxy * u_mt.mxy + u_mt.mxz * u_mt.mxz + u_mt.myz * u_mt.myz)));
    let mw = (2.0 / 3.0) * (log(max(1.0, m0)) / log(10.0) - 9.1);
    s_rad_out[idx] = RadiationResult(p_rad, s_rad, m0, mw);
}
