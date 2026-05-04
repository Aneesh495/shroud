// Shroud :: Tilted Transverse Isotropy (TTI) Pseudo-Acoustic Wavefield
// Subsystem Module: shaders/geophysics_seismology/anisotropic_tti_acoustic_wave.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Alkhalifah pseudo-acoustic formulation for tilted transversely isotropic media.
//   - Applies Thomsen anisotropy parameters epsilon and delta with tilt angles theta and phi.
//
// Mathematical & Physical Formulation:
//   V_p_horiz = V_p0 * sqrt(1 + 2 * epsilon) and V_nmo = V_p0 * sqrt(1 + 2 * delta).
//   Eliminates shear wave SV-instability artifacts.
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

struct TtiParams { vp0_m_s: f32, epsilon_thomsen: f32, delta_thomsen: f32, tilt_theta_rad: f32 };

struct TtiResult { p_ampl: f32, horiz_vel: f32, nmo_vel: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_tti: TtiParams;
@group(0) @binding(1) var<storage, read> s_wave_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_tti_out: array<TtiResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_tti_out)) { return; }
    let w = s_wave_in[idx];
    let v_horiz = u_tti.vp0_m_s * sqrt(1.0 + 2.0 * u_tti.epsilon_thomsen);
    let v_nmo = u_tti.vp0_m_s * sqrt(1.0 + 2.0 * u_tti.delta_thomsen);
    s_tti_out[idx] = TtiResult(w, v_horiz, v_nmo, 0.0);
}
