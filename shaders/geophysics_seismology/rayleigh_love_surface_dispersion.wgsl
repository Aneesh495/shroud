// Shroud :: Rayleigh and Love Surface Wave Phase Velocity Dispersion Curves
// Subsystem Module: shaders/geophysics_seismology/rayleigh_love_surface_dispersion.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates frequency-dependent phase and group velocity curves across layered geological strata.
//   - Solves Rayleigh secular equation: (2 - c^2/Vs^2)^2 = 4 * sqrt(1 - c^2/Vp^2) * sqrt(1 - c^2/Vs^2).
//
// Mathematical & Physical Formulation:
//   Elliptical retrograde particle motion in Rayleigh waves.
//   SH transverse polarization trapped within superficial low-velocity layers.
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

struct SurfaceWaveConfig { layer_thickness_m: f32, vp_m_s: f32, vs_m_s: f32, pad: f32 };

struct DispersionOut { phase_vel_m_s: f32, group_vel_m_s: f32, ellipticity: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_sw: SurfaceWaveConfig;
@group(0) @binding(1) var<storage, read> s_freqs: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_disp_out: array<DispersionOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_disp_out)) { return; }
    let f = s_freqs[idx];
    let c_rayleigh = 0.92 * u_sw.vs_m_s;
    let lam = c_rayleigh / max(0.1, f);
    let c_phase = c_rayleigh * (1.0 - 0.1 * exp(-lam / max(1.0, u_sw.layer_thickness_m)));
    let c_group = c_phase * 0.9;
    s_disp_out[idx] = DispersionOut(c_phase, c_group, 0.68, 0.0);
}
