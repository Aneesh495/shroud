// Shroud :: Viscoelastic Standard Linear Solid Seismic Quality Factor Q
// Subsystem Module: shaders/geophysics_seismology/attenuation_quality_factor_q.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Models seismic attenuation and phase velocity dispersion across anelastic rock strata.
//   - Quality factor Q^-1 = Delta_E / (2 * pi * E) defines intrinsic dissipation.
//
// Mathematical & Physical Formulation:
//   A(x, f) = A_0 * exp(-pi * f * x / (v * Q)) exponential attenuation.
//   Futterman causal dispersion relation relates phase velocity to frequency: v(f) = v_0 * [1 + (1 / pi * Q) * ln(f / f_0)].
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

struct AttenParams { q_factor: f32, v0_m_s: f32, f0_hz: f32, pad: f32 };

struct AttenResult { attenuated_amp: f32, dispersed_vel: f32, phase_delay_rad: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_att: AttenParams;
@group(0) @binding(1) var<storage, read> s_wave_samples: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_att_out: array<AttenResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_att_out)) { return; }
    let smp = s_wave_samples[idx];
    let dist_m = smp.x;
    let freq = smp.y;
    let att_fac = exp(-PI * freq * dist_m / (u_att.v0_m_s * u_att.q_factor));
    let v_disp = u_att.v0_m_s * (1.0 + (1.0 / (PI * u_att.q_factor)) * log(max(0.1, freq / u_att.f0_hz)));
    s_att_out[idx] = AttenResult(att_fac, v_disp, (TWO_PI * freq * dist_m) / v_disp, 0.0);
}
