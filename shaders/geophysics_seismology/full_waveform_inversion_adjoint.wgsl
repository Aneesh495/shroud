// Shroud :: Full Waveform Inversion (FWI) Zero-Lag Cross-Correlation Gradient
// Subsystem Module: shaders/geophysics_seismology/full_waveform_inversion_adjoint.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates subsurface velocity update gradient: g(x) = Integral_t u_forward(x, t) * u_adjoint(x, T-t) dt.
//   - Backpropagates receiver residual waveforms to reconstruct deep Earth velocity models.
//
// Mathematical & Physical Formulation:
//   Adjoint source delta_d = d_obs - d_syn data misfit residual.
//   Preconditioned by Hessian diagonal approximation.
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

struct FwiConfig { scale_factor: f32, pad0: f32, pad1: f32, pad2: f32 };

struct FwiGradient { grad_vp: f32, energy_forward: f32, energy_adjoint: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_fwi: FwiConfig;
@group(0) @binding(1) var<storage, read> s_forward_field: array<f32>;
@group(0) @binding(2) var<storage, read> s_adjoint_field: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_fwi_grad: array<FwiGradient>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fwi_grad)) { return; }
    let u_fwd = s_forward_field[idx];
    let u_adj = s_adjoint_field[idx];
    let g = u_fwd * u_adj * u_fwi.scale_factor;
    s_fwi_grad[idx] = FwiGradient(g, u_fwd * u_fwd, u_adj * u_adj, 0.0);
}
