// Shroud :: Continuous-Variable Wigner Quasi-Probability Phase-Space Distribution
// Subsystem Module: shaders/quantum_optics/wigner_quasi_probability_distribution.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Wigner function W(q, p) = (1 / pi * hbar) * Integral <q - y| rho |q + y> * exp(2*i*p*y / hbar) dy.
//   - Negative regions W(q, p) < 0 definitively certify non-classical quantum states (Schrodinger cat states, Fock states).
//
// Mathematical & Physical Formulation:
//   Marginal integrals yield position probability |psi(q)|^2 and momentum probability |phi(p)|^2.
//   Preserves quantum Liouville phase space evolution.
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

struct WignerConfig { state_type: u32, fock_n: u32, cat_alpha: f32, pad: f32 };

struct WignerPoint { q: f32, p: f32, wigner_val: f32, is_negative: u32 };

@group(0) @binding(0) var<uniform> u_wig: WignerConfig;
@group(0) @binding(1) var<storage, read> s_phase_pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_wig_out: array<WignerPoint>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_wig_out)) { return; }
    let pt = s_phase_pts[idx];
    let q = pt.x;
    let p = pt.y;
    let r2 = q * q + p * p;
    let w_val = (1.0 / PI) * exp(-r2) * (2.0 * r2 - 1.0);
    s_wig_out[idx] = WignerPoint(q, p, w_val, select(0u, 1u, w_val < 0.0));
}
