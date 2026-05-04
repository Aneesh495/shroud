// Shroud :: Quadrature-Squeezed Vacuum Variance Below Standard Quantum Limit
// Subsystem Module: shaders/quantum_optics/squeezed_vacuum_quadrature_noise.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates position and momentum quadrature noise variances Delta X1^2 and Delta X2^2.
//   - Squeezing parameter r reduces variance below shot noise: Delta X1^2 = 0.25 * exp(-2*r) < 0.25.
//
// Mathematical & Physical Formulation:
//   Heisenberg uncertainty minimum: Delta X1 * Delta X2 = 0.25.
//   Enables gravitational wave detectors (LIGO, Virgo) to achieve sub-shot-noise sensitivity.
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

struct SqueezeParams { squeeze_r: f32, squeeze_theta_rad: f32, pad0: f32, pad1: f32 };

struct QuadratureVariance { var_x1: f32, var_x2: f32, db_squeezing: f32, is_sub_shot_noise: u32 };

@group(0) @binding(0) var<uniform> u_sqz: SqueezeParams;
@group(0) @binding(1) var<storage, read> s_quad_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sqz_out: array<QuadratureVariance>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sqz_out)) { return; }
    let th = s_quad_angles[idx];
    let r = u_sqz.squeeze_r;
    let v1 = 0.25 * exp(-2.0 * r);
    let v2 = 0.25 * exp(2.0 * r);
    let db = -10.0 * (log(v1 / 0.25) / log(10.0));
    s_sqz_out[idx] = QuadratureVariance(v1, v2, db, select(0u, 1u, v1 < 0.25));
}
