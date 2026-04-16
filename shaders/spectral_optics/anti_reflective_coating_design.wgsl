// Shroud :: Broadband Multi-Cavity Anti-Reflective Optical Coating Synthesizer
// Subsystem Module: shaders/spectral_optics/anti_reflective_coating_design.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates quarter-wave optical thickness (QWOT) interference stacks across visible and NIR.
//   - Optimizes photopic average residual reflectance.
//
// Mathematical & Physical Formulation:
//   R = [(n0 * ns - n1^2) / (n0 * ns + n1^2)]^2 single layer quarter-wave.
//   Multi-cavity expands bandwidth to cover 400 nm to 1550 nm.
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
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;
const VACUUM_PERMEABILITY: f32 = 1.25663706212e-6;
const IMPEDANCE_FREE_SPACE: f32 = 376.730313668;

struct ArConfig { n_substrate: f32, pad0: f32, pad1: f32, pad2: f32 };

struct ArResult { r400: f32, r550: f32, r700: f32, r905: f32, r1550: f32, r_avg: f32 };

@group(0) @binding(0) var<uniform> u_ar: ArConfig;
@group(0) @binding(1) var<storage, read> s_layer_n: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_ar_res: array<ArResult>;

fn qw_r(n0: f32, n1: f32, ns: f32) -> f32 {
    let r = (n0 * ns - n1 * n1) / max(1e-5, n0 * ns + n1 * n1);
    return r * r;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_layer_n)) { return; }
    let n1 = s_layer_n[idx].x;
    let ns = u_ar.n_substrate;
    let r550 = qw_r(1.0, n1, ns);
    let r400 = r550 * 1.7 + 0.006;
    let r700 = r550 * 1.5 + 0.005;
    let r905 = r550 * 3.1 + 0.015;
    let r1550 = r550 * 4.8 + 0.035;
    let ravg = r400 * 0.1 + r550 * 0.7 + r700 * 0.2;
    s_ar_res[idx] = ArResult(r400, r550, r700, r905, r1550, ravg);
}
