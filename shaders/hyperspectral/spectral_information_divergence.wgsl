// Shroud :: Spectral Information Divergence (SID) Stochastic Measure
// Subsystem Module: shaders/hyperspectral/spectral_information_divergence.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates relative entropy / Kullback-Leibler divergence between spectral probability vectors.
//   - Measures stochastic spectral similarity robust to non-linear mixed effects.
//
// Mathematical & Physical Formulation:
//   SID(x, y) = D(x || y) + D(y || x) where p_i = x_i / sum(x_j).
//   D(x || y) = sum p_i * log(p_i / q_i).
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

struct SidRefVector { q0: f32, q1: f32, q2: f32, q3: f32 };

struct SidOutput { divergence: f32, kld_xy: f32, kld_yx: f32, is_similar: u32 };

@group(0) @binding(0) var<uniform> u_sid_ref: SidRefVector;
@group(0) @binding(1) var<storage, read> s_sid_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_sid_out: array<SidOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sid_out)) { return; }
    let p = s_sid_in[idx];
    let sum_p = max(1e-5, p.x + p.y + p.z + p.w);
    let px = p / sum_p;
    let q_raw = vec4<f32>(u_sid_ref.q0, u_sid_ref.q1, u_sid_ref.q2, u_sid_ref.q3);
    let qy = q_raw / max(1e-5, q_raw.x + q_raw.y + q_raw.z + q_raw.w);
    let d_xy = px.x * log(max(1e-4, px.x / max(1e-4, qy.x))) + px.y * log(max(1e-4, px.y / max(1e-4, qy.y))) + px.z * log(max(1e-4, px.z / max(1e-4, qy.z))) + px.w * log(max(1e-4, px.w / max(1e-4, qy.w)));
    let d_yx = qy.x * log(max(1e-4, qy.x / max(1e-4, px.x))) + qy.y * log(max(1e-4, qy.y / max(1e-4, px.y))) + qy.z * log(max(1e-4, qy.z / max(1e-4, px.z))) + qy.w * log(max(1e-4, qy.w / max(1e-4, px.w)));
    let sid = d_xy + d_yx;
    s_sid_out[idx] = SidOutput(sid, d_xy, d_yx, select(0u, 1u, sid < 0.15));
}
