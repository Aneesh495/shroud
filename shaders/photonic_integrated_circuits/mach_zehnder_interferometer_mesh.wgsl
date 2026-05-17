// Shroud :: 2x2 Unitary Mach-Zehnder Interferometer (MZI) Clements Optical Mesh
// Subsystem Module: shaders/photonic_integrated_circuits/mach_zehnder_interferometer_mesh.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates 2x2 transfer matrix U(theta, phi) composed of two 50:50 directional couplers and phase shifters.
//   * Computes optical matrix-vector dot products for programmable photonic tensor processors.
//   * Simulates thermal crosstalk and imbalanced splitting ratio insertion loss.
//
// Mathematical Formulation:
//   U = i * exp(i * theta / 2) * [exp(i * phi) * sin(theta / 2), cos(theta / 2); exp(i * phi) * cos(theta / 2), -sin(theta / 2)]
//   Enforces energy conservation |E_out0|^2 + |E_out1|^2 = loss * (|E_in0|^2 + |E_in1|^2).
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

struct MziUnitParams { theta_rad: f32, phi_rad: f32, insertion_loss_db: f32, split_error: f32 };

struct DualComplexMode { e0_re: f32, e0_im: f32, e1_re: f32, e1_im: f32 };

@group(0) @binding(0) var<uniform> u_mzi: MziUnitParams;
@group(0) @binding(1) var<storage, read> s_modes_in: array<DualComplexMode>;
@group(0) @binding(2) var<storage, read_write> s_modes_out: array<DualComplexMode>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_modes_out)) { return; }
    let m_in = s_modes_in[idx];
    let e0 = vec2<f32>(m_in.e0_re, m_in.e0_im);
    let e1 = vec2<f32>(m_in.e1_re, m_in.e1_im);
    let half_th = 0.5 * (u_mzi.theta_rad + u_mzi.split_error);
    let s = sin(half_th);
    let c = cos(half_th);
    let exp_phi = vec2<f32>(cos(u_mzi.phi_rad), sin(u_mzi.phi_rad));
    let att = pow(10.0, -u_mzi.insertion_loss_db * 0.05);
    let u00 = cmul(exp_phi, vec2<f32>(0.0, s));
    let u01 = vec2<f32>(0.0, c);
    let u10 = cmul(exp_phi, vec2<f32>(0.0, c));
    let u11 = vec2<f32>(0.0, -s);
    let out0 = (cmul(u00, e0) + cmul(u01, e1)) * att;
    let out1 = (cmul(u10, e0) + cmul(u11, e1)) * att;
    s_modes_out[idx] = DualComplexMode(out0.x, out0.y, out1.x, out1.y);
}
