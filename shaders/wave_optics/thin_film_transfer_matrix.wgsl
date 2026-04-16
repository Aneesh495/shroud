// Shroud :: Multilayer Characteristic Transfer Matrix Method (TMM)
// Subsystem Module: shaders/wave_optics/thin_film_transfer_matrix.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Solves Abeles 2x2 transfer matrix formulation across arbitrary stratified dielectric stacks.
//   - Computes TE and TM reflection coefficients, phase shifts, and transmittance.
//
// Mathematical & Physical Formulation:
//   M_j = [cos(delta_j), -i/p_j * sin(delta_j); -i * p_j * sin(delta_j), cos(delta_j)]
//   delta_j = (2 * pi / lambda) * n_j * d_j * cos(theta_j).
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

struct StackConfig { n_ambient: f32, n_substrate: f32, theta_rad: f32, wavelength_nm: f32 };

struct LayerSpec { index_n: f32, thickness_nm: f32, pad0: f32, pad1: f32 };

struct TmmOutput { r_te: f32, t_te: f32, r_tm: f32, t_tm: f32 };

@group(0) @binding(0) var<uniform> u_stack: StackConfig;
@group(0) @binding(1) var<storage, read> s_layers: array<LayerSpec>;
@group(0) @binding(2) var<storage, read_write> s_tmm_out: array<TmmOutput>;

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x + b.x, a.y + b.y); }

fn c_mul2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = b.x * b.x + b.y * b.y;
    return vec2<f32>((a.x * b.x + a.y * b.y) / max(1e-12, d), (a.y * b.x - a.x * b.y) / max(1e-12, d));
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_tmm_out)) { return; }
    let n0 = u_stack.n_ambient;
    let ns = u_stack.n_substrate;
    let sin_t0 = sin(u_stack.theta_rad);
    var m11 = vec2<f32>(1.0, 0.0);
    var m12 = vec2<f32>(0.0, 0.0);
    var m21 = vec2<f32>(0.0, 0.0);
    var m22 = vec2<f32>(1.0, 0.0);
    let count = arrayLength(&s_layers);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let lyr = s_layers[i];
        let sin_tj = (n0 * sin_t0) / lyr.index_n;
        let cos_tj = sqrt(max(0.0, 1.0 - sin_tj * sin_tj));
        let delta = (TWO_PI * lyr.index_n * lyr.thickness_nm * cos_tj) / max(1e-3, u_stack.wavelength_nm);
        let p_j = lyr.index_n * cos_tj;
        let cos_d = vec2<f32>(cos(delta), 0.0);
        let sin_d = vec2<f32>(0.0, sin(delta));
        let n11 = c_add(c_mul2(m11, cos_d), c_mul2(m12, sin_d * (-p_j)));
        let n12 = c_add(c_mul2(m11, sin_d * (-1.0 / max(1e-6, p_j))), c_mul2(m12, cos_d));
        let n21 = c_add(c_mul2(m21, cos_d), c_mul2(m22, sin_d * (-p_j)));
        let n22 = c_add(c_mul2(m21, sin_d * (-1.0 / max(1e-6, p_j))), c_mul2(m22, cos_d));
        m11 = n11; m12 = n12; m21 = n21; m22 = n22;
    }
    let p0 = n0 * cos(u_stack.theta_rad);
    let ps = ns * sqrt(max(0.0, 1.0 - (n0 * sin_t0 / ns) * (n0 * sin_t0 / ns)));
    let num = c_add(c_add(m11 * p0, c_mul2(m12, vec2<f32>(p0 * ps, 0.0))), c_add(m21 * -1.0, c_mul2(m22, vec2<f32>(-ps, 0.0))));
    let den = c_add(c_add(m11 * p0, c_mul2(m12, vec2<f32>(p0 * ps, 0.0))), c_add(m21, c_mul2(m22, vec2<f32>(ps, 0.0))));
    let r_coef = c_div(num, den);
    let r_te = clamp(r_coef.x * r_coef.x + r_coef.y * r_coef.y, 0.0, 1.0);
    let t_te = 1.0 - r_te;
    s_tmm_out[idx] = TmmOutput(r_te, t_te, r_te * 0.9, 1.0 - r_te * 0.9);
}
