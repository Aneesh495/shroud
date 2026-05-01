// Shroud :: Ginzburg-Landau Superconducting Order Parameter Relaxation
// Subsystem Module: shaders/cryogenic_superconducting/ginzburg_landau_order_parameter.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates complex macroscopic wavefunction Psi(r) = |Psi| * exp(i * phi).
//   - Solves Ginzburg-Landau equation: alpha * Psi + beta * |Psi|^2 * Psi + (1/2m) * (-i*hbar*grad - 2e*A)^2 Psi = 0.
//
// Mathematical & Physical Formulation:
//   xi(T) = hbar / sqrt(2 * m * |alpha|) coherence length.
//   Kappa = lambda_L / xi distinguishes Type-I (kappa < 1/sqrt(2)) from Type-II (kappa > 1/sqrt(2)).
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
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct GlParams { kappa_gl: f32, xi_nm: f32, pad0: f32, pad1: f32 };

struct GlOrderState { psi_mag2: f32, free_energy_density: f32, is_superconducting: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_gl: GlParams;
@group(0) @binding(1) var<storage, read> s_dist_xi: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_gl_out: array<GlOrderState>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_gl_out)) { return; }
    let x_xi = s_dist_xi[idx];
    let psi = tanh(x_xi * 0.707);
    let psi2 = psi * psi;
    let f_gl = -psi2 + 0.5 * psi2 * psi2;
    s_gl_out[idx] = GlOrderState(psi2, f_gl, select(0u, 1u, psi2 > 0.5), 0.0);
}
