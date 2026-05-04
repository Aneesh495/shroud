// Shroud :: Clauser-Horne-Shimony-Holt (CHSH) Bell Inequality Violation
// Subsystem Module: shaders/quantum_optics/bell_chsh_inequality_violation.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates correlator S = |E(a, b) - E(a, b') + E(a', b) + E(a', b')| across polarizer angle settings.
//   - Classical local realism bounds S <= 2. Quantum mechanics achieves Cirel'son bound S = 2 * sqrt(2) approx 2.828.
//
// Mathematical & Physical Formulation:
//   Proves quantum non-locality and excludes all local hidden variable (LHV) theories.
//   E(theta_a, theta_b) = -cos(2 * (theta_a - theta_b)) for singlet state |Psi->.
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

struct ChshAngles { theta_a: f32, theta_a_prime: f32, theta_b: f32, theta_b_prime: f32 };

struct BellResult { chsh_s: f32, is_violation: u32, cirelson_ratio: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_chsh: ChshAngles;
@group(0) @binding(1) var<storage, read> s_eval_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> s_bell_out: array<BellResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_bell_out)) { return; }
    let a = u_chsh.theta_a;
    let ap = u_chsh.theta_a_prime;
    let b = u_chsh.theta_b;
    let bp = u_chsh.theta_b_prime;
    let e_ab = -cos(2.0 * (a - b));
    let e_abp = -cos(2.0 * (a - bp));
    let e_apb = -cos(2.0 * (ap - b));
    let e_apbp = -cos(2.0 * (ap - bp));
    let s_val = abs(e_ab - e_abp + e_apb + e_apbp);
    let cirelson = 2.0 * sqrt(2.0);
    s_bell_out[idx] = BellResult(s_val, select(0u, 1u, s_val > 2.0), s_val / cirelson, 0.0);
}
