// Shroud :: Poisson-Boltzmann Electrostatic Debye Shielding Sheath
// Subsystem Module: shaders/plasma_aerodynamics/debye_shielding_sheath_potential.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates electrostatic potential screening around charged hypersonic probe bodies.
//   - Solves d^2 Phi / dx^2 = (e * n_0 / eps_0) * [exp(e * Phi / k * T_e) - 1].
//
// Mathematical & Physical Formulation:
//   lambda_D = sqrt(eps_0 * k * T_e / (n_e * e^2)) Debye length.
//   Potential decays exponentially: Phi(x) = Phi_0 * exp(-x / lambda_D).
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

struct DebyeConfig { phi0_v: f32, te_ev: f32, ne_m3: f32, pad: f32 };

struct SheathProfile { potential_v: f32, electric_field_v_m: f32, debye_len_m: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_deb: DebyeConfig;
@group(0) @binding(1) var<storage, read> s_dist_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sheath: array<SheathProfile>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sheath)) { return; }
    let x = s_dist_m[idx];
    let lambda_d = sqrt((8.854e-12 * u_deb.te_ev) / max(1e6, u_deb.ne_m3 * ELEMENTARY_CHARGE));
    let phi = u_deb.phi0_v * exp(-x / max(1e-6, lambda_d));
    let e_field = (phi / max(1e-6, lambda_d));
    s_sheath[idx] = SheathProfile(phi, e_field, lambda_d, 0.0);
}
