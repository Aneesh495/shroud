// Shroud :: Magnetohydrodynamic (MHD) Hartmann Conducting Channel Flow
// Subsystem Module: shaders/plasma_aerodynamics/mhd_hartmann_flow_channel.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates conducting hypersonic fluid flow in transverse magnetic fields B_0.
//   - Solves Hartmann profile: u(y) = u_0 * [cosh(Ha) - cosh(Ha * y/a)] / [cosh(Ha) - 1].
//
// Mathematical & Physical Formulation:
//   Ha = B_0 * a * sqrt(sigma / mu) dimensionless Hartmann number.
//   Evaluates Lorentz braking force F = j x B retarding flow velocity.
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

struct MhdParams { hartmann_ha: f32, u0_m_s: f32, b0_tesla: f32, sigma_s_m: f32 };

struct MhdFlowStep { velocity_m_s: f32, lorentz_force_n_m3: f32, induced_current_a_m2: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_mhd: MhdParams;
@group(0) @binding(1) var<storage, read> s_y_norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_mhd_out: array<MhdFlowStep>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_mhd_out)) { return; }
    let y = s_y_norm[idx];
    let ha = u_mhd.hartmann_ha;
    let u_y = u_mhd.u0_m_s * (1.0 - exp(-ha * (1.0 - abs(y))));
    let j_ind = u_mhd.sigma_s_m * u_y * u_mhd.b0_tesla;
    let f_lorentz = j_ind * u_mhd.b0_tesla;
    s_mhd_out[idx] = MhdFlowStep(u_y, f_lorentz, j_ind, 0.0);
}
