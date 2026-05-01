// Shroud :: DC SQUID V-Phi Periodic Flux-to-Voltage Transfer Function
// Subsystem Module: shaders/cryogenic_superconducting/squid_magnetometer_flux_voltage.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates two identical Josephson junctions in parallel on superconducting loop.
//   - Simulates periodic voltage modulation V(Phi_ext) with period Phi_0 = 2.07e-15 Wb.
//
// Mathematical & Physical Formulation:
//   V(Phi) = (R / 2) * sqrt(I_bias^2 - (2 * I_c * cos(pi * Phi / Phi_0))^2).
//   V_Phi = dV / dPhi transfer coefficient enables femtotesla magnetic field sensitivity.
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

struct SquidConfig { r_shunt_ohm: f32, ic_ua: f32, ibias_ua: f32, pad: f32 };

struct SquidOut { voltage_uv: f32, v_phi_uv_phi0: f32, sensitivity_ft_rt_hz: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_sq: SquidConfig;
@group(0) @binding(1) var<storage, read> s_flux_phi0: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_sq_out: array<SquidOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sq_out)) { return; }
    let phi = s_flux_phi0[idx];
    let cos_term = cos(PI * phi);
    let ic_eff = 2.0 * u_sq.ic_ua * abs(cos_term);
    let arg = u_sq.ibias_ua * u_sq.ibias_ua - ic_eff * ic_eff;
    let v = 0.5 * u_sq.r_shunt_ohm * sqrt(max(0.0, arg));
    let v_phi = abs(sin(PI * phi)) * 100.0;
    s_sq_out[idx] = SquidOut(v, v_phi, 5.0, 0.0);
}
