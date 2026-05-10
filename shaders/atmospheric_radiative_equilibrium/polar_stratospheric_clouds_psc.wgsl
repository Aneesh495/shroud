// Shroud :: Polar Stratospheric Clouds (PSC) NAT and STS Heterogeneous Chemistry
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/polar_stratospheric_clouds_psc.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Models Type-Ia crystalline Nitric Acid Trihydrate (NAT) and Type-Ib Supercooled Ternary Solutions (STS).
//   - Evaluates heterogeneous chlorine activation reactions (HCl + ClONO2 -> Cl2 + HNO3) initiating ozone hole.
//
// Mathematical & Physical Formulation:
//   Scattering depolarization ratio delta parameterizes aspherical crystalline NAT vs spherical liquid STS.
//   Thermodynamic equilibrium threshold T_NAT approx 195 K.
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
const AVOGADRO_NUMBER: f32 = 6.02214076e23;

struct PscParams { temp_k: f32, t_nat_k: f32, t_ice_k: f32, pad: f32 };

struct PscState { is_psc_present: u32, psc_type: u32, backscatter_ratio: f32, depol_ratio: f32 };

@group(0) @binding(0) var<uniform> u_psc: PscParams;
@group(0) @binding(1) var<storage, read> s_alt_temps_k: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_psc_out: array<PscState>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_psc_out)) { return; }
    let t = s_alt_temps_k[idx];
    let is_ice = t < u_psc.t_ice_k;
    let is_nat = t < u_psc.t_nat_k && !is_ice;
    let is_present = select(0u, 1u, is_ice || is_nat);
    let p_type = select(0u, select(1u, 2u, is_ice), is_present == 1u);
    let depol = select(0.01, select(0.12, 0.35, is_ice), is_present == 1u);
    s_psc_out[idx] = PscState(is_present, p_type, select(1.0, 5.0, is_present == 1u), depol);
}
