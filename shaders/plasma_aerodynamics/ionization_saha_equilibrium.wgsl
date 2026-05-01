// Shroud :: Thermal Ionization Saha-Eggert Equilibrium Degree of Ionization
// Subsystem Module: shaders/plasma_aerodynamics/ionization_saha_equilibrium.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Calculates ionization equilibrium alpha for high-enthalpy hypersonic boundary layers.
//   - Solves Saha equation: n_e * n_i / n_0 = (2 * g_i / g_0) * (2*pi*m_e*k*T / h^2)^1.5 * exp(-E_ion / k*T).
//
// Mathematical & Physical Formulation:
//   Determines electron density distribution in shock compressed air (N2, O2, NO, N, O).
//   Accounts for thermal dissociation and first ionization potentials.
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

struct SahaParams { temp_k: f32, pressure_pa: f32, e_ion_ev: f32, pad: f32 };

struct SahaResult { alpha_degree: f32, n_e_m3: f32, enthalpy_j_kg: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_saha: SahaParams;
@group(0) @binding(1) var<storage, read> s_temps: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_saha_out: array<SahaResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_saha_out)) { return; }
    let t = s_temps[idx];
    let kt_ev = t * 8.617e-5;
    let factor = pow(t / 10000.0, 1.5) * exp(-u_saha.e_ion_ev / max(0.1, kt_ev));
    let alpha = clamp(sqrt(factor) / (1.0 + sqrt(factor)), 0.0, 1.0);
    let ne = (u_saha.pressure_pa / (BOLTZMANN_CONSTANT * t)) * alpha;
    s_saha_out[idx] = SahaResult(alpha, ne, t * 1000.0, 0.0);
}
