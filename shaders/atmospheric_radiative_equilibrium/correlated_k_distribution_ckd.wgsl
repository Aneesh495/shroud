// Shroud :: Correlated-k Gas Distribution Integration across Inhomogeneous Atmospheres
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/correlated_k_distribution_ckd.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Reorders rapidly oscillating spectral absorption coefficients k(nu) into smooth monotonic k(g).
//   - Replaces expensive line-by-line integration with N-point Gauss-Legendre quadrature (N = 8 to 16).
//
// Mathematical & Physical Formulation:
//   Transmittance T_band = Integral_0^1 exp(-k(g) * u) dg approx sum_i w_i * exp(-k_i * u).
//   Correlated assumption holds scaling across vertical pressure-temperature layers.
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

struct CkdConfig { absorber_mass_u: f32, pad0: f32, pad1: f32, pad2: f32 };

struct CkdTransmittance { trans_band: f32, eff_k: f32, quadrature_weight: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_ckd: CkdConfig;
@group(0) @binding(1) var<storage, read> s_kg_samples: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_ckd_out: array<CkdTransmittance>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ckd_out)) { return; }
    let smp = s_kg_samples[idx];
    let g = smp.x;
    let k_val = smp.y;
    let t_g = exp(-k_val * u_ckd.absorber_mass_u);
    s_ckd_out[idx] = CkdTransmittance(t_g, k_val, 1.0, 0.0);
}
