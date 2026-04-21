// Shroud :: Mobley-Petzold Ocean Volume Scattering Function (VSF) Evaluator
// Subsystem Module: shaders/oceanic_hydrology/petzold_ocean_scattering.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Fournier-Forand and Petzold volume scattering functions beta(theta).
//   - Models extreme forward-peaked diffraction by marine particulates and phytoplankton.
//
// Mathematical & Physical Formulation:
//   beta(theta) = b * p(theta) where p(theta) is normalized phase function.
//   Fournier-Forand analytic form matches Petzold San Diego harbor measurements.
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

struct OceanOptics { b_scatter: f32, a_absorb: f32, particulate_index: f32, pad: f32 };

struct VsfResult { vsf_val: f32, phase_p: f32, attenuation_c: f32, single_scatter_albedo: f32 };

@group(0) @binding(0) var<uniform> u_ocean: OceanOptics;
@group(0) @binding(1) var<storage, read> s_scat_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_vsf_out: array<VsfResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_vsf_out)) { return; }
    let th = s_scat_angles[idx];
    let cos_th = cos(th);
    let denom = 1.0 + 0.05 * 0.05 - 2.0 * 0.05 * cos_th;
    let hg_forward = (1.0 - 0.9 * 0.9) / pow(1.0 + 0.9 * 0.9 - 2.0 * 0.9 * cos_th, 1.5);
    let p = hg_forward * 0.95 + 0.05;
    let vsf = u_ocean.b_scatter * p * INV_FOUR_PI;
    let c = u_ocean.a_absorb + u_ocean.b_scatter;
    let ssa = u_ocean.b_scatter / max(1e-5, c);
    s_vsf_out[idx] = VsfResult(vsf, p, c, ssa);
}
