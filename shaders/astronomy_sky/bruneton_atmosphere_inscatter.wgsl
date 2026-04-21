// Shroud :: Multiple Scattering Atmospheric Angular Inscatter Operator
// Subsystem Module: shaders/astronomy_sky/bruneton_atmosphere_inscatter.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates single and multiple scattering angular phase functions across atmospheric layers.
//   - Combines Rayleigh molecular dipolar scattering with Mie aerosol forward lobe.
//
// Mathematical & Physical Formulation:
//   p_ray(cos_th) = (3 / 16*pi) * (1 + cos_th^2).
//   p_mie(cos_th) Cornette-Shanks or Henyey-Greenstein representation.
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

struct InscatterConfig { mie_g: f32, pad0: f32, pad1: f32, pad2: f32 };

struct InscatterOut { s_ray: f32, s_mie: f32, combined_radiance: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_in: InscatterConfig;
@group(0) @binding(1) var<storage, read> s_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_inscat_out: array<InscatterOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_inscat_out)) { return; }
    let cos_t = s_angles[idx];
    let p_ray = (3.0 / 16.0 * INV_PI) * (1.0 + cos_t * cos_t);
    let g = u_in.mie_g;
    let p_mie = (1.5 * INV_PI) * (1.0 - g * g) / (2.0 + g * g) * (1.0 + cos_t * cos_t) / pow(1.0 + g * g - 2.0 * g * cos_t, 1.5);
    let rad = p_ray * 0.7 + p_mie * 0.3;
    s_inscat_out[idx] = InscatterOut(p_ray, p_mie, rad, 0.0);
}
