// Shroud :: Multi-Wavelength Ray Dispersion and Chromatic Aberration Analyzer
// Subsystem Module: shaders/spectral_optics/dispersion_chromatic_aberration.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Traces polychromatic ray bundles across multi-element refractive lens groups.
//   - Evaluates longitudinal chromatic aberration (LCA) and transverse chromatic aberration (TCA).
//
// Mathematical & Physical Formulation:
//   phi = (n - 1) * (c1 - c2) thin lens power.
//   LCA = f(lambda_F) - f(lambda_C), TCA = LCA * (y / f).
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

struct GlassData { nd: f32, nf: f32, nc: f32, abbe: f32, c1: f32, c2: f32, pad0: f32, pad1: f32 };

struct AberrationOut { focal_length: f32, lca: f32, tca: f32, achromat_err: f32 };

@group(0) @binding(0) var<uniform> u_glass: GlassData;
@group(0) @binding(1) var<storage, read> s_ray_heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_aber_out: array<AberrationOut>;

fn power(n: f32, c1: f32, c2: f32) -> f32 { return (n - 1.0) * (c1 - c2); }

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ray_heights)) { return; }
    let y = s_ray_heights[idx];
    let p_d = power(u_glass.nd, u_glass.c1, u_glass.c2);
    let p_f = power(u_glass.nf, u_glass.c1, u_glass.c2);
    let p_c = power(u_glass.nc, u_glass.c1, u_glass.c2);
    let f_d = 1.0 / max(1e-5, p_d);
    let f_f = 1.0 / max(1e-5, p_f);
    let f_c = 1.0 / max(1e-5, p_c);
    let lca = f_f - f_c;
    let tca = lca * (y / max(1e-4, f_d));
    let achromat = abs(p_d / u_glass.abbe);
    s_aber_out[idx] = AberrationOut(f_d, lca, tca, achromat);
}
