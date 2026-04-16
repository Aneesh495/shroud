// Shroud :: Anisotropic Uniaxial Crystal Birefringence Ray Splitting
// Subsystem Module: shaders/wave_optics/birefringence_uniaxial.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates ordinary (o) and extraordinary (e) ray propagation in uniaxial optical crystals.
//   - Calculates walk-off angle delta and optical path retardation.
//
// Mathematical & Physical Formulation:
//   1 / n_e(theta)^2 = cos(theta)^2 / n_o^2 + sin(theta)^2 / n_e^2
//   tan(psi) = (n_o^2 / n_e^2) * tan(theta) defines extraordinary energy flow Poynting vector.
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

struct CrystalData { no: f32, ne: f32, thickness_m: f32, wavelength_nm: f32, axis: vec4<f32> };

struct BifurcatedRay { ray_o: vec4<f32>, ray_e: vec4<f32>, walkoff_rad: f32, retardation_rad: f32 };

@group(0) @binding(0) var<uniform> u_crystal: CrystalData;
@group(0) @binding(1) var<storage, read> s_in_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_bifurcated: array<BifurcatedRay>;

fn calc_ne(no: f32, ne: f32, cos_t: f32) -> f32 {
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let inv = (cos_t * cos_t) / (no * no) + (sin_t * sin_t) / (ne * ne);
    return 1.0 / sqrt(max(1e-6, inv));
}

fn calc_walkoff(no: f32, ne: f32, cos_t: f32) -> f32 {
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let tan_t = sin_t / max(1e-4, cos_t);
    let tan_psi = (no * no) / (ne * ne) * tan_t;
    return abs(atan(tan_t) - atan(tan_psi));
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_in_rays)) { return; }
    let ray = normalize(s_in_rays[idx].xyz);
    let axis = normalize(u_crystal.axis.xyz);
    let cos_t = clamp(abs(dot(ray, axis)), 0.0, 1.0);
    let ne_eff = calc_ne(u_crystal.no, u_crystal.ne, cos_t);
    let walkoff = calc_walkoff(u_crystal.no, u_crystal.ne, cos_t);
    let cross_v = cross(ray, axis);
    let perp = normalize(cross_v + vec3<f32>(1e-5, 0.0, 0.0));
    let ray_e = normalize(ray * cos(walkoff) + cross(perp, ray) * sin(walkoff));
    let lambda_m = u_crystal.wavelength_nm * 1e-9;
    let delta_phase = TWO_PI * (u_crystal.thickness_m * abs(ne_eff - u_crystal.no)) / lambda_m;
    s_bifurcated[idx] = BifurcatedRay(vec4<f32>(ray, u_crystal.no), vec4<f32>(ray_e, ne_eff), walkoff, delta_phase);
}
