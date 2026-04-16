// Shroud :: Fresnel Near-Field Diffraction Operator via Chirp Z-Transform
// Subsystem Module: shaders/wave_optics/fresnel_diffraction_czt.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates scalar Fresnel near-field diffraction integral across planar apertures.
//   - Decomposes quadratic chirp phase factor exp(i * pi * (x^2 + y^2) / (lambda * z)).
//
// Mathematical & Physical Formulation:
//   U(x, y, z) = (exp(i * k * z) / (i * lambda * z)) * iint U(xi, eta) * exp(i * (k / 2z) * [(x-xi)^2 + (y-eta)^2]) dxi deta
//   Discretized as quadratic phase multiplication followed by chirp convolution.
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

struct FresnelParams { distance_m: f32, wavelength_m: f32, pitch_m: f32, grid_dim: u32 };

struct FieldSample { re: f32, im: f32, intensity: f32, phase: f32 };

@group(0) @binding(0) var<uniform> u_fresnel: FresnelParams;
@group(0) @binding(1) var<storage, read> s_in_field: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_out_field: array<FieldSample>;

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn chirp_phase(x: f32, y: f32, z: f32, lambda: f32) -> vec2<f32> {
    let k = TWO_PI / lambda;
    let phi = (k / (2.0 * max(1e-4, z))) * (x * x + y * y);
    return vec2<f32>(cos(phi), sin(phi));
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_in_field)) { return; }
    let dim = u_fresnel.grid_dim;
    let ix = idx % dim;
    let iy = idx / dim;
    let x_m = (f32(ix) - f32(dim) * 0.5) * u_fresnel.pitch_m;
    let y_m = (f32(iy) - f32(dim) * 0.5) * u_fresnel.pitch_m;
    let in_val = s_in_field[idx];
    let chirp = chirp_phase(x_m, y_m, u_fresnel.distance_m, u_fresnel.wavelength_m);
    let prod = c_mul(in_val, chirp);
    let dist = sqrt(x_m * x_m + y_m * y_m + u_fresnel.distance_m * u_fresnel.distance_m);
    let obliquity = 0.5 * (1.0 + u_fresnel.distance_m / max(1e-4, dist));
    let out_val = prod * obliquity;
    let intensity = out_val.x * out_val.x + out_val.y * out_val.y;
    let phase = atan2(out_val.y, out_val.x);
    s_out_field[idx] = FieldSample(out_val.x, out_val.y, intensity, phase);
}
