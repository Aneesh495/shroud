// Shroud :: Spectral Angle Mapper (SAM) N-Dimensional Vector Classifier
// Subsystem Module: shaders/hyperspectral/spectral_angle_mapper.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes spectral angle theta between pixel spectrum and reference library signatures.
//   - Provides illuminance-invariant material classification in hyperspectral space.
//
// Mathematical & Physical Formulation:
//   theta = acos((x . r) / (||x|| * ||r||)).
//   Insensitive to solar shading and topography-induced radiance scaling.
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

struct SamRefVector { r0: f32, r1: f32, r2: f32, r3: f32 };

struct SamOutput { angle_rad: f32, angle_deg: f32, cosine_sim: f32, class_match: u32 };

@group(0) @binding(0) var<uniform> u_ref: SamRefVector;
@group(0) @binding(1) var<storage, read> s_pixels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_sam_out: array<SamOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sam_out)) { return; }
    let p = s_pixels[idx];
    let r = vec4<f32>(u_ref.r0, u_ref.r1, u_ref.r2, u_ref.r3);
    let num = dot(p, r);
    let den = max(1e-5, length(p) * length(r));
    let cos_th = clamp(num / den, -1.0, 1.0);
    let theta = acos(cos_th);
    let th_deg = theta * (180.0 / PI);
    s_sam_out[idx] = SamOutput(theta, th_deg, cos_th, select(0u, 1u, th_deg < 10.0));
}
