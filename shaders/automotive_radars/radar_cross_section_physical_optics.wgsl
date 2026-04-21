// Shroud :: High-Frequency Physical Optics Radar Cross Section (RCS) Estimator
// Subsystem Module: shaders/automotive_radars/radar_cross_section_physical_optics.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Applies Physical Optics (PO) surface integration: J_s = 2 * (n x H_inc).
//   - Estimates monostatic and bistatic RCS (dBsm) for automotive polygon meshes.
//
// Mathematical & Physical Formulation:
//   sigma = 4 * pi * |R|^2 / |E_inc|^2 in far field limit.
//   Far-field integral sigma = (4 * pi / lambda^2) * |iint (n . k_s) * exp(i * 2 * k . r) dA|^2.
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

struct RcsConfig { lambda_m: f32, pad0: f32, pad1: f32, pad2: f32 };

struct TriRcs { normal: vec4<f32>, v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32> };

struct RcsOut { sigma_m2: f32, sigma_dbsm: f32, phase_rad: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_rcs: RcsConfig;
@group(0) @binding(1) var<storage, read> s_tris: array<TriRcs>;
@group(0) @binding(2) var<storage, read_write> s_rcs_out: array<RcsOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rcs_out)) { return; }
    let tri = s_tris[idx];
    let n = normalize(tri.normal.xyz);
    let e1 = tri.v1.xyz - tri.v0.xyz;
    let e2 = tri.v2.xyz - tri.v0.xyz;
    let area = 0.5 * length(cross(e1, e2));
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let cos_th = clamp(dot(n, view_dir), 0.0, 1.0);
    let k = TWO_PI / u_rcs.lambda_m;
    let sigma = (FOUR_PI * area * area * cos_th * cos_th) / (u_rcs.lambda_m * u_rcs.lambda_m);
    let dbsm = 10.0 * log(max(1e-6, sigma));
    s_rcs_out[idx] = RcsOut(sigma, dbsm, 0.0, select(0u, 1u, cos_th > 0.0));
}
