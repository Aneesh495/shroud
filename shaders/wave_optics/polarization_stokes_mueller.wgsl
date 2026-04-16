// Shroud :: Polarization Vector Transport via Stokes-Mueller Calculus
// Subsystem Module: shaders/wave_optics/polarization_stokes_mueller.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates 4-vector Stokes radiative transfer through birefringent optical trains.
//   - Computes Degree of Linear Polarization (DoLP) and Angle of Linear Polarization (AoLP).
//
// Mathematical & Physical Formulation:
//   S' = M * S where S = [S0, S1, S2, S3]^T and M is a 4x4 Mueller matrix.
//   DoLP = sqrt(S1^2 + S2^2) / S0, AoLP = 0.5 * atan2(S2, S1).
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

struct StokesVec { s0: f32, s1: f32, s2: f32, s3: f32 };

struct MuellerMat { m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32>, m3: vec4<f32> };

struct PolarMetrics { dolp: f32, docp: f32, aolp_deg: f32, dop: f32 };

@group(0) @binding(0) var<storage, read> s_in_stokes: array<StokesVec>;
@group(0) @binding(1) var<storage, read> s_mueller: array<MuellerMat>;
@group(0) @binding(2) var<storage, read_write> s_out_stokes: array<StokesVec>;
@group(0) @binding(3) var<storage, read_write> s_metrics: array<PolarMetrics>;

fn mul_ms(m: MuellerMat, s: StokesVec) -> StokesVec {
    let sv = vec4<f32>(s.s0, s.s1, s.s2, s.s3);
    let o0 = dot(m.m0, sv);
    let o1 = dot(m.m1, sv);
    let o2 = dot(m.m2, sv);
    let o3 = dot(m.m3, sv);
    return StokesVec(o0, o1, o2, o3);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_in_stokes)) { return; }
    let in_s = s_in_stokes[idx];
    let m = s_mueller[idx % arrayLength(&s_mueller)];
    let out_s = mul_ms(m, in_s);
    let s0 = max(1e-6, out_s.s0);
    let dolp = clamp(sqrt(out_s.s1 * out_s.s1 + out_s.s2 * out_s.s2) / s0, 0.0, 1.0);
    let docp = clamp(abs(out_s.s3) / s0, 0.0, 1.0);
    let dop = clamp(sqrt(out_s.s1 * out_s.s1 + out_s.s2 * out_s.s2 + out_s.s3 * out_s.s3) / s0, 0.0, 1.0);
    let aolp = (0.5 * atan2(out_s.s2, out_s.s1) * (180.0 / PI) + 180.0) % 180.0;
    s_out_stokes[idx] = out_s;
    s_metrics[idx] = PolarMetrics(dolp, docp, aolp, dop);
}
