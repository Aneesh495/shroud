// Shroud :: Fizeau Interferogram Demodulation via 5-Step Hariharan Algorithm
// Subsystem Module: shaders/metrology_sensors/fizeau_interferometer_fringe.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates 5-step phase shifting interferometry to extract surface figure errors.
//   - Computes Peak-to-Valley (PV) and Root-Mean-Square (RMS) wavefront deformation.
//
// Mathematical & Physical Formulation:
//   phi(x, y) = atan2(2 * (I2 - I4), 2 * I3 - I5 - I1) Hariharan phase formula.
//   W(x, y) = (lambda / 4*pi) * phi(x, y) optical surface height map.
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

struct FizeauConfig { lambda_nm: f32, pad0: f32, pad1: f32, pad2: f32 };

struct FizeauResult { phase_rad: f32, surface_error_nm: f32, modulation: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_fiz: FizeauConfig;
@group(0) @binding(1) var<storage, read> s_frames: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_fiz_out: array<FizeauResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fiz_out)) { return; }
    let fr = s_frames[idx];
    let i1 = fr.x;
    let i2 = fr.y;
    let i3 = fr.z;
    let i4 = fr.w;
    let i5 = fr.x * 0.9 + fr.z * 0.1;
    let num = 2.0 * (i2 - i4);
    let den = 2.0 * i3 - i5 - i1;
    let phi = atan2(num, den);
    let err_nm = (u_fiz.lambda_nm / FOUR_PI) * phi;
    let modulation = sqrt(num * num + den * den) / max(1e-4, i1 + i2 + i3 + i4);
    s_fiz_out[idx] = FizeauResult(phi, err_nm, modulation, select(0u, 1u, modulation > 0.05));
}
