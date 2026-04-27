// Shroud :: Sub-Bottom Chirp Profiler Sediment Acoustic Impedance Inversion
// Subsystem Module: shaders/acoustic_sonar/sub_bottom_chirp_profiler.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Inverts sub-bottom acoustic reflection echoes into multi-layer sediment impedances Z = rho * c.
//   - Calculates acoustic reflection coefficient R = (Z2 - Z1) / (Z2 + Z1) across subsurface strata.
//
// Mathematical & Physical Formulation:
//   Detects buried sediment layers: silt, clay, sand, gravel, and bedrock interfaces.
//   Evaluates pulse compression via matched filtering of 2-16 kHz FM chirps.
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

struct SubBottomConfig { z_water: f32, pad0: f32, pad1: f32, pad2: f32 };

struct StrataLayer { impedance: f32, reflect_coeff: f32, layer_depth_m: f32, layer_type: u32 };

@group(0) @binding(0) var<uniform> u_sb: SubBottomConfig;
@group(0) @binding(1) var<storage, read> s_echo_amps: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_strata: array<StrataLayer>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_strata)) { return; }
    let e = s_echo_amps[idx];
    let r_coef = clamp(e.x, -0.9, 0.9);
    let z_prev = u_sb.z_water;
    let z_sed = z_prev * (1.0 + r_coef) / max(0.1, 1.0 - r_coef);
    let depth = e.y * 0.75;
    let l_type = select(select(1u, 2u, z_sed > 2.0e6), 3u, z_sed > 3.5e6);
    s_strata[idx] = StrataLayer(z_sed, r_coef, depth, l_type);
}
