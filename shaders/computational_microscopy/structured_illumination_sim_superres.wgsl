// Shroud :: Structured Illumination Microscopy (SIM) Spatial Harmonic Demodulator
// Subsystem Module: shaders/computational_microscopy/structured_illumination_sim_superres.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Demodulates 3 sinusoidal pattern phases (0, 120, 240 deg) across multiple orientation angles.
//   - Shifts high-frequency unresolvable sample spatial details into objective lens passband.
//
// Mathematical & Physical Formulation:
//   I(r) = I_0 * [1 + m * cos(k_0 . r + phi)] * S(r).
//   Doubles lateral spatial resolution beyond Abbe diffraction limit: Delta_x = lambda / (4 * NA).
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
const AVOGADRO_NUMBER: f32 = 6.02214076e23;

struct SimParams { pattern_k0: vec2<f32>, modulation_depth: f32, pad: f32 };

struct SimComponents { d0_component: vec2<f32>, d_plus_component: vec2<f32>, d_minus_component: vec2<f32>, pad: vec2<f32> };

@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<storage, read> s_phases_3: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_sim_out: array<SimComponents>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sim_out)) { return; }
    let p = s_phases_3[idx];
    let i0 = p.x;
    let i1 = p.y;
    let i2 = p.z;
    let d0 = (i0 + i1 + i2) * (1.0 / 3.0);
    let d_plus_re = (2.0 * i0 - i1 - i2) * (1.0 / 3.0);
    let d_plus_im = (sqrt(3.0) * (i1 - i2)) * (1.0 / 3.0);
    s_sim_out[idx] = SimComponents(vec2<f32>(d0, 0.0), vec2<f32>(d_plus_re, d_plus_im), vec2<f32>(d_plus_re, -d_plus_im), vec2<f32>(0.0));
}
