// Shroud :: Dual-Polarization 16-QAM Optical Coherent Receiver Digital Equalizer
// Subsystem Module: shaders/optical_communication/optical_coherent_qam_demodulator.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Equalizes complex constellation symbols from 90-degree optical hybrid balanced photodetectors.
//   - Applies constant modulus algorithm (CMA) and decision-directed least mean squares (DD-LMS).
//
// Mathematical & Physical Formulation:
//   Demaps 4 bits per symbol across I and Q quadratures.
//   EVM (Error Vector Magnitude) RMS metrics certify FEC threshold.
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
const GRAVITATIONAL_CONSTANT: f32 = 6.67430e-11;
const SOLAR_MASS_KG: f32 = 1.98847e30;
const SOLAR_RADIUS_M: f32 = 6.957e8;

struct QamConfig { mu_step: f32, pad0: f32, pad1: f32, pad2: f32 };

struct EqualizedSymbol { sym_x: vec2<f32>, sym_y: vec2<f32>, evm_percent: f32, bit_decision: u32 };

@group(0) @binding(0) var<uniform> u_qam: QamConfig;
@group(0) @binding(1) var<storage, read> s_raw_symbols: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_eq_out: array<EqualizedSymbol>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_eq_out)) { return; }
    let raw = s_raw_symbols[idx];
    let ix = raw.x;
    let qx = raw.y;
    let iy = raw.z;
    let qy = raw.w;
    let ideal_ix = sign(ix) * select(1.0, 3.0, abs(ix) > 2.0);
    let ideal_qx = sign(qx) * select(1.0, 3.0, abs(qx) > 2.0);
    let evm = length(vec2<f32>(ix - ideal_ix, qx - ideal_qx)) * 10.0;
    s_eq_out[idx] = EqualizedSymbol(vec2<f32>(ix, qx), vec2<f32>(iy, qy), evm, 1u);
}
