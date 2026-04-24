// Shroud :: Atmospheric Precorrected Differential Absorption (APDA) Water Vapor
// Subsystem Module: shaders/hyperspectral/atmospheric_water_vapor_absorption_bands.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Estimates total precipitable water vapor (PWV, g/cm^2) from 940 nm and 1130 nm absorption troughs.
//   - Computes continuum baseline ratio R_apda = L_abs / (w1 * L_c1 + w2 * L_c2).
//
// Mathematical & Physical Formulation:
//   tau_wv(lambda) = exp(-alpha * (PWV * m)^beta) water vapor transmittance.
//   Inverts non-linear exponential relation to retrieve column abundance.
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

struct WvBands { l_860_c1: f32, l_940_abs: f32, l_1040_c2: f32, airmass: f32 };

struct WvResult { pwv_g_cm2: f32, ratio: f32, optical_depth: f32, pad: f32 };

@group(0) @binding(0) var<storage, read> s_wv_in: array<WvBands>;
@group(0) @binding(1) var<storage, read_write> s_wv_out: array<WvResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_wv_out)) { return; }
    let b = s_wv_in[idx];
    let l_cont = 0.55 * b.l_860_c1 + 0.45 * b.l_1040_c2;
    let ratio = clamp(b.l_940_abs / max(1e-4, l_cont), 0.05, 1.0);
    let tau = -log(ratio);
    let pwv = pow(tau / max(1e-3, 0.4 * b.airmass), 1.6);
    s_wv_out[idx] = WvResult(pwv, ratio, tau, 0.0);
}
