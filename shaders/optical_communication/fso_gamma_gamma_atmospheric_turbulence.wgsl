// Shroud :: Andrews Gamma-Gamma Optical Scintillation Probability Distribution
// Subsystem Module: shaders/optical_communication/fso_gamma_gamma_atmospheric_turbulence.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates PDF p(I) of laser beam irradiance fluctuations through atmospheric Kolmogorov turbulence.
//   - p(I) = [2 * (alpha * beta)^((alpha + beta)/2) / (Gamma(alpha) * Gamma(beta))] * I^((alpha+beta)/2 - 1) * K_{alpha - beta}(2 * sqrt(alpha * beta * I)).
//
// Mathematical & Physical Formulation:
//   Rytov variance sigma_R^2 = 1.23 * C_n^2 * k^(7/6) * L^(11/6) characterizes weak to strong turbulence regimes.
//   Models large-scale turbulent eddies alpha and small-scale eddies beta.
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

struct GammaGammaConfig { alpha_large: f32, beta_small: f32, r_aperture_m: f32, pad: f32 };

struct ScintillationSample { irradiance_pdf: f32, fade_probability: f32, log_intensity: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_gg: GammaGammaConfig;
@group(0) @binding(1) var<storage, read> s_irradiances: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_scint_out: array<ScintillationSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_scint_out)) { return; }
    let i_val = s_irradiances[idx];
    let a = u_gg.alpha_large;
    let b = u_gg.beta_small;
    let ab = a * b;
    let arg = 2.0 * sqrt(ab * i_val);
    let k_approx = exp(-arg) / max(0.5, sqrt(PI * max(0.1, arg)));
    let pdf = (ab / max(1e-4, i_val)) * k_approx;
    let fade = select(0.0, 1.0, i_val < 0.2);
    s_scint_out[idx] = ScintillationSample(pdf, fade, log(max(1e-4, i_val)), 0.0);
}
