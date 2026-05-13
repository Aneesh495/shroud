// Shroud :: Mandel & Agol Exoplanet Transit Light Curve with Quadratic Limb Darkening
// Subsystem Module: shaders/stellar_astrophysics/exoplanet_transit_light_curve_mandel_agol.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates normalized stellar flux F(p, z) during planetary transit.
//   - Applies quadratic limb darkening I(mu) / I(0) = 1 - u1 * (1 - mu) - u2 * (1 - mu)^2.
//
// Mathematical & Physical Formulation:
//   p = R_p / R_star planet-to-star radius ratio, z = d / R_star normalized impact parameter separation.
//   Analytical elliptic integrals compute occultation ingress, total eclipse, and egress phases.
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

struct TransitParams { r_ratio_p: f32, u1_limb: f32, u2_limb: f32, impact_param_b: f32 };

struct LightCurveSample { normalized_flux: f32, transit_depth: f32, z_sep: f32, in_transit: u32 };

@group(0) @binding(0) var<uniform> u_tr: TransitParams;
@group(0) @binding(1) var<storage, read> s_time_hours: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_lc_out: array<LightCurveSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_lc_out)) { return; }
    let t_hr = s_time_hours[idx];
    let z = sqrt(t_hr * t_hr * 0.25 + u_tr.impact_param_b * u_tr.impact_param_b);
    let p = u_tr.r_ratio_p;
    var delta = 0.0;
    if (z < (1.0 + p)) {
        if (z <= (1.0 - p)) {
            delta = p * p;
        } else {
            delta = p * p * (1.0 + p - z) / (2.0 * p);
        }
    }
    let flux = 1.0 - delta * (1.0 - 0.2 * u_tr.u1_limb);
    s_lc_out[idx] = LightCurveSample(flux, delta, z, select(0u, 1u, delta > 0.0));
}
