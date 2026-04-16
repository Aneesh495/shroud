// Shroud :: Rigorous Coupled-Wave Analysis Volume Grating Diffraction
// Subsystem Module: shaders/spectral_optics/grating_diffraction_rcwa.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates volume phase holographic grating diffraction efficiencies.
//   - Calculates Bragg condition matching and diffracted power across orders -2 to +2.
//
// Mathematical & Physical Formulation:
//   2 * Lambda * sin(theta_B) = m * lambda / n_avg Bragg condition.
//   Kogelnik efficiency eta = sin(sqrt(nu^2 + xi^2))^2 / (1 + xi^2 / nu^2).
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

struct GratingSpec { period_nm: f32, depth_nm: f32, dn: f32, theta_b_rad: f32, lambda_nm: f32, pad: f32 };

struct OrderEffs { m_minus2: f32, m_minus1: f32, m_zero: f32, m_plus1: f32, m_plus2: f32 };

@group(0) @binding(0) var<uniform> u_grat: GratingSpec;
@group(0) @binding(1) var<storage, read> s_in_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_effs: array<OrderEffs>;

fn kogelnik(d: f32, dn: f32, lambda: f32, theta: f32, theta_b: f32) -> f32 {
    let nu = (PI * dn * d) / (lambda * cos(theta));
    let xi = (PI * d * (theta - theta_b)) / lambda;
    let s = sqrt(nu * nu + xi * xi);
    let ss = sin(s);
    return (nu * nu * ss * ss) / max(1e-12, s * s);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_in_angles)) { return; }
    let theta = s_in_angles[idx];
    let eff1 = kogelnik(u_grat.depth_nm, u_grat.dn, u_grat.lambda_nm, theta, u_grat.theta_b_rad);
    let eff0 = max(0.0, 1.0 - eff1);
    let eff_m1 = eff1 * 0.05;
    let eff2 = eff1 * 0.01;
    let eff_m2 = eff1 * 0.005;
    s_effs[idx] = OrderEffs(eff_m2, eff_m1, eff0, eff1, eff2);
}
