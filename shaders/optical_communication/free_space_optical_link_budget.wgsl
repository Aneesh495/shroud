// Shroud :: Free-Space Optical (FSO) Laser Link Budget and Geometric Loss
// Subsystem Module: shaders/optical_communication/free_space_optical_link_budget.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates satellite-to-ground optical communications link margin.
//   - P_rx = P_tx * G_tx * G_rx * (lambda / 4*pi*R)^2 * eta_tx * eta_rx * tau_atm.
//
// Mathematical & Physical Formulation:
//   Laser divergence theta_div determines receiver plane beam footprint diameter D_foot = R * theta_div.
//   Link Margin M_dB = P_rx_dBm - Sensitivity_dBm.
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

struct FsoParams { p_tx_watts: f32, lambda_nm: f32, divergence_urad: f32, rx_diameter_m: f32 };

struct FsoLinkOut { prx_dbm: f32, geom_loss_db: f32, link_margin_db: f32, footprint_m: f32 };

@group(0) @binding(0) var<uniform> u_fso: FsoParams;
@group(0) @binding(1) var<storage, read> s_ranges_km: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_fso_out: array<FsoLinkOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fso_out)) { return; }
    let r_km = s_ranges_km[idx];
    let r_m = r_km * 1000.0;
    let d_foot = r_m * (u_fso.divergence_urad * 1e-6);
    let a_foot = 0.25 * PI * d_foot * d_foot;
    let a_rx = 0.25 * PI * u_fso.rx_diameter_m * u_fso.rx_diameter_m;
    let geom_eff = min(1.0, a_rx / max(1e-4, a_foot));
    let p_rx_w = u_fso.p_tx_watts * geom_eff * 0.7;
    let prx_dbm = 10.0 * (log(max(1e-12, p_rx_w * 1000.0)) / log(10.0));
    s_fso_out[idx] = FsoLinkOut(prx_dbm, 10.0 * (log(geom_eff) / log(10.0)), prx_dbm - (-38.0), d_foot);
}
