// Shroud :: Hypersonic Re-Entry Shock Plasma Sheath RF Attenuation
// Subsystem Module: shaders/plasma_aerodynamics/hypersonic_plasma_sheath_attenuation.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates electron plasma frequency omega_p and collision frequency nu_en.
//   - Calculates electromagnetic radio wave transmission loss and telemetry blackout.
//
// Mathematical & Physical Formulation:
//   omega_p^2 = (n_e * e^2) / (m_e * epsilon_0).
//   alpha = (omega / c) * sqrt(0.5 * (sqrt(1 + (nu/omega)^2) - 1)) attenuation constant.
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
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct PlasmaParams { n_e_m3: f32, nu_en_hz: f32, sheath_thickness_m: f32, rf_freq_ghz: f32 };

struct RfAttenuationOut { attenuation_db: f32, omega_p_rad_s: f32, is_blackout: u32, pad: f32 };

@group(0) @binding(0) var<uniform> u_plas: PlasmaParams;
@group(0) @binding(1) var<storage, read> s_plasma_cells: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_atten_out: array<RfAttenuationOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_atten_out)) { return; }
    let ne = u_plas.n_e_m3 * s_plasma_cells[idx];
    let wp2 = (ne * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE) / (ELECTRON_MASS * 8.854e-12);
    let wp = sqrt(max(0.0, wp2));
    let w_rf = TWO_PI * u_plas.rf_freq_ghz * 1e9;
    let atten_np = (wp2 / max(1e6, w_rf * w_rf + u_plas.nu_en_hz * u_plas.nu_en_hz)) * (u_plas.nu_en_hz / (2.0 * SPEED_OF_LIGHT));
    let atten_db = atten_np * u_plas.sheath_thickness_m * 8.686;
    s_atten_out[idx] = RfAttenuationOut(atten_db, wp, select(0u, 1u, w_rf < wp), 0.0);
}
