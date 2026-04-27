// Shroud :: Thorp Acoustic Attenuation and Oceanic Transmission Loss
// Subsystem Module: shaders/acoustic_sonar/sonar_transmission_loss_thorp.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates spherical spreading loss plus seawater chemical absorption alpha(f).
//   - Models boric acid B(OH)3 (< 1 kHz) and magnesium sulfate MgSO4 (1-100 kHz) relaxation.
//
// Mathematical & Physical Formulation:
//   alpha(f) = 0.11 * f^2 / (1 + f^2) + 44 * f^2 / (4100 + f^2) + 2.75e-4 * f^2 + 0.003 dB/km.
//   Transmission Loss TL = 20 * log10(R) + alpha * R * 1e-3.
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

struct ThorpConfig { freq_khz: f32, depth_m: f32, salinity_psu: f32, pad: f32 };

struct TransLossResult { tl_db: f32, alpha_db_km: f32, spreading_loss_db: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_thorp: ThorpConfig;
@group(0) @binding(1) var<storage, read> s_ranges_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_tl_out: array<TransLossResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_tl_out)) { return; }
    let r_m = s_ranges_m[idx];
    let f = u_thorp.freq_khz;
    let f2 = f * f;
    let a_boric = (0.11 * f2) / (1.0 + f2);
    let a_mgso4 = (44.0 * f2) / (4100.0 + f2);
    let a_pure = 2.75e-4 * f2 + 0.003;
    let alpha = a_boric + a_mgso4 + a_pure;
    let spread = 20.0 * log(max(1.0, r_m));
    let tl = spread + alpha * (r_m * 1e-3);
    s_tl_out[idx] = TransLossResult(tl, alpha, spread, 0.0);
}
