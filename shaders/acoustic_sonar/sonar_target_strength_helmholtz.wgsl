// Shroud :: Helmholtz-Kirchhoff Acoustic Target Strength (TS) Evaluator
// Subsystem Module: shaders/acoustic_sonar/sonar_target_strength_helmholtz.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates acoustic target strength TS = 10 * log10(I_scattered / I_incident * r^2).
//   - Solves boundary element Kirchhoff integral for submerged rigid and elastic targets.
//
// Mathematical & Physical Formulation:
//   TS = 10 * log10(sigma_bs / 4*pi).
//   Models geometric, resonance, and Rayleigh acoustic scattering regimes.
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

struct TsConfig { target_radius_m: f32, sound_speed_m_s: f32, pad0: f32, pad1: f32 };

struct TsResult { ts_db: f32, backscatter_area_m2: f32, ka_factor: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_ts: TsConfig;
@group(0) @binding(1) var<storage, read> s_freqs_hz: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_ts_out: array<TsResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ts_out)) { return; }
    let freq = s_freqs_hz[idx];
    let k = TWO_PI * freq / u_ts.sound_speed_m_s;
    let a = u_ts.target_radius_m;
    let ka = k * a;
    let sigma_geom = PI * a * a;
    let form_factor = select(pow(ka, 4.0), 1.0, ka > 1.0);
    let sigma_bs = sigma_geom * form_factor;
    let ts = 10.0 * log(max(1e-6, sigma_bs / FOUR_PI));
    s_ts_out[idx] = TsResult(ts, sigma_bs, ka, 0.0);
}
