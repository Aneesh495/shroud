// Shroud :: Lambert Acoustic Backscattering and Seafloor Reverberation
// Subsystem Module: shaders/acoustic_sonar/sea_bottom_reverberation_lambert.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates seafloor acoustic backscattering strength S_b(theta) via Lambert's rule.
//   - Models seafloor sediment grain classifications: mud, fine sand, coarse gravel, rock.
//
// Mathematical & Physical Formulation:
//   S_b(theta) = 10 * log10(mu) + 10 * log10(sin(theta)^2).
//   Lambert scattering coefficient mu ranges from -30 dB (mud) to -10 dB (rock).
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

struct BottomParams { mu_db: f32, pad0: f32, pad1: f32, pad2: f32 };

struct BottomRevResult { backscatter_strength_db: f32, linear_coef: f32, grazing_rad: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_bot: BottomParams;
@group(0) @binding(1) var<storage, read> s_grazing_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_bot_out: array<BottomRevResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_bot_out)) { return; }
    let th = s_grazing_angles[idx];
    let sin_th = sin(th);
    let s_b = u_bot.mu_db + 20.0 * log(max(1e-4, sin_th));
    let lin_c = pow(10.0, s_b * 0.1);
    s_bot_out[idx] = BottomRevResult(s_b, lin_c, th, 0.0);
}
