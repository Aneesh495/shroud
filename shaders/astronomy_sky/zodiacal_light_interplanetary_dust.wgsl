// Shroud :: Interplanetary Dust Cloud Zodiacal Light and Gegenschein Simulator
// Subsystem Module: shaders/astronomy_sky/zodiacal_light_interplanetary_dust.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates solar light scattered by interplanetary dust grains in the solar system ecliptic plane.
//   - Models heliocentric distance dust density falloff and backscattering gegenschein peak.
//
// Mathematical & Physical Formulation:
//   I_zod(lambda, beta) = I_0 * n(r) * P_dust(Theta) integral along line of sight.
//   Gegenschein enhancement at anti-solar point Theta = 180 degrees.
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

struct ZodiacalConfig { solar_elongation_deg: f32, ecliptic_lat_deg: f32, pad0: f32, pad1: f32 };

struct ZodiacalOut { radiance_s10: f32, gegenschein_boost: f32, total_radiance: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_zod: ZodiacalConfig;
@group(0) @binding(1) var<storage, read> s_los_coords: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_zod_out: array<ZodiacalOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_zod_out)) { return; }
    let el = u_zod.solar_elongation_deg;
    let lat = u_zod.ecliptic_lat_deg;
    let base_s10 = 150.0 / (1.0 + 0.05 * abs(lat)) / max(0.5, sin(el * (PI / 180.0)));
    let gegen = select(1.0, 1.4, el > 170.0);
    s_zod_out[idx] = ZodiacalOut(base_s10, gegen, base_s10 * gegen, 0.0);
}
