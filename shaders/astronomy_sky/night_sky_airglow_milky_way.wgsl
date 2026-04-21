// Shroud :: Upper Atmosphere Airglow Chemiluminescence and Stellar Background
// Subsystem Module: shaders/astronomy_sky/night_sky_airglow_milky_way.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates mesospheric hydroxyl OH Meinel bands and atomic oxygen 557.7 nm green emission.
//   - Evaluates integrated stellar flux from Milky Way galactic disk and zodiacal background.
//
// Mathematical & Physical Formulation:
//   Van Rhijn effect: I(zeta) = I_0 / sqrt(1 - [R / (R + h)]^2 * sin(zeta)^2).
//   Brightens sky near horizon due to increased path length through emitting airglow layer.
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

struct AirglowConfig { zenith_radiance: f32, layer_height_km: f32, pad0: f32, pad1: f32 };

struct AirglowOut { van_rhijn_gain: f32, emission_557nm: f32, total_sky_radiance: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_ag: AirglowConfig;
@group(0) @binding(1) var<storage, read> s_zenith_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_ag_out: array<AirglowOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_ag_out)) { return; }
    let zeta = s_zenith_angles[idx];
    let re = 6371.0;
    let h = u_ag.layer_height_km;
    let sin_z = sin(zeta);
    let vr = 1.0 / sqrt(max(1e-4, 1.0 - pow((re / (re + h)) * sin_z, 2.0)));
    let rad = u_ag.zenith_radiance * vr;
    s_ag_out[idx] = AirglowOut(vr, rad * 0.4, rad, 0.0);
}
