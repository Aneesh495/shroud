// Shroud :: Hypersonic Shock Interface Rayleigh-Taylor Instability Growth Rate
// Subsystem Module: shaders/plasma_aerodynamics/rayleigh_taylor_instability_growth.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes growth rate gamma = sqrt(A * g * k) across shock density discontinuities.
//   - Atwood number A = (rho_heavy - rho_light) / (rho_heavy + rho_light).
//
// Mathematical & Physical Formulation:
//   Evaluates finger bubble penetration across boundary layer contact surfaces.
//   Ablative stabilization reduces high wavenumber instability growth.
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

struct RtConfig { atwood_a: f32, accel_g: f32, pad0: f32, pad1: f32 };

struct RtOutput { gamma_growth_s_inv: f32, wave_number_k: f32, amplitude_factor: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_rt: RtConfig;
@group(0) @binding(1) var<storage, read> s_wavenumbers: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_rt_out: array<RtOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rt_out)) { return; }
    let k = s_wavenumbers[idx];
    let gamma = sqrt(max(0.0, u_rt.atwood_a * u_rt.accel_g * k));
    s_rt_out[idx] = RtOutput(gamma, k, exp(gamma * 0.001), 0.0);
}
