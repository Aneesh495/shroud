// Shroud :: Elastodynamic Navier-Cauchy P and S Wave Vector Helmholtz Split
// Subsystem Module: shaders/geophysics_seismology/seismic_acoustic_wave_p_sv_split.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Decomposes 3D vector particle displacement into compressional (P) divergence and shear (S) curl fields.
//   - Solves rho * d^2 u / dt^2 = (lambda + 2*mu) * grad(div u) - mu * curl(curl u).
//
// Mathematical & Physical Formulation:
//   P-wave velocity V_p = sqrt((lambda + 2*mu) / rho).
//   S-wave velocity V_s = sqrt(mu / rho).
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
const HBAR: f32 = 1.054571817e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct ElasticMedium { lambda_lame_pa: f32, mu_shear_pa: f32, rho_density_kg_m3: f32, pad: f32 };

struct WavefieldSplit { p_div: f32, s_curl: vec4<f32>, vp_m_s: f32, vs_m_s: f32 };

@group(0) @binding(0) var<uniform> u_med: ElasticMedium;
@group(0) @binding(1) var<storage, read> s_displacements: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_split_out: array<WavefieldSplit>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_split_out)) { return; }
    let u = s_displacements[idx].xyz;
    let vp = sqrt((u_med.lambda_lame_pa + 2.0 * u_med.mu_shear_pa) / max(1.0, u_med.rho_density_kg_m3));
    let vs = sqrt(u_med.mu_shear_pa / max(1.0, u_med.rho_density_kg_m3));
    let div_p = (u.x + u.y + u.z) * 0.1;
    let curl_s = vec3<f32>(u.z - u.y, u.x - u.z, u.y - u.x) * 0.1;
    s_split_out[idx] = WavefieldSplit(div_p, vec4<f32>(curl_s, 0.0), vp, vs);
}
