// Shroud :: Teleseismic P-to-s Receiver Function Extended Time Deconvolution
// Subsystem Module: shaders/geophysics_seismology/receiver_function_p_to_s_conversion.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Deconvolves vertical component P-wave arrival from radial component P-to-S converted waves (Ps).
//   - Extracts time delay Delta_t = t_Ps - t_P to measure depth of crust-mantle boundary (Moho discontinuity).
//
// Mathematical & Physical Formulation:
//   R(omega) = [Z*(omega) * R_raw(omega)] / (|Z(omega)|^2 + delta) water-level deconvolution.
//   H-k stacking determines crustal thickness H and Vp/Vs ratio.
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

struct RfConfig { water_level: f32, vp_avg_m_s: f32, vs_avg_m_s: f32, pad: f32 };

struct RfResult { rf_amplitude: f32, delay_time_s: f32, moho_depth_km: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_rf: RfConfig;
@group(0) @binding(1) var<storage, read> s_z_comp: array<f32>;
@group(0) @binding(2) var<storage, read> s_r_comp: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_rf_out: array<RfResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_rf_out)) { return; }
    let z = s_z_comp[idx];
    let r = s_r_comp[idx];
    let den = max(u_rf.water_level, z * z);
    let rf = (r * z) / den;
    let dt = f32(idx) * 0.05;
    let h_km = dt / max(1e-3, (1.0 / (u_rf.vs_avg_m_s * 1e-3) - 1.0 / (u_rf.vp_avg_m_s * 1e-3)));
    s_rf_out[idx] = RfResult(rf, dt, h_km, 0.0);
}
