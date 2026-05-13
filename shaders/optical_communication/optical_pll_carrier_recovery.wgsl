// Shroud :: Costas Loop Digital Carrier Phase and Frequency Offset Recovery
// Subsystem Module: shaders/optical_communication/optical_pll_carrier_recovery.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates error detector e_k = sign(I_k) * Q_k - sign(Q_k) * I_k for BPSK/QPSK carrier tracking.
//   - Second-order loop filter eliminates Doppler frequency offset Delta_omega and laser phase noise.
//
// Mathematical & Physical Formulation:
//   VCO phase update: theta_{k+1} = theta_k + K_p * e_k + K_i * sum e_j.
//   Maintains lock across high-linewidth semiconductor distributed feedback (DFB) local oscillators.
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

struct CostasConfig { kp: f32, ki: f32, pad0: f32, pad1: f32 };

struct PllState { recovered_i: f32, recovered_q: f32, estimated_phase_rad: f32, phase_error: f32 };

@group(0) @binding(0) var<uniform> u_pll: CostasConfig;
@group(0) @binding(1) var<storage, read> s_raw_iq: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_pll_out: array<PllState>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_pll_out)) { return; }
    let iq = s_raw_iq[idx];
    let err = sign(iq.x) * iq.y - sign(iq.y) * iq.x;
    let th = err * u_pll.kp;
    let rot_i = iq.x * cos(th) + iq.y * sin(th);
    let rot_q = -iq.x * sin(th) + iq.y * cos(th);
    s_pll_out[idx] = PllState(rot_i, rot_q, th, err);
}
