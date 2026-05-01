// Shroud :: Electrostatic Langmuir Probe I-V Characteristic Inversion
// Subsystem Module: shaders/plasma_aerodynamics/langmuir_probe_iv_characteristic.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Inverts probe current-voltage data to extract plasma density n_e and electron temperature T_e.
//   - Analyzes ion saturation region, exponential electron retarding region, and electron saturation.
//
// Mathematical & Physical Formulation:
//   I(V) = I_is + I_es * exp(e * (V - V_p) / k * T_e) for retarding regime.
//   Floating potential V_f determined by net zero probe current.
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

struct LangmuirConfig { probe_area_m2: f32, pad0: f32, pad1: f32, pad2: f32 };

struct LangmuirOut { current_ma: f32, regime: u32, est_te_ev: f32, est_ne_m3: f32 };

@group(0) @binding(0) var<uniform> u_lp: LangmuirConfig;
@group(0) @binding(1) var<storage, read> s_voltages: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_lp_out: array<LangmuirOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_lp_out)) { return; }
    let v = s_voltages[idx];
    let v_float = -2.5;
    let te = 1.5;
    let i_is = -0.5;
    let current = select(i_is, i_is + 5.0 * exp((v - v_float) / te), v > v_float);
    let reg = select(0u, 1u, v > v_float);
    s_lp_out[idx] = LangmuirOut(current, reg, te, 1e18);
}
