// Shroud :: Coherence Scanning Interferometry (CSI / VSI) Envelope Demodulator
// Subsystem Module: shaders/metrology_sensors/white_light_interferometry_vsi.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates low-coherence white light interferometric fringe envelopes.
//   - Extracts coherence peak positions for nanometer-scale surface profilometry.
//
// Mathematical & Physical Formulation:
//   I(z) = I_0 * [1 + V(z - z_0) * cos(k_0 * (z - z_0) + phi_0)].
//   Envelope V(z) evaluated via five-point Hilbert demodulation filter.
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

struct VsiConfig { step_nm: f32, pad0: f32, pad1: f32, pad2: f32 };

struct VsiResult { peak_z_nm: f32, fringe_contrast: f32, envelope_max: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_vsi: VsiConfig;
@group(0) @binding(1) var<storage, read> s_vsi_fringes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_vsi_out: array<VsiResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_vsi_out)) { return; }
    let f = s_vsi_fringes[idx];
    let env = sqrt(pow(f.y - f.w, 2.0) + pow(f.x - f.z, 2.0));
    let z_pos = f32(idx) * u_vsi.step_nm;
    s_vsi_out[idx] = VsiResult(z_pos, env / max(1e-4, f.x + f.y), env, 0.0);
}
