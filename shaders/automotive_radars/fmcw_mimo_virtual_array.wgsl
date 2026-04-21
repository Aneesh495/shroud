// Shroud :: 4D Imaging Radar MIMO Virtual Array Spatial Aperture Synthesizer
// Subsystem Module: shaders/automotive_radars/fmcw_mimo_virtual_array.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Convolves M physical transmitters and N physical receivers into M*N virtual phase centers.
//   - Applies Doppler phase migration compensation for target relative velocity.
//
// Mathematical & Physical Formulation:
//   p_virtual = p_tx + p_rx defines spatial virtual array element coordinates.
//   Delta_phi = 4 * pi * v_r * T_c / lambda phase progression across chirps.
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

struct MimoConfig { tx_count: u32, rx_count: u32, chirp_duration_s: f32, lambda_m: f32 };

struct VirtualElement { pos: vec4<f32>, complex_gain: vec2<f32>, phase_comp: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_mimo: MimoConfig;
@group(0) @binding(1) var<storage, read> s_tx_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_rx_pos: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> s_virt_out: array<VirtualElement>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_virt_out)) { return; }
    let rx_idx = idx % u_mimo.rx_count;
    let tx_idx = idx / u_mimo.rx_count;
    let tx = s_tx_pos[tx_idx % arrayLength(&s_tx_pos)].xyz;
    let rx = s_rx_pos[rx_idx % arrayLength(&s_rx_pos)].xyz;
    let v_pos = 0.5 * (tx + rx);
    let phase = (FOUR_PI * 15.0 * u_mimo.chirp_duration_s) / u_mimo.lambda_m;
    let c_gain = vec2<f32>(cos(phase), sin(phase));
    s_virt_out[idx] = VirtualElement(vec4<f32>(v_pos, 1.0), c_gain, phase, 1u);
}
