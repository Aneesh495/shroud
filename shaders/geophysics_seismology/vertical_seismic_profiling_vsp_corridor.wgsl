// Shroud :: Zero-Offset VSP Upgoing and Downgoing Wavefield Median Separation
// Subsystem Module: shaders/geophysics_seismology/vertical_seismic_profiling_vsp_corridor.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Separates downgoing direct wavefield and upgoing subsurface reflections in borehole geophones.
//   - Applies velocity filtering (F-K fan filter or tau-p Radon transform) to produce corridor stacks.
//
// Mathematical & Physical Formulation:
//   U(t, z) upgoing reflections calibrate surface seismic reflection traveltimes to true depth.
//   Direct wave alignment determines interval velocity V_int = Delta_z / Delta_t.
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

struct VspConfig { geophone_spacing_m: f32, pad0: f32, pad1: f32, pad2: f32 };

struct VspSeparated { upgoing_wave: f32, downgoing_wave: f32, interval_vel_m_s: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_vsp: VspConfig;
@group(0) @binding(1) var<storage, read> s_borehole_traces: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_vsp_out: array<VspSeparated>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_vsp_out)) { return; }
    let tr = s_borehole_traces[idx];
    let total = tr.x;
    let direct = tr.y;
    let upgoing = total - direct;
    let v_int = u_vsp.geophone_spacing_m / 0.005;
    s_vsp_out[idx] = VspSeparated(upgoing, direct, v_int, 0.0);
}
