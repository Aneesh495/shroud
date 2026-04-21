// Shroud :: Picosecond Electrostatic Sweep Deflection Streak Camera Simulator
// Subsystem Module: shaders/quantum_photodetectors/streak_camera_femtosecond.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Maps ultrafast temporal light profiles into spatial sensor coordinates.
//   - Applies high-voltage fast linear ramp voltage sweep across electrostatic deflection plates.
//
// Mathematical & Physical Formulation:
//   Delta_y(t) = (e * V_sweep * L * D) / (2 * m_e * v_z^2 * d) * t temporal-to-spatial mapping.
//   Enables sub-picosecond temporal resolution pulse characterization.
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

struct StreakParams { sweep_speed_mm_ns: f32, temporal_dispersion_ps: f32, pad0: f32, pad1: f32 };

struct StreakResult { sensor_y_mm: f32, intensity: f32, time_ps: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_st: StreakParams;
@group(0) @binding(1) var<storage, read> s_pulse_samples: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_st_out: array<StreakResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_st_out)) { return; }
    let p = s_pulse_samples[idx];
    let t_ps = p.x;
    let y_mm = (t_ps * 1e-3) * u_st.sweep_speed_mm_ns;
    s_st_out[idx] = StreakResult(y_mm, p.y, t_ps, 0.0);
}
