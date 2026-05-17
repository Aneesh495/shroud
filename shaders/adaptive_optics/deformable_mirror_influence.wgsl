// Shroud :: Continuous Facesheet Deformable Mirror Gaussian Influence Function Solver
// Subsystem Module: shaders/adaptive_optics/deformable_mirror_influence.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Models mechanical facesheet coupling across piezoelectric / MEMS actuator arrays.
//   * Applies Biharmonic plate influence kernel I(r) = exp(-alpha * (r / d)^beta) with inter-actuator coupling c.
//   * Computes per-actuator stroke clamping and mechanical stress saturation limits.
//
// Mathematical Formulation:
//   S(x, y) = sum_k a_k * exp(ln(c_coupling) * ((x - x_k)^2 + (y - y_k)^2) / d_pitch^2)
//   Surface wavefront correction W_corr = 2 * S(x, y) on reflection.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct DmParameters { actuator_pitch_mm: f32, coupling_coeff: f32, max_stroke_um: f32, exponent_beta: f32, actuator_count: u32, pad0: f32, pad1: f32, pad2: f32 };

struct ActuatorCmd { x_mm: f32, y_mm: f32, command_um: f32, pad: f32 };

struct DmSurfaceSample { height_um: f32, opd_um: f32, gradient_mag: f32, saturated: u32 };

@group(0) @binding(0) var<uniform> u_dm: DmParameters;
@group(0) @binding(1) var<storage, read> s_actuators: array<ActuatorCmd>;
@group(0) @binding(2) var<storage, read> s_mirror_coords: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> s_dm_surface: array<DmSurfaceSample>;

fn influence_kernel(dist_mm: f32, pitch_mm: f32, coupling: f32, beta: f32) -> f32 {
    let alpha = -log(clamp(coupling, 0.01, 0.95));
    let r_norm = dist_mm / max(1e-3, pitch_mm);
    return exp(-alpha * pow(r_norm, beta));
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_dm_surface)) { return; }
    let pt = s_mirror_coords[idx];
    var accum_h = 0.0;
    var sat_flag = 0u;
    let count = min(u_dm.actuator_count, u32(arrayLength(&s_actuators)));
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let act = s_actuators[k];
        let cmd = clamp(act.command_um, -u_dm.max_stroke_um, u_dm.max_stroke_um);
        if (abs(act.command_um) > u_dm.max_stroke_um) { sat_flag = 1u; }
        let d = length(pt - vec2<f32>(act.x_mm, act.y_mm));
        accum_h = accum_h + cmd * influence_kernel(d, u_dm.actuator_pitch_mm, u_dm.coupling_coeff, u_dm.exponent_beta);
    }
    s_dm_surface[idx] = DmSurfaceSample(accum_h, 2.0 * accum_h, abs(accum_h) / max(1e-3, u_dm.actuator_pitch_mm), sat_flag);
}
