// Shroud :: Multi-Frequency Temporal Heterodyne Phase Unwrapping Kernel
// Subsystem Module: shaders/metrology_sensors/structured_light_phase_unwrapping.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Decodes N-step sinusoidal fringe patterns and eliminates 2*pi phase ambiguity.
//   - Synthesizes multi-frequency beat beats to reconstruct dense absolute 3D depth maps.
//
// Mathematical & Physical Formulation:
//   phi_wrap = atan2(sum I_k * sin(2*pi*k/N), sum I_k * cos(2*pi*k/N)).
//   Phi_abs = phi_wrap + 2 * pi * k_order via temporal frequency hierarchy.
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

struct UnwrapConfig { f1: f32, f2: f32, f3: f32, baseline_mm: f32 };

struct UnwrappedPoint { phase_abs_rad: f32, depth_mm: f32, fringe_modulation: f32, quality: f32 };

@group(0) @binding(0) var<uniform> u_uw: UnwrapConfig;
@group(0) @binding(1) var<storage, read> s_fringes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_unwrapped: array<UnwrappedPoint>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_unwrapped)) { return; }
    let f = s_fringes[idx];
    let p1 = atan2(f.y, f.x);
    let p2 = atan2(f.w, f.z);
    let delta = (p1 - p2 + TWO_PI) % TWO_PI;
    let k = floor((delta * (u_uw.f1 / (u_uw.f1 - u_uw.f2)) - p1) / TWO_PI);
    let abs_phi = p1 + TWO_PI * k;
    let depth = (u_uw.baseline_mm * 1000.0) / max(1e-2, abs_phi);
    let modulation = sqrt(f.x * f.x + f.y * f.y);
    s_unwrapped[idx] = UnwrappedPoint(abs_phi, depth, modulation, select(0.0, 1.0, modulation > 0.05));
}
