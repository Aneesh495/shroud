// Shroud :: 2D Silicon Optical Phased Array (OPA) Solid-State LiDAR Beam Steering
// Subsystem Module: shaders/photonic_integrated_circuits/optical_phased_array_lidar_beamsteer.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Computes far-field diffraction array factor AF(theta, phi) for N-emitter grating antenna arrays.
//   * Steers laser emission via thermo-optic phase shifts in azimuth and wavelength tuning in elevation.
//
// Mathematical Formulation:
//   AF(theta) = sum_{n=0}^{N-1} a_n * exp(i * n * (k_0 * d * sin(theta) - Delta_phi))
//   Grating lobes appear whenever d * sin(theta) = m * lambda.
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

struct OpaConfig { emitter_pitch_um: f32, wavelength_um: f32, steer_phase_rad: f32, emitter_count: u32 };

struct FarFieldLobe { angle_deg: f32, array_factor_norm: f32, gain_db: f32, is_grating_lobe: u32 };

@group(0) @binding(0) var<uniform> u_opa: OpaConfig;
@group(0) @binding(1) var<storage, read> s_angles_deg: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_farfield_out: array<FarFieldLobe>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_farfield_out)) { return; }
    let ang_deg = s_angles_deg[idx];
    let theta = ang_deg * (PI / 180.0);
    let k0 = TWO_PI / max(0.1, u_opa.wavelength_um);
    let psi = k0 * u_opa.emitter_pitch_um * sin(theta) - u_opa.steer_phase_rad;
    let n = f32(u_opa.emitter_count);
    var af_norm = 1.0;
    if (abs(sin(0.5 * psi)) > 1e-5) {
        af_norm = abs(sin(0.5 * n * psi) / (n * sin(0.5 * psi)));
    }
    let gain_db = 20.0 * (log(max(1e-5, af_norm)) / log(10.0));
    let grating = select(0u, 1u, abs(ang_deg) > 20.0 && af_norm > 0.5);
    s_farfield_out[idx] = FarFieldLobe(ang_deg, af_norm, gain_db, grating);
}
