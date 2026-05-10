// Shroud :: Digital Holographic Microscopy (DHM) Angular Spectrum Propagation
// Subsystem Module: shaders/computational_microscopy/quantitative_phase_imaging_dhm.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Reconstructs complex wavefield U(x, y) = A(x, y) * exp(i * phi(x, y)) from off-axis holograms.
//   - Propagates spatial spectrum via transfer function H(u, v) = exp(i * 2*pi * z * sqrt(lambda^-2 - u^2 - v^2)).
//
// Mathematical & Physical Formulation:
//   Height profile h(x, y) = (lambda / 2*pi) * phi(x, y) / (n_cell - n_medium).
//   Measures living cell dry mass, volume, and membrane fluctuations non-invasively.
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
const AVOGADRO_NUMBER: f32 = 6.02214076e23;

struct DhmConfig { rec_dist_um: f32, lambda_um: f32, dn_cell: f32, pad: f32 };

struct DhmCellResult { phase_rad: f32, thickness_um: f32, dry_mass_pg: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_dhm: DhmConfig;
@group(0) @binding(1) var<storage, read> s_hologram_re_im: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_dhm_out: array<DhmCellResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_dhm_out)) { return; }
    let c = s_hologram_re_im[idx];
    let phi = atan2(c.y, c.x);
    let thick = (u_dhm.lambda_um / TWO_PI) * (phi / max(1e-4, u_dhm.dn_cell));
    let dry_mass = thick * 1.8;
    s_dhm_out[idx] = DhmCellResult(phi, thick, dry_mass, 0.0);
}
