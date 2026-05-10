// Shroud :: Nomarski DIC Wollaston Shear Prism Wavefront Phase-to-Intensity
// Subsystem Module: shaders/computational_microscopy/differential_interference_contrast_dic.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates orthogonal polarization beam shearing by modified Wollaston / Nomarski prism.
//   - Converts spatial optical path length gradient d(OPD)/dx into high-contrast pseudo-3D relief shading.
//
// Mathematical & Physical Formulation:
//   I_dic(x) = I_0 * [1 - cos(Delta_phi(x) + theta_bias)] / 2 where Delta_phi = (2*pi/lambda) * d(OPD)/dx * Delta_x.
//   De Senarmont compensator adjusts optical bias phase theta_bias.
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

struct DicParams { shear_dist_um: f32, bias_rad: f32, wavelength_um: f32, pad: f32 };

struct DicPixel { dic_intensity: f32, opd_gradient: f32, phase_diff_rad: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_dic: DicParams;
@group(0) @binding(1) var<storage, read> s_phase_map_rad: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_dic_out: array<DicPixel>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_dic_out)) { return; }
    let p = s_phase_map_rad[idx];
    let p_next = s_phase_map_rad[min(idx + 1u, arrayLength(&s_phase_map_rad) - 1u)];
    let d_phi = p_next - p;
    let grad = d_phi / max(1e-3, u_dic.shear_dist_um);
    let intensity = 0.5 * (1.0 - cos(d_phi + u_dic.bias_rad));
    s_dic_out[idx] = DicPixel(intensity, grad, d_phi, 0.0);
}
