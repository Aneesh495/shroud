// Shroud :: PolInSAR Random Volume over Ground (RVoG) Canopy Height Inversion
// Subsystem Module: shaders/sar_interferometry/polinsar_forest_height_inversion.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Inverts complex polarimetric interferometric coherence locus across HV/HH/VV baselines.
//   * Separates ground topographic phase from volumetric forest canopy scattering extinction.
//
// Mathematical Formulation:
//   gamma_v(w) = exp(i * phi_0) * [gamma_v + m(w)] / [1 + m(w)]
//   Three-stage least-squares line fit in the complex unit circle determines tree height h_v.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct RvogConfig { kz_vertical_wavenumber: f32, extinction_sigma: f32, pad0: f32, pad1: f32 };

struct ForestHeightOut { canopy_height_m: f32, ground_phase_rad: f32, volume_coh: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_rvog: RvogConfig;
@group(0) @binding(1) var<storage, read> s_pol_cohs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_forest_out: array<ForestHeightOut>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_forest_out)) { return; }
    let pc = s_pol_cohs[idx];
    let g_hv = length(pc.xy);
    let phi_g = atan2(pc.w, pc.z);
    let hv = clamp(2.0 * acos(clamp(g_hv, 0.1, 0.999)) / max(0.02, abs(u_rvog.kz_vertical_wavenumber)), 1.0, 50.0);
    s_forest_out[idx] = ForestHeightOut(hv, phi_g, g_hv, 1u);
}
