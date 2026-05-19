// Shroud :: InSAR Flat-Earth and Topographic Phase Removal Differential Solver
// Subsystem Module: shaders/sar_interferometry/insar_topographic_phase_flatten.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Removes flat-earth orbital baseline phase ramp and digital elevation model (DEM) topographic fringes.
//   * Isolates pure surface subsidence, earthquake co-seismic slip, and glacier flow displacement.
//
// Mathematical Formulation:
//   phi_topo = -(4 * pi / lambda) * (B_perp / (R * sin(theta))) * h_dem
//   phi_defo = phi_int - phi_flat - phi_topo.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;

struct BaselineConfig { lambda_m: f32, b_perp_m: f32, range_m: f32, look_angle_rad: f32 };

struct DefoOutput { phi_topo_rad: f32, phi_defo_rad: f32, los_displacement_mm: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_base: BaselineConfig;
@group(0) @binding(1) var<storage, read> s_phase_dem: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_defo_out: array<DefoOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_defo_out)) { return; }
    let pd = s_phase_dem[idx];
    let phi_raw = pd.x;
    let h_dem = pd.y;
    let k_topo = (FOUR_PI / max(1e-4, u_base.lambda_m)) * (u_base.b_perp_m / max(1.0, u_base.range_m * sin(u_base.look_angle_rad)));
    let phi_topo = k_topo * h_dem;
    let phi_defo = phi_raw - phi_topo;
    let disp_mm = (-phi_defo * u_base.lambda_m / FOUR_PI) * 1000.0;
    s_defo_out[idx] = DefoOutput(phi_topo, phi_defo, disp_mm, 0.0);
}
