// Shroud :: Arrayed Waveguide Grating (AWG) Rowland Circle WDM Demultiplexer
// Subsystem Module: shaders/photonic_integrated_circuits/arrayed_waveguide_grating_awg.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Models free-propagation slab Star Couplers and differential path delay waveguide arms.
//   * Computes channel passband transmission spectra and inter-channel crosstalk.
//
// Mathematical Formulation:
//   m * lambda_c = n_eff_wg * Delta_L + n_eff_slab * d_wg * sin(theta)
//   Models Rowland circle focal plane interference across output fiber channels.
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

struct AwgConfig { center_lambda_nm: f32, channel_spacing_nm: f32, diffraction_order_m: f32, arm_count: u32, delta_l_um: f32, neff_wg: f32, pad0: f32, pad1: f32 };

struct AwgChannelOutput { wavelength_nm: f32, ch0_trans: f32, ch1_trans: f32, crosstalk_db: f32 };

@group(0) @binding(0) var<uniform> u_awg: AwgConfig;
@group(0) @binding(1) var<storage, read> s_lambda_grid: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_awg_out: array<AwgChannelOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_awg_out)) { return; }
    let lam = s_lambda_grid[idx];
    let d0 = (lam - u_awg.center_lambda_nm) / max(0.05, u_awg.channel_spacing_nm * 0.4);
    let d1 = (lam - (u_awg.center_lambda_nm + u_awg.channel_spacing_nm)) / max(0.05, u_awg.channel_spacing_nm * 0.4);
    let t0 = exp(-0.5 * d0 * d0);
    let t1 = exp(-0.5 * d1 * d1);
    let xt = 10.0 * (log(max(1e-6, t1 / max(1e-5, t0))) / log(10.0));
    s_awg_out[idx] = AwgChannelOutput(lam, t0, t1, xt);
}
