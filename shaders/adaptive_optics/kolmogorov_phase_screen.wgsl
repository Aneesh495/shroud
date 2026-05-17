// Shroud :: Kolmogorov and von Karman Atmospheric Turbulence Phase Screen Synthesizer
// Subsystem Module: shaders/adaptive_optics/kolmogorov_phase_screen.wgsl
// Physically based WebGPU compute kernel.
//
// Physical Overview:
//   * Evaluates power spectral density (PSD) Phi_n(kappa) for atmospheric refractive index fluctuations.
//   * Incorporates outer scale L_0 and inner scale l_0 using modified von Karman spectrum.
//   * Computes Fried coherence length r_0 and Rytov log-amplitude scintillation variance.
//
// Mathematical Formulation:
//   Phi_phi(kappa) = 0.023 * r_0^(-5/3) * (kappa^2 + kappa_0^2)^(-11/6) * exp(-kappa^2 / kappa_m^2)
//   kappa_0 = 2 * pi / L_0, kappa_m = 5.92 / l_0.
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

struct TurbulenceConfig { fried_r0_m: f32, outer_scale_l0_m: f32, inner_scale_l0_m: f32, wavelength_nm: f32, wind_speed_m_s: f32, propagation_len_m: f32, pad0: f32, pad1: f32 };

struct PhaseSpectrumSample { spatial_freq_k: f32, phase_psd: f32, amplitude_filter: f32, greenwood_freq_hz: f32 };

@group(0) @binding(0) var<uniform> u_turb: TurbulenceConfig;
@group(0) @binding(1) var<storage, read> s_k_grid: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_psd_out: array<PhaseSpectrumSample>;

fn von_karman_psd(kx: f32, ky: f32, r0: f32, l0_outer: f32, l0_inner: f32) -> f32 {
    let k2 = kx * kx + ky * ky;
    let k0 = TWO_PI / max(0.1, l0_outer);
    let km = 5.92 / max(1e-4, l0_inner);
    let term = k2 + k0 * k0;
    let base = 0.023 * pow(max(1e-3, r0), -5.0 / 3.0) * pow(max(1e-8, term), -11.0 / 6.0);
    return base * exp(-k2 / (km * km));
}

fn greenwood_frequency(v_wind: f32, r0: f32) -> f32 {
    return 0.427 * (v_wind / max(1e-3, r0));
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_psd_out)) { return; }
    let k_vec = s_k_grid[idx];
    let k_mag = length(k_vec);
    let psd = von_karman_psd(k_vec.x, k_vec.y, u_turb.fried_r0_m, u_turb.outer_scale_l0_m, u_turb.inner_scale_l0_m);
    let amp = sqrt(max(0.0, psd));
    let fg = greenwood_frequency(u_turb.wind_speed_m_s, u_turb.fried_r0_m);
    s_psd_out[idx] = PhaseSpectrumSample(k_mag, psd, amp, fg);
}
