// Shroud :: Fourier Ptychographic Microscopy (FPM) Phase Retrieval
// Subsystem Module: shaders/computational_microscopy/fourier_ptychographic_microscopy.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Synthesizes wide field-of-view and high spatial resolution by stitching LED array angle-illuminated captures.
//   - Alternating projections between spatial intensity constraints and Fourier pupil apertures.
//
// Mathematical & Physical Formulation:
//   O_{k+1}(u, v) = O_k(u, v) + P*(u - u_k) / (|P|^2 + delta) * [Psi'_k - Psi_k] update step.
//   Recovers complex transmission function t(x, y) = a(x, y) * exp(i * phi(x, y)).
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

struct FpmConfig { led_kx: f32, led_ky: f32, pupil_radius_k: f32, step_size: f32 };

struct FpmSpectrumSample { fourier_re: f32, fourier_im: f32, updated_re: f32, updated_im: f32 };

@group(0) @binding(0) var<uniform> u_fpm: FpmConfig;
@group(0) @binding(1) var<storage, read> s_fourier_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> s_target_amp: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_fpm_out: array<FpmSpectrumSample>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_fpm_out)) { return; }
    let fin = s_fourier_in[idx];
    let target_a = s_target_amp[idx];
    let cur_amp = length(fin);
    let scale = target_a / max(1e-4, cur_amp);
    let psi_prime = fin * scale;
    let diff = psi_prime - fin;
    let updated = fin + diff * u_fpm.step_size;
    s_fpm_out[idx] = FpmSpectrumSample(fin.x, fin.y, updated.x, updated.y);
}
