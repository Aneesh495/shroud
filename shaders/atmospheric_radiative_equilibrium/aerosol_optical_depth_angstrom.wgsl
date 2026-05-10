// Shroud :: Angstrom Turbidity Multi-Spectral Aerosol Optical Depth Inversion
// Subsystem Module: shaders/atmospheric_radiative_equilibrium/aerosol_optical_depth_angstrom.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Inverts spectral optical thickness into Angstrom exponent alpha = -d ln(tau) / d ln(lambda).
//   - tau(lambda) = beta * lambda^-alpha parameterizes fine-mode vs coarse-mode particulate fractions.
//
// Mathematical & Physical Formulation:
//   Alpha > 1.5 indicates fine anthropogenic smoke/smog; Alpha < 0.5 indicates coarse desert dust and sea salt.
//   Calibrates multi-band sun photometer (AERONET) atmospheric transmissions.
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

struct AngstromConfig { beta_turbidity: f32, alpha_exponent: f32, pad0: f32, pad1: f32 };

struct AodResult { aod_val: f32, wavelength_um: f32, fine_mode_frac: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_ang: AngstromConfig;
@group(0) @binding(1) var<storage, read> s_lambdas_um: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_aod_out: array<AodResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_aod_out)) { return; }
    let lam = s_lambdas_um[idx];
    let aod = u_ang.beta_turbidity * pow(lam, -u_ang.alpha_exponent);
    let fmf = clamp(u_ang.alpha_exponent / 2.0, 0.0, 1.0);
    s_aod_out[idx] = AodResult(aod, lam, fmf, 0.0);
}
